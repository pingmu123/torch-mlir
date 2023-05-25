//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include <cstdlib>
#include <ctime>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static void kernelWidening(MLIRContext *context, Operation *f) {
  // Use zero to pad kernel
  llvm::SmallVector<mlir::Operation*, 32> convOpWorklist;
  f->walk([&](Operation *op) {
    if(isa<AtenConvolutionOp>(op)) {
      convOpWorklist.push_back(op);
    }
  });
  if (convOpWorklist.empty()) {
    llvm::errs() << "No convKernel to pad!\n";
    return;
  }
  for(auto it=convOpWorklist.begin();it!=convOpWorklist.end();it++){
    srand(unsigned(time(0)));
    int jump = std::rand();
    if(jump%2==0) break; // randomly jump this convOp
    int padNum = std::rand() % 5; // we assume hPadding is equal to wPadding and padNum < 5
    AtenConvolutionOp convOp =  dyn_cast<AtenConvolutionOp>(*it);
    IRRewriter rewriter(context);
    rewriter.setInsertionPoint(convOp);
    auto loc = convOp.getLoc();

    auto convKernel = convOp.getOperand(1);

    auto convKernelShape = convKernel.getType().cast<ValueTensorType>().getSizes().vec();
    auto convKernelData = convKernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();

    std::vector<float> kernelPerWidthData;
    std::vector<std::vector<float>> newKernelData;
    std::vector<float> zeroPerWidth(convKernelShape[3]+2*padNum, 0);
    int channelSize = convKernelShape[2] * convKernelShape[3];
    int kernelSize = convKernelShape[1] * channelSize; 
    
    for(int i=0;i<convKernelShape[0];i++){
      int base1 = i * kernelSize;
      for(int j=0;j<convKernelShape[1];j++){
        int base2 = base1 + j * channelSize;
        for(int padCount=0;padCount<padNum;padCount++) newKernelData.push_back(zeroPerWidth);
        for(int k=0;k<convKernelShape[2];k++){
          int base3 = base2 + k * convKernelShape[3];
          int count=0;
          while(count<convKernelShape[3]){
            kernelPerWidthData.push_back(convKernelData[base3+count]);
            count++;
          }
          for(int padCount=0;padCount<padNum;padCount++){
            kernelPerWidthData.insert(kernelPerWidthData.begin(), 0); // begin
            kernelPerWidthData.insert(kernelPerWidthData.end(), 0); // end
          }
          newKernelData.push_back(kernelPerWidthData);
          kernelPerWidthData.clear(); // reset
        }
        for(int padCount=0;padCount<padNum;padCount++) newKernelData.push_back(zeroPerWidth);
      }
    }
    auto newConvKernelShape = convKernelShape;
    newConvKernelShape[2] = newConvKernelShape[2] + 2 * padNum;
    newConvKernelShape[3] = newConvKernelShape[3] + 2 * padNum;

    auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(newConvKernelShape), rewriter.getF32Type());


    // std::vector<std::vector<float>>  -->   std::vector<float>
    std::vector<float> newKernelDataToVec;
    for(int i=0;i<(int)newKernelData.size();i++){
      for(auto num: newKernelData[i]) newKernelDataToVec.push_back(num);
    }
    auto dense = DenseElementsAttr::get(RankedTensorType::get(llvm::ArrayRef(newConvKernelShape), rewriter.getF32Type()), 
                                        llvm::ArrayRef(newKernelDataToVec)); 
    Value newKernel = rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

    // auto convKernelOp = convKernel.getDefiningOp();
    convOp.getResult().getDefiningOp()->replaceUsesOfWith(convOp.getOperand(1), newKernel);

    // pad input
    auto convPadding = convOp.getOperand(4);
    auto convPaddingDataOp = convPadding.getDefiningOp<PrimListConstructOp>();
    int hPadding = convPaddingDataOp.getOperand(0).getDefiningOp<ConstantIntOp>().getValue().getSExtValue();
    int wPadding = convPaddingDataOp.getOperand(1).getDefiningOp<ConstantIntOp>().getValue().getSExtValue();
    hPadding+=padNum;
    wPadding+=padNum;
    Value intHeightPad =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(hPadding));
    Value intWidthPad =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(wPadding));

    Value listIntPad_Pad = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)),
        ValueRange({intHeightPad, intWidthPad}));
    // auto convKernelPaddingOp = convPadding.getDefiningOp();
    convOp.getResult().getDefiningOp()->replaceUsesOfWith(convOp.getOperand(4), listIntPad_Pad);
  } // end 
  // todo: Dilation?

}

namespace {
class KernelWideningPass : public KernelWideningBase<KernelWideningPass> {
public:
  KernelWideningPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    kernelWidening(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createKernelWideningPass() {
  return std::make_unique<KernelWideningPass>();
}