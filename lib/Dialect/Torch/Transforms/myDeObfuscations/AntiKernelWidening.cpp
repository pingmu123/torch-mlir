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

static void antiKernelWidening(MLIRContext *context, Operation *f) {
  // Delete the zeroPadding of kernel
  llvm::outs() << "AKW start!\n";
  llvm::SmallPtrSet<Operation *, 16> convOpWorklist;
  f->walk([&](Operation *op) {
    if(isa<AtenConvolutionOp>(op)) {
      convOpWorklist.insert(op);
    }
  });
  if (convOpWorklist.empty()) {
    llvm::errs() << "No convKernel so no padding!\n";
    return;
  }
  for(auto it=convOpWorklist.begin();it!=convOpWorklist.end();it++){
    AtenConvolutionOp convOp =  dyn_cast<AtenConvolutionOp>(*it);
    IRRewriter rewriter(context);
    rewriter.setInsertionPoint(convOp);
    auto loc = convOp.getLoc();

    auto convKernel = convOp.getOperand(1);

    auto convKernelShape = convKernel.getType().cast<ValueTensorType>().getSizes().vec();
    auto convKernelData = convKernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();

    // check and get zeroPaddingNum
    int zeroPaddingNum=0;
    int height =  convKernelShape[2];
    int width = convKernelShape[3];
    std::vector<float> perWidthData;
    std::vector<float> zeroWidthData(width, 0.0);
    for(int i=0;i<height;i++){
      int begin = i * width;
      int end = (i+1) * width;
      for(int j=begin;j<end;j++){
        perWidthData.push_back(convKernelData[j]);
      }
      if(perWidthData==zeroWidthData){
        zeroPaddingNum++;
        perWidthData.clear();
      }
      else break;
    }
    if(zeroPaddingNum>0 && (2*zeroPaddingNum)<height){ // solve skip convOp error

      // change convPadding
      auto convPadding = convOp.getOperand(4);
      auto convPaddingDataOp = convPadding.getDefiningOp<PrimListConstructOp>();
      int hPadding = convPaddingDataOp.getOperand(0).getDefiningOp<ConstantIntOp>().getValue().getSExtValue();
      int wPadding = convPaddingDataOp.getOperand(1).getDefiningOp<ConstantIntOp>().getValue().getSExtValue();
      hPadding-=zeroPaddingNum;
      wPadding-=zeroPaddingNum;
      Value intHeightPad =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(hPadding));
      Value intWidthPad =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(wPadding));
      Value listIntPad_Pad = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(IntType::get(context)),
          ValueRange({intHeightPad, intWidthPad}));
      // auto convKernelPaddingOp = convPadding.getDefiningOp();
      convOp.getResult().getDefiningOp()->replaceUsesOfWith(convOp.getOperand(4), listIntPad_Pad);


      // change convKernel
      std::vector<float> newConvKernelData;
      int channelSize = height * width;
      int kernelSize = convKernelShape[1] * channelSize; 
      
      for(int i=0;i<convKernelShape[0];i++){
        int base1 = i * kernelSize;
        for(int j=0;j<convKernelShape[1];j++){
          int base2 = base1 + j * channelSize;
          for(int k = zeroPaddingNum; k < height - zeroPaddingNum; k++){
            int base3 = base2 + k * convKernelShape[3] + zeroPaddingNum;
            int count=0;
            while(count < width - 2 * zeroPaddingNum){
              newConvKernelData.push_back(convKernelData[base3+count]);
              count++;
            }
          }
        }
      }
      auto newConvKernelShape = convKernelShape;
      newConvKernelShape[2] = convKernelShape[2] - 2 * zeroPaddingNum;
      newConvKernelShape[3] = convKernelShape[3] - 2 * zeroPaddingNum;

      auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(newConvKernelShape), rewriter.getF32Type());

      auto dense = DenseElementsAttr::get(RankedTensorType::get(llvm::ArrayRef(newConvKernelShape), rewriter.getF32Type()), 
                                          llvm::ArrayRef(newConvKernelData)); 
      Value newKernel = rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

      // auto convKernelOp = convKernel.getDefiningOp();
      convOp.getResult().getDefiningOp()->replaceUsesOfWith(convOp.getOperand(1), newKernel);

      } // end of process zeroPadding
    }

  // delete ops which are unused (produced by pass)
  llvm::SmallPtrSet<mlir::Operation *, 16> OpWorklist;
  f->walk([&](mlir::Operation *op){ // all Ops
  OpWorklist.insert(op);
  });
  for(auto it=OpWorklist.begin();it!=OpWorklist.end();it++){
    auto op = *(it);
    if(isa<ConstantIntOp, PrimListConstructOp, ValueTensorLiteralOp>(op)){
        auto usersOp = op->getUses();
        if(usersOp.begin()==usersOp.end()){
            op->erase();
        }
    }
  }
}

namespace {
class AntiKernelWideningPass : public AntiKernelWideningBase<AntiKernelWideningPass> {
public:
  AntiKernelWideningPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    antiKernelWidening(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createAntiKernelWideningPass() {
  return std::make_unique<AntiKernelWideningPass>();
}