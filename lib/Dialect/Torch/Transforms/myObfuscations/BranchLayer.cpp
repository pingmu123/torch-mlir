//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

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

static void branchLayer(MLIRContext *context, Operation *f) {
  // this demo Compute convolutions with kernel-wise

  llvm::SmallPtrSet<Operation *, 16> convOpWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      convOpWorklist.insert(op);
    }
  });
  if(convOpWorklist.empty()){
    llvm::errs() << "This NN model doesn't have ConvOp!\n";
    return;
  }

  // select a random place to branch
  Operation* convOp = *(std::next(convOpWorklist.begin(), std::rand() % convOpWorklist.size()));
  IRRewriter rewriter(context);
  rewriter.setInsertionPointAfter(convOp);
  Location loc = convOp->getLoc();

  // branch layer
  Value convKernel = convOp->getOperand(1);
  auto convKernelData = convKernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
  auto convKernelShape = convKernel.getType().cast<ValueTensorType>().getSizes().vec();
  Value convBias = convOp->getOperand(2);
  auto convBiasData = convBias.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
  auto convBiasShape = convBias.getType().cast<ValueTensorType>().getSizes().vec();

  // branch method
  std::vector<int> branch;
  int kernelNum = convKernelShape[0];
  int kernelSize = convKernelShape[1] * convKernelShape[2] *convKernelShape[3];
  int subKernelNum = 0;
  if(kernelNum==1){
    llvm::errs() << "There is only one kernel, return directly. \n";
    return;
  }
  while(kernelNum){
    while(subKernelNum==0) subKernelNum = std::rand() % kernelNum;
    branch.push_back(subKernelNum);
    kernelNum-=subKernelNum;
    if(kernelNum==1){
      branch.push_back(kernelNum);
      break;
    }
    subKernelNum=0; // reset
  }

  // create newOp
  llvm::SmallVector<Value, 16> convOpWorklist_2;
  int begin=0;
  for(auto i=0;i<(int)branch.size();i++){
    std::vector<float> subConvKernelData;
    std::vector<float> subConvBiasData;
    int begin1 = begin * kernelSize;
    int end1 = begin1 + branch[i]*kernelSize;
    int begin2 = begin;
    int end2 = begin2 + branch[i];
    for(int j=begin1;j<end1;j++) subConvKernelData.push_back(convKernelData[j]);
    for(int j=begin2;j<end2;j++) subConvBiasData.push_back(convBiasData[j]);

    auto tmpShape = convKernelShape;
    tmpShape[0] = branch[i];
    auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(tmpShape), rewriter.getF32Type()); 
    auto dense = DenseElementsAttr::get(RankedTensorType::get(llvm::ArrayRef(tmpShape),rewriter.getF32Type()), 
                                        llvm::ArrayRef(subConvKernelData)); 
    Value subKernel = rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

    tmpShape = convBiasShape;
    tmpShape[0] = branch[i];
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(tmpShape), rewriter.getF32Type()); 
    dense = DenseElementsAttr::get(RankedTensorType::get(llvm::ArrayRef(tmpShape), rewriter.getF32Type()),
                                    llvm::ArrayRef(subConvBiasData));
    Value subBias = rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

    Value tmpConv=rewriter.create<AtenConvolutionOp>( loc, convOp->getResult(0).getType(), convOp->getOperand(0), subKernel, subBias, 
                                                      convOp->getOperand(3), convOp->getOperand(4), convOp->getOperand(5), 
                                                      convOp->getOperand(6), convOp->getOperand(7), convOp->getOperand(8));
    // set result type
    ValueTensorType tensorTy = tmpConv.getType().dyn_cast<ValueTensorType>();
    tmpShape = tensorTy.getSizes().vec();
    tmpShape[1] = branch[i];
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(tmpShape), rewriter.getF32Type());
    tmpConv.setType(resultTensorType);

    convOpWorklist_2.push_back(tmpConv); // for concat
    begin+=branch[i];
  }

  // concat above convOpWorklist2
  mlir::ValueRange tensorList_vRange(convOpWorklist_2);
  Value tensorList= rewriter.create<PrimListConstructOp>(
      loc, ListType::get(ValueTensorType::getWithLeastStaticInformation(context)), // autoGet
      tensorList_vRange);

  Value dim = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value concatOp = rewriter.create<AtenCatOp>(loc, convOp->getResult(0).getType(), 
                                              tensorList, dim);
  auto convKernelOp = convKernel.getDefiningOp();
  auto convBiasOp = convBias.getDefiningOp();
  rewriter.replaceOp(convOp, concatOp);
  convKernelOp->erase();
  convBiasOp->erase();
}

namespace {
class BranchLayerPass : public BranchLayerBase<BranchLayerPass> {
public:
  BranchLayerPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    branchLayer(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createBranchLayerPass() {
  return std::make_unique<BranchLayerPass>();
}