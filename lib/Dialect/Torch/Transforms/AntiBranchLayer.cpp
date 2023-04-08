//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

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

static void antiBranchLayer(MLIRContext *context, Operation *f) {
  // this demo Compute convolution with merging same dimSize kernel

  llvm::SmallPtrSet<Operation *, 16> convOpWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) { // all convOp
      convOpWorklist.insert(op);
    }
  });
  for(auto it=convOpWorklist.begin();it!=convOpWorklist.end();it++){
    AtenConvolutionOp convOp = dyn_cast<AtenConvolutionOp>(*it);
    llvm::SmallPtrSet<Operation *, 16> processConvOplist;
    processConvOplist.insert(*it);
    mlir::IRRewriter rewriter(context);
    rewriter.setInsertionPoint(convOp);
    auto loc = convOp.getLoc();
    auto conv1Input = convOp.getOperand(0);
    auto conv1Kernel = convOp.getOperand(1);
    auto conv1KernelShape = conv1Kernel.getType().cast<ValueTensorType>().getSizes().vec();
    auto conv1KernelData = conv1Kernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
    std::vector<float> newConvKernelData;
    for(auto num: conv1KernelData) newConvKernelData.push_back(num);
    auto conv1Bias = convOp.getOperand(2);
    auto conv1BiasShape = conv1Bias.getType().cast<ValueTensorType>().getSizes().vec();
    auto conv1BiasData = conv1Bias.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
    std::vector<float> newConvBiasData;
    for(auto num: conv1BiasData) newConvBiasData.push_back(num);
    auto conv1Stride = convOp.getOperand(3);
    auto conv1Padding = convOp.getOperand(4);
    auto conv1Dilation = convOp.getOperand(5);
    auto conv1Transposed = convOp.getOperand(6);
    auto conv1OutputPadding = convOp.getOperand(7);
    auto conv1Groups = convOp.getOperand(8);
    auto it_2=it;
    auto pre_it2=it_2;
    it_2++;
    bool together=true;
    int flag=1;
    while(together){
        if(it_2!=convOpWorklist.end() && dyn_cast<AtenConvolutionOp>(*it_2)){ // if it==convOpWorklist.end() then dyn_cast<AtenxxxOp>(*it) is error.
            AtenConvolutionOp convOp = dyn_cast<AtenConvolutionOp>(*it_2);
            auto conv2Input = convOp.getOperand(0);
            auto conv2Kernel = convOp.getOperand(1);
            auto conv2KernelShape = conv2Kernel.getType().cast<ValueTensorType>().getSizes().vec();
            auto conv2KernelData = conv2Kernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
            auto conv2Bias = convOp.getOperand(2);
            auto conv2BiasShape = conv2Bias.getType().cast<ValueTensorType>().getSizes().vec();
            auto conv2BiasData = conv2Bias.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
            auto conv2Stride = convOp.getOperand(3);
            auto conv2Padding = convOp.getOperand(4);
            auto conv2Dilation = convOp.getOperand(5);
            auto conv2Transposed = convOp.getOperand(6);
            auto conv2OutputPadding = convOp.getOperand(7);
            auto conv2Groups = convOp.getOperand(8);
            if(conv1Input==conv2Input && conv1KernelShape[1]==conv2KernelShape[1] &&
            conv1KernelShape[2]==conv2KernelShape[2] && conv1KernelShape[3]==conv2KernelShape[3] &&
            conv1Stride==conv2Stride && conv1Padding==conv2Padding && conv1Dilation==conv2Dilation &&
            conv1Transposed==conv2Transposed && conv1OutputPadding==conv2OutputPadding &&
            conv1Groups==conv2Groups){
                processConvOplist.insert(*it_2);
                conv1KernelShape[0]+=conv2KernelShape[0];
                conv1BiasShape[0]+=conv2BiasShape[0];
                for(auto num: conv2KernelData) newConvKernelData.push_back(num);
                for(auto num:conv2BiasData) newConvBiasData.push_back(num);
                pre_it2=it_2;
                it_2++;

                flag++;
            }
            else{
                together=false;
            }
        }
        else{
           together=false;
        }
           
    }
    if(flag>1){

        // newConvKernel
        auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(conv1KernelShape),
                                            rewriter.getF32Type());
        auto dense = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(conv1KernelShape), rewriter.getF32Type()),
          llvm::ArrayRef(newConvKernelData));
        Value convKernel =
            rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

        // newConvBias
        resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(conv1BiasShape),
                                            rewriter.getF32Type());
        dense = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(conv1BiasShape), rewriter.getF32Type()),
          llvm::ArrayRef(newConvBiasData));
        Value convBias =
            rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
        

        // newConv
        Value newConv = rewriter.create<AtenConvolutionOp>(
            loc, convOp.getType(), conv1Input, convKernel, convBias, conv1Stride, // TODO: convOp.getType()
            conv1Padding, conv1Dilation, conv1Transposed,
            conv1OutputPadding, conv1Groups);

        // set result type
        ValueTensorType tensorTy = newConv.getType().dyn_cast<ValueTensorType>();
        auto newConvShape = tensorTy.getSizes().vec();
        newConvShape[1] = conv1KernelShape[0];
        resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(newConvShape), rewriter.getF32Type());
        newConv.setType(resultTensorType);

        // handle previous convOp
        auto convOp = dyn_cast<AtenConvolutionOp>(*it);
        auto primListOp = convOp->getUses().begin()->getOwner(); // get tensorList
        int sizeOfPrimList = primListOp->getNumOperands();
        auto it_3=processConvOplist.begin();
        llvm::SmallVector<Value, 16> convOpVector;
        for(int i=0;i<sizeOfPrimList;i++){ // todo: decrease cost
            Value tmpValue=primListOp->getOperand(i);
            auto tmpOp = dyn_cast<AtenConvolutionOp>(*it_3);
            if(tmpValue!=tmpOp->getResult(0)) convOpVector.push_back(tmpValue);
            else{
                convOpVector.push_back(newConv);
                i=i+flag-1; // BranchLayer must successive
            }
        }
        if(flag==sizeOfPrimList){
          auto catOp = primListOp->getUses().begin()->getOwner(); // get catOp
          auto postOp = catOp->getUses().begin()->getOwner(); // get Op which uses catOp
          postOp->replaceUsesOfWith(postOp->getOperand(0), newConv);
          // handle Ops
          catOp->erase();
          primListOp->erase();
        }
        else{
          mlir::ValueRange tensorList_vRange(convOpVector);
          Value tensorList= rewriter.create<PrimListConstructOp>(
            loc, ListType::get(ValueTensorType::getWithLeastStaticInformation(context)), // autoGet
            tensorList_vRange);
            auto catOp = primListOp->getUses().begin()->getOwner(); // get catOp
            auto tmpListOp=catOp->getOperand(0).getDefiningOp();
            catOp->replaceUsesOfWith(catOp->getOperand(0), tensorList);
            tmpListOp->erase();
        }
        // erase convOp(0) to convOp(flag-1)
        while(it_3!=processConvOplist.end()){
          auto tmpOp = *it_3;
          it_3++;
          auto tmpOp_kernelOp = tmpOp->getOperand(1).getDefiningOp();
          auto tmpOp_biasOp = tmpOp->getOperand(2).getDefiningOp();
          tmpOp->erase();
          tmpOp_kernelOp->erase();
          tmpOp_biasOp->erase();
        }
    }
    it=pre_it2;
  }
  // erase the ops which difficultly process in the process
  llvm::SmallPtrSet<mlir::Operation *, 16> OpWorklist;
  f->walk([&](mlir::Operation *op){ // all Ops
    OpWorklist.insert(op);
  });
  for(auto it_4=OpWorklist.begin();it_4!=OpWorklist.end();it_4++){
    auto op = *(it_4);
    if(dyn_cast<ConstantIntOp>(op)){
        auto usersOp = op->getUses();
        if(usersOp.begin()==usersOp.end()){
            op->erase();
        }
    }
  }
}
    

namespace {
class AntiBranchLayerPass : public AntiBranchLayerBase<AntiBranchLayerPass> {
public:
  AntiBranchLayerPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    antiBranchLayer(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createAntiBranchLayerPass() {
  return std::make_unique<AntiBranchLayerPass>();
}