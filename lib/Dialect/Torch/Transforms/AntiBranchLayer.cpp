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

static void branchLayer(MLIRContext *context, Operation *f) {
  // this demo Compute convolution with merging same dimSize kernel

  llvm::SmallPtrSet<Operation *, 16> convOpWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
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
    auto conv1Input = convOp->getOperand(0);
    auto conv1Kernel = convOp->getOperand(1);
    auto conv1KernelShape = conv1Kernel.getType().cast<ValueTensorType>().getSizes().vec();
    auto conv1KernelData = conv1Kernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
    std::vector<float> newConvKernelData;
    for(auto num: conv1KernelData) newConvKernelData.push_back(num);
    auto conv1Bias = convOp->getOperand(2);
    auto conv1BiasShape = conv1Bias.getType().cast<ValueTensorType>().getSizes().vec();
    auto conv1BiasData = conv1Bias.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
    std::vector<float> newConvBiasData;
    for(auto num: conv1BiasData) newConvBiasData.push_back(num);
    auto conv1Stride = convOp->getOperand(3);
    auto conv1Padding = convOp->getOperand(4);
    auto conv1Dilation = convOp->getOperand(5);
    auto conv1Transposed = convOp->getOperand(6);
    auto conv1OutputPadding = convOp->getOperand(7);
    auto conv1Groups = convOp->getOperand(8);
    auto it_2=it;
    auto pre_it2=it_2;
    it_2++;
    bool together=true;
    int flag=1;
    while(together){
        if(dyn_cast<AtenConvolutionOp>(*it_2)){
            AtenConvolutionOp convOp = dyn_cast<AtenConvolutionOp>(*it_2);
            auto conv2Input = convOp->getOperand(0);
            auto conv2Kernel = convOp->getOperand(1);
            auto conv2KernelShape = conv2Kernel.getType().cast<ValueTensorType>().getSizes().vec();
            auto conv2KernelData = conv2Kernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
            auto conv2Bias = convOp->getOperand(2);
            auto conv2BiasShape = conv2Bias.getType().cast<ValueTensorType>().getSizes().vec();
            auto conv2BiasData = conv2Bias.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
            auto conv2Stride = convOp->getOperand(3);
            auto conv2Padding = convOp->getOperand(4);
            auto conv2Dilation = convOp->getOperand(5);
            auto conv2Transposed = convOp->getOperand(6);
            auto conv2OutputPadding = convOp->getOperand(7);
            auto conv2Groups = convOp->getOperand(8);
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
        else together=false;
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
            loc, convOp.getType(), conv1Input, convKernel, convBias, conv1Stride,
            conv1Padding, conv1Dilation, conv1Transposed,
            conv1OutputPadding, conv1Groups);

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
                i=i+flag-1;
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
            catOp->replaceUsesOfWith(catOp->getOperand(0),tensorList);
            tmpListOp->erase();
        }
    }
    it=pre_it2;
  }
}
    
    
 






    /*PrimListConstructOp listOp = catOp->getOperand(0); // get tensorList
    int numOfListOp = listOp->getNumOperands();
    llvm::SmallPtrSet<Operation *, 16> convOpWorklist;
    llvm::SmallPtrSet<Operation *, 16> othersOpWorklist;
    std::vector<int> convOperand;
    std::vector<int> othersOperand;
    for(int i=0;i<numOfListOp;i++){
        auto op = listOp->getOperand(i).getDefiningOp();
        if(isa<AtenConvolutionOp>(op)){
            // put the convOps which has same input together
            convOpOperand.push(i);
            convOpWorklist.insert(op);
        }
        else{
            othersOperand.push_back(i);
            othersOpWorklist.insert(op);
        }
    }
    for(auto it_2=convOpWorklist.begin();it_2!=convOpWorklist.end();it_2++){
        AtenConvolutionOp convOp=dyn_cast<AtenConvolutionOp>(*it_2);
        auto inputOp = convOp->getOperand(0).getDefiningOp();
        auto convKernel = convOp->getOperand(1);
        auto it_3 = it2;
        it_3++;
        AtenConvolutionOp convOp=dyn_cast<AtenConvolutionOp>(*it_3);
        if(inputOp == convOp->getOperand(0).getDefiningOp()){

        }
    }



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
  for(auto i=0;i<branch.size();i++){
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
  rewriter.replaceOp(convOp, concatOp);
  convKernelOp->erase();
}*/

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