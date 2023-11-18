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

static void widenConvLayer(MLIRContext *context, Operation *f) {
  llvm::outs() << "widenConvLayer start!\n";
  // widen convolution layer
  llvm::SmallVector<mlir::Operation*, 32> opWorklist;
  // bool flag = false;
  int convOpNum=0;
  std::map<Operation*, int> mp;
  int plcFlag = 0;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      if((op->getUses().begin()!=op->getUses().end()) && !(isa<PrimListConstructOp>(op->getUses().begin()->getOwner()))){
        mp[op] = convOpNum;
        convOpNum++;
      }
      else{
        plcFlag++;
      }
    }
    opWorklist.push_back(op);
  });
  if(plcFlag) convOpNum++;
  if(convOpNum<2){
    llvm::errs() << "convOpNum<2: can't widen!\n";
    return;
  }
  // select a random place to widen
  srand(unsigned(time(0)));
  auto it = (std::next(opWorklist.begin(), std::rand() % opWorklist.size()));
  while(!isa<AtenConvolutionOp>(*it)){
    it++;
    if(it==opWorklist.end() || (isa<AtenConvolutionOp>(*it) && mp[*it]==(convOpNum-1))){
      it = opWorklist.begin(); // last conv or no conv can't widen, widen first convOp
    }
  }
  if(((*it)->getUses().begin()!=(*it)->getUses().end()) && isa<PrimListConstructOp>((*it)->getUses().begin()->getOwner())){
    llvm::outs() << "jump this widenConvLayer!\n";
    return;
  }
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);

  // add a channel by copy existing channels each time
  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);
  auto oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
  auto oldBiasOp = oldBias.getDefiningOp<ValueTensorLiteralOp>();

  
  // widen conv kernel
  std::vector<float> kernelVec;
  for (auto i : oldKernelOp.getValue().getValues<float>()) {
    kernelVec.push_back(i);
  }
  // skip + widenlayer would cause error
  std::vector<float> zeroKernelVec(kernelVec.size(), 0);
  if(kernelVec==zeroKernelVec){
    llvm::outs() << "there is skipping of convOp, jump this widenConvLayer!\n";
    return;
  }

  // kernel layout is CCHW: new channels, old channels, height, width
  auto shape = oldKernel.getType().cast<ValueTensorType>().getSizes().vec();
  srand(unsigned(time(0)));
  auto copyPos = std::rand() % shape[0];
  shape[0] = shape[0] + 1;

  int channelSize = shape[1] * shape[2] * shape[3];
  kernelVec.insert(kernelVec.end(), kernelVec.begin() + copyPos * channelSize,
                   kernelVec.begin() + (copyPos + 1) * channelSize);
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                          rewriter.getF32Type());
  auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(kernelVec));
  rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldKernelOp,
                                                    resultTensorType, dense);

  // widen conv bias
  std::vector<float> biasVec;
  // is there better way to get the tensor data?
  for (auto i : oldBiasOp.getValue().getValues<float>()) {
    biasVec.push_back(i);
  }
  // shape of bias is C
  shape = oldBias.getType().cast<ValueTensorType>().getSizes().vec();
  shape[0] = shape[0] + 1;
  biasVec.push_back(biasVec[copyPos]);
  // create a constant tensor of float type by `shape` and `biasVec`
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                               rewriter.getF32Type());
  dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(biasVec));
  rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldBiasOp, resultTensorType,
                                                    dense);
  // modify ops between two conv according to new channel number
  if (ValueTensorType tensorTy =
          (*it)->getResult(0).getType().dyn_cast<ValueTensorType>()) {
    shape = tensorTy.getSizes().vec();
    shape[1] += 1;
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                            rewriter.getF32Type());
    (*it)->getResult(0).setType(resultTensorType);
  }
  it++; // jump first convOp
  // todo: line 113~203 may cause error
  for (; it != opWorklist.end(); it = std::next(it)) {
    // the second conv doesn't need change result shape
    // if (std::next(it) == opWorklist.end())
    //   break;
    
    auto tmpOp = *it;
    if(isa<AtenConvolutionOp>(tmpOp)) break;
    
    // 05.19: update tensor
    // llvm::outs() << "1111111111111111111111111\n";
    if(isa<ValueTensorLiteralOp>(tmpOp)){ // sometimes the next conVOp's kernel is also at here
      if((tmpOp->getUses().begin()!=tmpOp->getUses().end()) && isa<AtenConvolutionOp>(tmpOp->getUses().begin()->getOwner())) continue;
      // llvm::outs() << "1111111111111111222222222222222\n";
      auto tmpOpValue = tmpOp->getResult(0); // Operation* -> Value
      auto tmpShape = tmpOpValue.getType().cast<ValueTensorType>().getSizes().vec();
      auto tmpData = tmpOpValue.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
      std::vector<float> dataVec;
      // llvm::outs() << "222222222222222222222222222\n";
      if(tmpShape.size()==4){
        int tmpChannelSize =  tmpShape[2] * tmpShape[3];
        int tmpKernelSize = tmpShape[1] * tmpChannelSize;
        for(int i=0;i<tmpShape[0];i++){
          int begin = i * tmpKernelSize;
          for(int count=0;count<tmpKernelSize;count++){
            dataVec.push_back(tmpData[begin+count]);
          }
          // copy
          for(int count=0;count<tmpChannelSize;count++){
            dataVec.push_back(tmpData[begin+copyPos*tmpChannelSize+count]);
          }
        }
        tmpShape[1] +=1;
        // llvm::outs() << "333333333333333333333333333\n";
      }
      else if(tmpShape.size()==1){
        for(auto num : tmpData) dataVec.push_back(num);
        dataVec.push_back(tmpData[copyPos]);
        tmpShape[0]+=1;
      }
      // llvm::outs() << "44444444444444444444444444\n";
      auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(tmpShape),
                                          rewriter.getF32Type());
      auto dense = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(tmpShape), rewriter.getF32Type()),
          llvm::ArrayRef(dataVec));
      dataVec.clear();
      rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(tmpOp,
                                                    resultTensorType, dense);
    }
    else if (ValueTensorType tensorTy =
            tmpOp->getResult(0).getType().dyn_cast<ValueTensorType>()) {
      // llvm::outs() << "5555555555556666666666666666\n";
      shape = tensorTy.getSizes().vec();
      shape[1] += 1;
      resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                              rewriter.getF32Type());
      tmpOp->getResult(0).setType(resultTensorType);
    }
    // llvm::outs() << "66666666666666666\n";
  }

  // widen second conv kernel, no need to widen bias
  // llvm::outs() << "77777777777777777777\n";
  auto tempConvOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  if(tempConvOp->getUses().begin()!=tempConvOp->getUses().end() && isa<PrimListConstructOp>(tempConvOp->getUses().begin()->getOwner())){
    auto plcOp = tempConvOp->getUses().begin()->getOwner();
    auto opNum = plcOp->getNumOperands();
    for(size_t i=0;i<opNum;i++){
      convOp = plcOp->getOperand(i).getDefiningOp<AtenConvolutionOp>();
      oldKernel = convOp.getOperand(1);
      oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
      kernelVec.clear();
      for (auto i : oldKernelOp.getValue().getValues<float>()) {
        kernelVec.push_back(i);
      }
      // kernel shape is CCHW: new channels, old channels, height, width
      shape = oldKernel.getType().cast<ValueTensorType>().getSizes().vec();
      int hwSize = shape[2] * shape[3];
      channelSize = hwSize * shape[1];
      shape[1] = shape[1] + 1;
      std::vector<float> newKernelVec;
      for (int i = 0; i < shape[0]; i++) {
        int base = i * channelSize;
        for (int j = 0; j < hwSize; j++) {
          kernelVec[base + copyPos * hwSize + j] /= 2;
        }
        newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                            kernelVec.begin() + base + channelSize);
        // modify
        newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base + copyPos * hwSize,
                            kernelVec.begin() + base + (copyPos + 1) * hwSize);
      }
      resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                              rewriter.getF32Type());
      dense = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
          llvm::ArrayRef(newKernelVec));
      rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldKernelOp,
                                                        resultTensorType, dense);
    }
  }
  else{
    convOp = tempConvOp;
    oldKernel = convOp.getOperand(1);
    oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
    kernelVec.clear();
    for (auto i : oldKernelOp.getValue().getValues<float>()) {
      kernelVec.push_back(i);
    }
    // kernel shape is CCHW: new channels, old channels, height, width
    shape = oldKernel.getType().cast<ValueTensorType>().getSizes().vec();
    int hwSize = shape[2] * shape[3];
    channelSize = hwSize * shape[1];
    shape[1] = shape[1] + 1;
    std::vector<float> newKernelVec;
    for (int i = 0; i < shape[0]; i++) {
      int base = i * channelSize;
      for (int j = 0; j < hwSize; j++) {
        kernelVec[base + copyPos * hwSize + j] /= 2;
      }
      newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                          kernelVec.begin() + base + channelSize);
      // modify
      newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base + copyPos * hwSize,
                          kernelVec.begin() + base + (copyPos + 1) * hwSize);
    }
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                            rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
        llvm::ArrayRef(newKernelVec));
    rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldKernelOp,
                                                      resultTensorType, dense);
  }
  llvm::outs() << "widenConvLayer end!\n";
}

namespace {
class WidenConvLayerPass : public WidenConvLayerBase<WidenConvLayerPass> {
public:
  WidenConvLayerPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    widenConvLayer(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createWidenConvLayerPass() {
  return std::make_unique<WidenConvLayerPass>();
}