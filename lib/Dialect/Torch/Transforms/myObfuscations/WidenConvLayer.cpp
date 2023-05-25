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
  // widen convolution layer
  llvm::SmallVector<mlir::Operation*, 32> opWorklist;
  // bool flag = false;
  int convOpNum=0;
  std::map<Operation*, int> mp;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      mp[op] = convOpNum;
      convOpNum++;
    }
    opWorklist.push_back(op);
  });
  if(convOpNum<2){
    llvm::errs() << "convOpNum<2: can't widen!\n";
    return;
  }
  // select a random place to wide
  srand(unsigned(time(0)));
  auto it = (std::next(opWorklist.begin(), std::rand() % opWorklist.size()));
  while(!isa<AtenConvolutionOp>(*it)){
    it++;
    if((isa<AtenConvolutionOp>(*it) && mp[*it]==convOpNum)||it==opWorklist.end()){
      it = opWorklist.begin(); // last conv or no conv can't widen, widen first convOp
    }
  }
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);

  // add three channels by copy existing channels, two channel 0 and one
  // channel 1
  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);
  auto oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
  auto oldBiasOp = oldBias.getDefiningOp<ValueTensorLiteralOp>();

  // widen conv bias
  std::vector<float> biasVec;
  // is there better way to get the tensor data?
  for (auto i : oldBiasOp.getValue().getValues<float>()) {
    biasVec.push_back(i);
  }
  // shape of bias is C
  auto shape = oldBias.getType().cast<ValueTensorType>().getSizes().vec();
  shape[0] = shape[0] + 3;
  biasVec.push_back(biasVec[0]);
  biasVec.push_back(biasVec[0]);
  biasVec.push_back(biasVec[1]);
  // create a constant tensor of float type by `shape` and `biasVec`
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                               rewriter.getF32Type());
  auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(biasVec));
  rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldBiasOp, resultTensorType,
                                                    dense);
  // widen conv kernel
  std::vector<float> kernelVec;
  for (auto i : oldKernelOp.getValue().getValues<float>()) {
    kernelVec.push_back(i);
  }
  // kernel layout is CCHW: new channels, old channels, height, width
  shape = oldKernel.getType().cast<ValueTensorType>().getSizes().vec();
  shape[0] = shape[0] + 3;

  int channelSize = shape[1] * shape[2] * shape[3];
  kernelVec.insert(kernelVec.end(), kernelVec.begin(),
                   kernelVec.begin() + channelSize);
  kernelVec.insert(kernelVec.end(), kernelVec.begin(),
                   kernelVec.begin() + channelSize);
  kernelVec.insert(kernelVec.end(), kernelVec.begin() + channelSize,
                   kernelVec.begin() + 2 * channelSize);
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                          rewriter.getF32Type());
  dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(kernelVec));
  rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldKernelOp,
                                                    resultTensorType, dense);

  // modify ops between two conv according to new channel number
  if (ValueTensorType tensorTy =
          (*it)->getResult(0).getType().dyn_cast<ValueTensorType>()) {
    shape = tensorTy.getSizes().vec();
    shape[1] += 3;
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                            rewriter.getF32Type());
    (*it)->getResult(0).setType(resultTensorType);
  }
  it++; // jump first convOp
  for (; it != opWorklist.end(); it = std::next(it)) {
    // the second conv doesn't need change result shape
    // if (std::next(it) == opWorklist.end())
    //   break;
    
    auto tmpOp = *it;
    if(isa<AtenConvolutionOp>(tmpOp)) break;
    
    // 05.19: update tensor
    if(isa<ValueTensorLiteralOp>(tmpOp)){ // sometimes the next conVOp's kernel is also at here
      if(isa<AtenConvolutionOp>(tmpOp->getUses().begin()->getOwner())) continue;
      auto tmpOpValue = tmpOp->getResult(0); // Operation* -> Value
      auto tmpShape = tmpOpValue.getType().cast<ValueTensorType>().getSizes().vec();
      auto tmpData = tmpOpValue.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
      std::vector<float> dataVec;
      if(tmpShape.size()==4){
        int tmpChannelSize =  tmpShape[2] * tmpShape[3];
        int tmpKernelSize = tmpShape[1] * tmpChannelSize;
        for(int i=0;i<tmpShape[0];i++){
          int begin1=i*tmpKernelSize;
          for(int count=0;count<tmpKernelSize;count++){
            dataVec.push_back(tmpData[begin1+count]);
          }
          // copy
          for(int count=0;count<tmpChannelSize;count++){
            dataVec.push_back(tmpData[begin1+count]);
          }
          for(int count=0;count<tmpChannelSize;count++){
            dataVec.push_back(tmpData[begin1+count]);
          }
          int begin2 = begin1 + tmpChannelSize;
          for(int count=0;count<tmpChannelSize;count++){
            dataVec.push_back(tmpData[begin2+count]);
          }
        }
        tmpShape[1] +=3;
      }
      if(tmpShape.size()==1){
        for(auto num : tmpData) dataVec.push_back(num);
        dataVec.push_back(tmpData[0]);
        dataVec.push_back(tmpData[0]);
        dataVec.push_back(tmpData[0]);
        tmpShape[0]+=3;
      }
      auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(tmpShape),
                                          rewriter.getF32Type());
      auto dense = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(tmpShape), rewriter.getF32Type()),
          llvm::ArrayRef(dataVec));
      rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(tmpOp,
                                                    resultTensorType, dense);
    }
    else if (ValueTensorType tensorTy =
            tmpOp->getResult(0).getType().dyn_cast<ValueTensorType>()) {
      shape = tensorTy.getSizes().vec();
      shape[1] += 3;
      resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                              rewriter.getF32Type());
      tmpOp->getResult(0).setType(resultTensorType);
    }
  }

  // widen second conv kernel, no need to widen bias
  convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  llvm::outs() << *convOp << "\n";
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
  shape[1] = shape[1] + 3;
  for(auto num: shape) llvm::outs() << num << " ";
  llvm::outs() << "\n";
  std::vector<float> newKernelVec;
  for (int i = 0; i < shape[0]; i++) {
    int base = i * channelSize;
    for (int j = 0; j < hwSize; j++) {
      kernelVec[base + j] /= 3;
      kernelVec[base + hwSize + j] /= 2;
    }
    newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                        kernelVec.begin() + base + channelSize);
    newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                        kernelVec.begin() + base + hwSize);
    newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                        kernelVec.begin() + base + hwSize);
    newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base + hwSize,
                        kernelVec.begin() + base + 2 * hwSize);
  }
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                          rewriter.getF32Type());
  dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(newKernelVec));
  rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldKernelOp,
                                                    resultTensorType, dense);
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