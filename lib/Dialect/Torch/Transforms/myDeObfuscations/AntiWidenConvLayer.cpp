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


static void antiWidenConvLayer(MLIRContext *context, Operation *f) {
  
  llvm::SmallPtrSet<mlir::Operation *, 16> ConvOpWorklist;
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  
  bool flag = false;
  f->walk([&](Operation *op) { // convOp op11 ... op1N conv2Op   conv3Op op21 ... op2N conv4Op  ...
    if (isa<AtenConvolutionOp>(op)) {
      flag = !flag;
      opWorklist.insert(op);
    } else if (flag) {
      opWorklist.insert(op);
    }
  });

  // anti WidenConvLayer
  f->walk([&](mlir::Operation *op){ // find all ConvolutionOp
    if(dyn_cast<AtenConvolutionOp>(op)){ 
      ConvOpWorklist.insert(op);
    }
  });
  
  auto it_tmp=opWorklist.begin(); // for process ops in the middlie of two convOps
  int N=0; // conv1 conv2 ... convN

  for(auto it=ConvOpWorklist.begin();it!=ConvOpWorklist.end();it++){
    // """                                                                    
    // totalKernelData: all data                                               
    // perKernelData: data of a kernel                                          
    // totalKernelData[i]: data of ith kernel                                    
    // mp<int, int>: first-->channel, second-->Number of this channel             
    //     example: 0 1 2 3 4 5 0 0 1 -->  0:3   1:2   2:1   3:1   4:1   5:1
    //              0 0 1 2 3 4 4 5 6 -->  0:2   1:1   2:1   3:1   4:2   5:1    6:1      
                                                                                 
    // convKernelSize: size of a kernel                                              
    // convChannelSize: size of a channel   
                                         
    // """ 
    AtenConvolutionOp conv1Op =  dyn_cast<AtenConvolutionOp>(*it);
    IRRewriter rewriter(context);
    rewriter.setInsertionPoint(conv1Op);
    Value conv1Kernel = conv1Op.getOperand(1);
    Value conv1Bias = conv1Op.getOperand(2);

    auto conv1KernelShape = conv1Kernel.getType().cast<ValueTensorType>().getSizes().vec();
    auto conv1BiasShape = conv1Bias.getType().cast<ValueTensorType>().getSizes().vec();

    // find same kernel
    std::vector<int> repeatKernel; // record the Serial Number(SN) of repeat kernel
    int repeatKernelSize=0;
    std::map<int, int> mp;
    std::vector<std::vector<float>> totalKernelData;
    int conv1KernelSize = conv1KernelShape[1] * conv1KernelShape[2] * conv1KernelShape[3];

    std::vector<float> conv1KernelData;
    std::vector<float> conv1BiasData;
    for(auto i:conv1Kernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>()){
      conv1KernelData.push_back(i);
    }
    for(auto i:conv1Bias.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>()){
      conv1BiasData.push_back(i);
    }
    std::vector<float> perKernelData;
    for(auto i=0;i<conv1KernelShape[0];i++){
      auto begin = i * conv1KernelSize;
      // auto end = begin + conv1KernelSize;
      for(auto count=0;count<conv1KernelSize;count++){
        perKernelData.push_back(conv1KernelData[begin+count]);
      }
      totalKernelData.push_back(perKernelData);
      perKernelData.clear();
    }

    // check same channel
    std::vector<float> newConv1KernelData;
    std::vector<bool> flag_1(conv1KernelShape[0], false);
    for(int i=0;i<conv1KernelShape[0];i++){
      if(flag_1[i]) continue;
      for(auto num:totalKernelData[i]) newConv1KernelData.push_back(num);
      mp[i]++;
      flag_1[i] = true;
      for(int j=i+1;j<conv1KernelShape[0];j++){
        if(flag_1[j]) continue;
        if(totalKernelData[j]==totalKernelData[i]){
          flag_1[j] = true;
          repeatKernel.push_back(j);
          repeatKernelSize++;
          mp[i]++;
        }
      }
    }

    if(!repeatKernel.empty()){

      // update kernel of conv1
      conv1KernelShape[0] = mp.size();
      auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(conv1KernelShape),
                                            rewriter.getF32Type());
      auto dense = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(conv1KernelShape), rewriter.getF32Type()),
          llvm::ArrayRef(newConv1KernelData));
      rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(conv1Kernel.getDefiningOp<ValueTensorLiteralOp>(),
                                                        resultTensorType, dense);
      std::vector<float> newConv1BiasData;
      for(int i=0;i<conv1BiasShape[0];i++){
        bool search = false;
        for(auto j=0;j<repeatKernelSize;j++){ // TODO: binSearch
          if(i==repeatKernel[j]){
            search = true;
            break;
          }
        }
        if(!search){
          newConv1BiasData.push_back(conv1BiasData[i]);
        }
      }

      // update bias of conv1
      conv1BiasShape[0] = conv1KernelShape[0];
      auto resultTensorType_2 = ValueTensorType::get(context, llvm::ArrayRef(conv1BiasShape),
                                              rewriter.getF32Type());
      auto dense_2 = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(conv1BiasShape), rewriter.getF32Type()),
          llvm::ArrayRef(newConv1BiasData));
      rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(conv1Bias.getDefiningOp<ValueTensorLiteralOp>(),
                                                        resultTensorType_2, dense_2);   

      // update ops in the between of conv(2N+1) and conv(2N+2)
      while(it_tmp!=opWorklist.end()) {
        if(dyn_cast<AtenConvolutionOp>(*it_tmp)){
          N++;
          if(N%2==0) break;
        }
        auto tmpOp = *it_tmp;
        if (ValueTensorType tensorTy =
                tmpOp->getResult(0).getType().dyn_cast<ValueTensorType>()) {
          auto tmpShape = tensorTy.getSizes().vec();
          tmpShape[1] = conv1KernelShape[0];
          resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(tmpShape),
                                                  rewriter.getF32Type());
          tmpOp->getResult(0).setType(resultTensorType);
        }
        it_tmp++;
      }
  
      // process conv2
      it++;
      AtenConvolutionOp conv2Op =  llvm::dyn_cast<AtenConvolutionOp>(*it);
      Value conv2Kernel = conv2Op.getOperand(1);
      auto conv2KernelShape = conv2Kernel.getType().cast<ValueTensorType>().getSizes().vec();
      
      std::vector<float> conv2KernelData;
      for(auto i:conv2Kernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>()){
        conv2KernelData.push_back(i);
      }
      std::vector<float> newConv2KernelData;
      int conv2ChannelSize = conv2KernelShape[2] *conv2KernelShape[3];
      int conv2KernelSize = conv2KernelShape[1] * conv2ChannelSize;

      // auto it_3=mp.begin(); // error
      for(int i=0;i<conv2KernelShape[0];i++){
        auto base = i * conv2KernelSize;
        auto it_3=mp.begin(); // this is correct position
        for(int j=0;j<conv2KernelShape[1];j++){
          auto begin = base + j * conv2ChannelSize;
          // end = begin + conv2ChannelSize;
          bool search_2 = false;
          for(auto k=0;k<repeatKernelSize;k++){ // TODO: binSearch
            if(j==repeatKernel[k]){
              search_2 = true;
              break;
            }
          }
          if(search_2) continue; // check next channel
          for(auto count=0;count<conv2ChannelSize;count++){
            newConv2KernelData.push_back(it_3->second * conv2KernelData[begin+count]);
          }
          it_3++;
        }
      }
      // update kernel of conv2
      conv2KernelShape[1]=conv1KernelShape[0];
      auto resultTensorType_3 = ValueTensorType::get(context, llvm::ArrayRef(conv2KernelShape),
                                          rewriter.getF32Type());
      auto dense_3 = DenseElementsAttr::get(
          RankedTensorType::get(llvm::ArrayRef(conv2KernelShape), rewriter.getF32Type()),
          llvm::ArrayRef(newConv2KernelData));
      rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(conv2Kernel.getDefiningOp<ValueTensorLiteralOp>(),
                                                    resultTensorType_3, dense_3);
    }
    // TODO: clear() ?
    // repeatKernel.clear();
    // mp.clear();
    // totalKernelData.clear();
    // conv1KernelData.clear();
    // conv1BiasData.clear();
    // newConv1KernelData.clear();
    // flag_1
    // newConv1BiasData.clear();
    // conv2KernelData.clear();
    // newConv2KernelData.clear();
  }
}


namespace {
class AntiWidenConvLayerPass : public AntiWidenConvLayerBase<AntiWidenConvLayerPass> {
public:
  AntiWidenConvLayerPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    antiWidenConvLayer(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createAntiWidenConvLayerPass() {
  return std::make_unique<AntiWidenConvLayerPass>();
}