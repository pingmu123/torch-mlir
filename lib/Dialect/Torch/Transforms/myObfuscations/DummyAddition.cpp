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


static void dummyAddition(MLIRContext *context, Operation *f) {

    // dummy addition
    llvm::SmallVector<mlir::Operation*, 32> reluOpWorklist;
    f->walk([&](mlir::Operation *op){ // find all ReluOp
      if(dyn_cast<AtenReluOp>(op)){ 
        reluOpWorklist.push_back(op);
      }
    });
    // int i=0;
    for(auto it=reluOpWorklist.begin();it!=reluOpWorklist.end();it++){
      AtenReluOp preReluOp =  dyn_cast<AtenReluOp>(*it);
      mlir::IRRewriter rewriter(context);
      rewriter.setInsertionPoint(preReluOp);
      auto loc = preReluOp.getLoc();

      /*
      step:

      (1) preRelu
      
      (2) create: newRelu (=preRelu)
          create: zeroTensor
          create: float1 
          create: addOp (newRelu + zeroTensor * float1)
          preRelu

      (3) // addOp replace preRelu, not erase preRelu
          newRelu (=preRelu)
          zeroTensor
          float1 
          addOp  

      */

      // create: newRelu (=preRelu)
      Value newRelu = rewriter.create<AtenReluOp>(loc, preReluOp.getType(), preReluOp.getOperand());

      // create an zeroTensor with same tensor after newReluOp
      Value opNum_0 = preReluOp.getOperand();
      auto shape = opNum_0.getType().cast<ValueTensorType>().getSizes().vec(); // torch.vtensor-->tensor-->shape-->shape(vector)
      int zeroTensorSize=1;
      for(auto i=0;i<shape.size();i++){
        zeroTensorSize*=shape[i];
      }
      std::vector<float> zeroVec(zeroTensorSize, 0.0);
      auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape), rewriter.getF32Type());
      auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(zeroVec));
      Value zeroTensor = rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);

      // the third parameter of AddTensorOp
      Value float1 = rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0));

      
      // dummyAddition
      Value dummyAddition = rewriter.create<AtenAddTensorOp>(loc, newRelu.getType(), // other direct methods to get this parameter?
                                                             newRelu, zeroTensor, float1);
      
      // replace
      rewriter.replaceOp(preReluOp, dummyAddition);

      // i++;
      // if(i==2) break;

    }
}


namespace {
class DummyAdditionPass : public DummyAdditionPassBase<DummyAdditionPass> {
public:
  DummyAdditionPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    dummyAddition(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createDummyAdditionPass() {
  return std::make_unique<DummyAdditionPass>();
}