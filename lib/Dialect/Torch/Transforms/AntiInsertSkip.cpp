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


static void antiInsertSkip(MLIRContext *context, Operation *f) {
  
  llvm::SmallPtrSet<mlir::Operation *, 16> ConvOpWorklist;

  // anti insert skip(conv)

  f->walk([&](mlir::Operation *op){ // find all AddTensorOp
    if(dyn_cast<AtenConvolutionOp>(op)){ 
      ConvOpWorklist.insert(op);
    }
  });
  // X + skip(0) --> X
  for(auto it=ConvOpWorklist.begin();it!=ConvOpWorklist.end();it++){
    AtenConvolutionOp convOp =  dyn_cast<AtenConvolutionOp>(*it);
    Value opNum_1 = convOp.getOperand(1); // conv kernel
    Value opNum_2 = convOp.getOperand(2); // conv bias

    auto opNum_1Data = opNum_1.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
    auto opNum_2Data = opNum_2.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
    
    // Is it inserting a conv skip
    bool isEqual = true;
    for (size_t i = 0; i < opNum_1Data.size(); ++i) {
      if (opNum_1Data[i] != 0) {
        isEqual = false;
        break;
      }
    }
    for (size_t i = 0; i < opNum_2Data.size(); ++i) {
      if (opNum_2Data[i] != 0) {
        isEqual = false;
        break;
      }
    }
    if(isEqual){
      // handle the Ops related to this convOp： it is always an AddTensorOp
      auto convOp_userOps = convOp->getUses(); // get OpOperand
      auto addTensorOp = convOp_userOps.begin()->getOwner(); // get addTensorOp
      // handle the Ops related to this addTensorOp
      auto addTensorOp_userOps = addTensorOp->getUses();
      for (auto it=addTensorOp_userOps.begin();it!=addTensorOp_userOps.end();it++){
        auto tmpOp=it->getOwner(); // get Op 
        tmpOp->replaceUsesOfWith(tmpOp->getOperand(0), tmpOp->getOperand(0).getDefiningOp()->getOperand(0));
        
        auto preFloat1Op = addTensorOp->getOperand(2).getDefiningOp(); // ConstantFloatOp: we can delete it directly here
      
        addTensorOp->erase();
        preFloat1Op->erase();

        auto skipConvOpNum1Op=opNum_1.getDefiningOp();
        auto skipConvOpNum2Op=opNum_2.getDefiningOp();

        convOp->erase();
        skipConvOpNum1Op->erase();
        skipConvOpNum2Op->erase();

      } // end of handle the Ops related to this addTensorOp

    } // end of handle the Ops related to this convOp
  }
}


namespace {
class AntiInsertSkipPass : public AntiInsertSkipBase<AntiInsertSkipPass> {
public:
  AntiInsertSkipPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    antiInsertSkip(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createAntiInsertSkipPass() {
  return std::make_unique<AntiInsertSkipPass>();
}