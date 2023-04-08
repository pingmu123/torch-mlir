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


static void antiDummyAddition(MLIRContext *context, Operation *f) {

  llvm::SmallPtrSet<mlir::Operation *, 16> AddTensorOpWorklist;

  // anti dummy addition
  
  f->walk([&](mlir::Operation *op){ // find all AddTensorOp
    if(dyn_cast<AtenAddTensorOp>(op)){ 
      AddTensorOpWorklist.insert(op);
    }
  });

  // X + 0 --> X
  for(auto it=AddTensorOpWorklist.begin();it!=AddTensorOpWorklist.end();it++){
    AtenAddTensorOp addTensorOp =  dyn_cast<AtenAddTensorOp>(*it);
    
    // Whether operand 1 or operand 2 is zero
    Value opNum_1 = addTensorOp.getOperand(1);
    Value opNum_2 = addTensorOp.getOperand(2);
    auto opNum_1Data = opNum_1.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
    auto opNum_2Data = opNum_2.getDefiningOp<ConstantFloatOp>().getValue().convertToDouble();
    bool isEqual = true;
    if(opNum_2Data!=0.0){
        for (size_t i = 0; i < opNum_1Data.size(); ++i) {
        if (opNum_1Data[i] != 0.0) {
          isEqual = false;
          break;
        }
      }
    }

    // it is X + 0 
    if(isEqual){
      // handle the Ops related to this addTensorOp
      auto userOps = addTensorOp->getUses(); // get OpOperand(s)
      auto it=userOps.begin();
      while(it!=userOps.end()){
        auto tmpOp=it->getOwner();
        tmpOp->replaceUsesOfWith(tmpOp->getOperand(0), addTensorOp->getOperand(0));

        userOps = addTensorOp->getUses(); 
        it=userOps.begin(); // the next Op which use addTensorOp
      }

      // process addTensorOp
      auto preZeroTensorOp = addTensorOp.getOperand(1).getDefiningOp(); // ValueTensorLiteralOp
      auto preFloat1Op = addTensorOp.getOperand(2).getDefiningOp(); // ConstantFloatOp
      addTensorOp->erase();
      preZeroTensorOp->erase();
      preFloat1Op->erase();
    } // end of handle the Ops related to this addTensorOp
  }
}


namespace {
class AntiDummyAdditionPass : public AntiDummyAdditionPassBase<AntiDummyAdditionPass> {
public:
  AntiDummyAdditionPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    antiDummyAddition(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createAntiDummyAdditionPass() {
  return std::make_unique<AntiDummyAdditionPass>();
}