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


static void antiInsertConv(MLIRContext *context, Operation *f) {
  
  llvm::SmallVector<mlir::Operation *, 16> OpWorklist;

  // anti insert conv

  f->walk([&](mlir::Operation *op){ // all Ops
      OpWorklist.push_back(op);
  });


  for(auto it=OpWorklist.begin();it!=OpWorklist.end();it++){
    if(!dyn_cast<AtenConvolutionOp>(*it)) continue;
    AtenConvolutionOp convOp = dyn_cast<AtenConvolutionOp>(*it);
    Value convInput = convOp.getOperand(0);
    Value convKernel = convOp.getOperand(1);
    Value convBias = convOp.getOperand(2);
    Value convResult = convOp.getResult();
    // value convStride = convOp.getOperand(3);

    // Value convPadding = convOp.getOperand(4);

    auto convInputShape = convInput.getType().cast<ValueTensorType>().getSizes().vec();
    auto convKernelShape = convKernel.getType().cast<ValueTensorType>().getSizes().vec();
    auto convResultShape = convResult.getType().cast<ValueTensorType>().getSizes().vec();

    if(convInputShape==convResultShape){
        auto convKernelData = convKernel.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
        auto convBiasData = convBias.getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();

        /*auto convPaddingDataOp = convPadding.getDefiningOp<PrimListConstructOp>();
        int hPadding = convPaddingDataOp.getOperand(0).getDefiningOp<ConstantIntOp>().getValue().getSExtValue();
        int wPadding = convPaddingDataOp.getOperand(1).getDefiningOp<ConstantIntOp>().getValue().getSExtValue();*/

        // llvm::outs() << hPadding << ' ' << wPadding << "\n";
        
        int convChannelSize = convKernelShape[2] * convKernelShape[3];
        int convKernelSize = convKernelShape[1] * convChannelSize;

        // Is it an unitConv
        bool isUnitConv = true;
        for(auto i=0;i<convKernelShape[0];i++){
            auto base1 = i*convKernelSize;
            for(auto j=0;j<convKernelShape[1];j++){
                auto base2 = base1 + j * convChannelSize;
                for(auto k=0;k<convChannelSize;k++){
                    // if(j!=i || k != (hPadding * convKernelShape[3] + wPadding)){
                    // Note: j!=i， inputChannel-wise
                    if(j!=i || k != (convKernelShape[2]/2*convKernelShape[3] + convKernelShape[3]/2)){
                        if(convKernelData[base2+k]!=0.0){
                            isUnitConv = false;
                            break;
                        }
                    }
                    else{
                        if(convKernelData[base2+k]!=1.0){
                            isUnitConv = false;
                            break;
                        }
                    }
                }
                if(!isUnitConv) break;
            }
            if(!isUnitConv) break;
        }
        for (size_t i = 0; i < convBiasData.size(); ++i) {
            if (convBiasData[i] != 0.0) {
                isUnitConv = false;
                break;
            }
        }
        llvm::outs() << isUnitConv << "\n";

        if(isUnitConv){
            // """
            // insertConvsOp:
            //     reluOp  --> preOp

            //     insertUnsqueezeOp(optional)
        
            //     insertConvsOp
            //     insertReluOp  -->  preUseOp
                
            //     insertSqueezedimOp(optional) ( -->  preUseOp)

            //     nextOp

            // antiInsertConvsOp:
            //     reluOp
            //     nextOp

            // """

            auto preOp = convInput.getDefiningOp();
            bool squeezeFlag=false;
            while(dyn_cast<AtenUnsqueezeOp>(preOp)){
                squeezeFlag=true;
                preOp=preOp->getOperand(0).getDefiningOp();
            }
            auto convOp_userOps = convOp->getUses();
            auto it_2=convOp_userOps.begin();
            auto actfuncOp=it_2->getOwner(); // get Op: activation function Op

            auto usersOp=actfuncOp->getUses();
            auto it_tmp=usersOp.begin();
            auto tmpOp=it_tmp->getOwner();
            auto preUseOp=actfuncOp;
            if(squeezeFlag){
                while(dyn_cast<AtenSqueezeDimOp>(tmpOp)){
                    usersOp=tmpOp->getUses();
                    it_tmp=usersOp.begin();
                    preUseOp=tmpOp;
                    tmpOp=it_tmp->getOwner();
                }
            }
            // llvm::outs() << *preUseOp << "\n";
            while(it_tmp!=usersOp.end()){
                tmpOp=it_tmp->getOwner();
                tmpOp->replaceUsesOfWith(tmpOp->getOperand(0), preOp->getResult(0));

                usersOp = preUseOp->getUses();
                it_tmp=usersOp.begin(); // the next Op which use actfuncOp or squeezeDimOp
            }

            // process other Ops
            while(preUseOp!=preOp){
                if(!dyn_cast<AtenConvolutionOp>(preUseOp)){
                    tmpOp = preUseOp->getOperand(0).getDefiningOp();
                    preUseOp->erase();
                    preUseOp = tmpOp;
                }
                else{
                    tmpOp = preUseOp->getOperand(0).getDefiningOp();
                    auto tmpOp1 = preUseOp->getOperand(1).getDefiningOp();
                    auto tmpOp2 = preUseOp->getOperand(2).getDefiningOp();
                    auto tmpOp3 = preUseOp->getOperand(3).getDefiningOp();
                    auto tmpOp4 = preUseOp->getOperand(4).getDefiningOp();
                    // auto tmpOp5 = preUseOp->getOperand(5).getDefiningOp();
                    auto tmpOp6 = preUseOp->getOperand(6).getDefiningOp();
                    auto tmpOp7 = preUseOp->getOperand(7).getDefiningOp();
                    auto tmpOp8 = preUseOp->getOperand(8).getDefiningOp();
                    preUseOp->erase();
                    tmpOp1->erase();tmpOp2->erase();
                    tmpOp3->erase();tmpOp4->erase();
                    // tmpOp5->erase(); // TODO: 3, 5 are point to same Op  here
                    tmpOp6->erase();tmpOp7->erase();tmpOp8->erase();
                    preUseOp = tmpOp;
                }
            }
        }
    }
  }
  // erase the ops which difficultly process in the process
  OpWorklist.clear();
  f->walk([&](mlir::Operation *op){ // all Ops
      OpWorklist.push_back(op);
  });
  for(auto it_3=OpWorklist.begin();it_3!=OpWorklist.end();it_3++){
    auto op = *(it_3);
    if(isa<ConstantIntOp, PrimListConstructOp>(op)){
        auto usersOp = op->getUses();
        if(usersOp.begin()==usersOp.end()){
            op->erase();
        }
    }
  }
}


namespace {
class AntiInsertConvPass : public AntiInsertConvBase<AntiInsertConvPass> {
public:
  AntiInsertConvPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    antiInsertConv(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createAntiInsertConvPass() {
  return std::make_unique<AntiInsertConvPass>();
}