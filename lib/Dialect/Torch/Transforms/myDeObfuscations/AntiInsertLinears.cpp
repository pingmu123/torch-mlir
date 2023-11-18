//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include <unordered_set>

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
using namespace std;


// matAdd
vector<vector<float>> matAdd(vector<vector<float>> A, vector<vector<float>> B){
    // We assume A+B is allways ok  
    int m = A.size();
    int n = A[0].size();
    vector<float> col(n, 0);
    vector<vector<float>> res(m, col);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            res[i][j]=A[i][j]+B[i][j];
        }
    }
    return res;
}

// matMul
vector<vector<float>> matMul(vector<vector<float>> A, vector<vector<float>> B){
    // We assume that A*B is allways ok  
    int m = A.size();
    int n = A[0].size();
    int s = B[0].size();
    vector<float> col(s, 0);
    vector<vector<float>> res(m, col);
    for(int k=0;k<s;k++){
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                res[i][j]+=A[i][k]*B[k][j];
            }
        }
    }
    return res;
}

// flatten
vector<float> flatten(vector<vector<float>> A){
    vector<float> res;
    for(size_t i=0;i<A.size();i++){
        for(size_t j=0;j<A[0].size();j++){
            res.push_back(A[i][j]);
        }
    }
    return res;
}

// toMatrix
vector<vector<float>> toMatrix(vector<float> A, int m, int n){
    vector<float> col;
    vector<vector<float>> res;
    if(A.size()!= m*n){
        llvm::outs() << "error: matrix size is not match!\n";
        return res;
    }
    for(int i=0;i<m*n;i++){
        col.push_back(A[i]);
        if(i%n==(n-1)){
            res.push_back(col);
            col.clear();
        }
    }
    return res;
}

bool isEqual(vector<float>& A, vector<float>& B){
    bool flag=true;
    for(int i=0;i<int(A.size());i++){
        // if(i<15) llvm::outs() << A[i] << " " << B[i] << "\n";
        if(abs(A[i]-B[i])>1e-3){ // 1e-3
            flag=false;
            break;
        }
    }
    return flag;
}

static void antiInsertLinear(MLIRContext *context, Operation *f) {
  
  llvm::SmallVector<mlir::Operation*, 32> opWorklist;
  f->walk([&](mlir::Operation* op){ // all Ops
    opWorklist.push_back(op);
  });

  // anti insert linear
  llvm::outs() << "AIL start!\n";
  vector<vector<int>> mmOpWeightShape;
  vector<vector<float>> mmOpWeightData;
  vector<vector<int>> mmOpBiasShape;
  vector<vector<float>> mmOpBiasData;
  llvm::SmallVector<mlir::Operation *, 16> mmOpWorklist;
  llvm::SmallVector<mlir::Operation *, 16> mmBiasOpWorklist;
  int numOfOp=0; // mmOp + addTensorOp count
  llvm::outs() << "1111111111111111111111111111!\n";
  for(auto it=opWorklist.begin();it!=opWorklist.end();it++){
    if(isa<AtenAddTensorOp>(*it)){
        // mmOp + addTensorOp(mmBiasOp)
        auto op = *it;
        auto mmOp = op->getOperand(0).getDefiningOp();
        // it is error when the first parapeter is not Op(origin input)
        // AntiInsertLinearPass should be behind of AntiInsertSkipPass
        llvm::outs() << "1.1!\n";
        // llvm::outs() << *mmOp << "\n";
        if(mmOp!=nullptr && isa<AtenMmOp>(mmOp) && isa<ValueTensorLiteralOp>(mmOp->getOperand(1).getDefiningOp())){
            llvm::outs() << "1.2!\n";
            numOfOp++;
            mmOpWorklist.push_back(mmOp);
            auto opWeightShape = mmOp->getOperand(1).getType().cast<ValueTensorType>().getSizes().vec();
            vector<int> tmp1;
            for(auto num: opWeightShape){
                tmp1.push_back(num); 
            }
            llvm::outs() << "1.3!\n";
            auto opWeightData = mmOp->getOperand(1).getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
            vector<float> tmp2;
            for(auto num: opWeightData) tmp2.push_back(num);
            mmOpWeightShape.push_back(tmp1);
            mmOpWeightData.push_back(tmp2);

            
            // todo: addTensorOp has three parameters
            llvm::outs() << "1.4!\n";
            mmBiasOpWorklist.push_back(op); 
            auto opBiasShape = op->getOperand(1).getType().cast<ValueTensorType>().getSizes().vec();
            tmp1.clear();
            for(auto num: opBiasShape) tmp1.push_back(num); 
            auto opBiasData = op->getOperand(1).getDefiningOp<ValueTensorLiteralOp>().getValue().getValues<float>();
            tmp2.clear();
            for(auto num: opBiasData) tmp2.push_back(num); 
            mmOpBiasShape.push_back(tmp1);
            mmOpBiasData.push_back(tmp2);
        }
    } // todo: only insert mmOp?
  }
  llvm::outs() << "2222222222222222222222222!\n";
  for(int i=0;i<numOfOp;i++){
    if(mmOpWeightShape[i][0]==mmOpWeightShape[i][1]){
        vector<float> col(mmOpWeightShape[i][1], 0);
        vector<vector<float>> unitMatrix(mmOpWeightShape[i][0], col);
        for(int j=0;j<mmOpWeightShape[i][1];j++) unitMatrix[j][j]=1;
        auto unitMatrixFlatten = flatten(unitMatrix);
        if(isEqual(mmOpWeightData[i], unitMatrixFlatten)){ // it is an unitMatrixFlatten, let us check bias
            if(isEqual(mmOpBiasData[i], col)){ // B==0
                // process
                auto it_1 = mmOpWorklist.begin() + i;
                auto it_2 = mmBiasOpWorklist.begin() + i;
                auto userOps = (*it_2)->getUses();
                auto it = userOps.begin();
                while(it!=userOps.end()){
                    auto tmpOp=it->getOwner();
                    tmpOp->replaceUsesOfWith(tmpOp->getOperand(0), (*it_1)->getOperand(0));

                    userOps = (*it_2)->getUses(); 
                    it=userOps.begin();
                }
                (*it_2)->erase();
                (*it_1)->erase();
            }
        }
    }
  }
  llvm::outs() << "3333333333333333333333333333!\n";
  for(int i=0;i<numOfOp;i++){
    // check 
    // (xA+B)C+D = xAC + BC + D
    // Xn*m Am*m Bn*m Cm*m Dn*m
    if((i+1)<numOfOp){ // shoule be i+1 but bugs
        if( mmOpWeightShape[i][0]==mmOpWeightShape[i][1] &&
            mmOpWeightShape[i][0]==mmOpWeightShape[i+1][1] &&
            mmOpWeightShape[i][1]==mmOpWeightShape[i+1][0]){
            vector<float> col(mmOpWeightShape[i+1][1], 0);
            vector<vector<float>> unitMatrix(mmOpWeightShape[i+1][0], col);
            for(int j=0;j<mmOpWeightShape[i+1][1];j++) unitMatrix[j][j]=1;
            vector<float> unitMatrixFlatten = flatten(unitMatrix);

            int m = mmOpWeightShape[i][0];
            
            // A*C == E?
            vector<float> AC=flatten(matMul(toMatrix(mmOpWeightData[i], m, m), toMatrix(mmOpWeightData[i+1], m, m)));
            if(isEqual(AC, unitMatrixFlatten)){ 
                // BC + D == 0？
                // biasShape:  align by broadcasting 
                vector<float> BCD = flatten(matAdd(matMul(toMatrix(mmOpBiasData[i], 1, m), // B  
                                                         toMatrix(mmOpWeightData[i+1], m, m)), // C
                                                toMatrix(mmOpBiasData[i+1], 1, m))); // D
                vector<float> zeroVec(BCD.size(), 0);
                if(isEqual(BCD, zeroVec)){
                    /*
                    mm1
                    add1
                    relu1
                    mm2
                    add2
                    relu2
                    */
                    auto it_1 = mmOpWorklist.begin() + i; // mm1
                    auto it_2 = mmBiasOpWorklist.begin() + i + 1; // add2
                    auto userOps = (*it_2)->getUses();
                    auto it = userOps.begin();
                    while(it!=userOps.end()){
                        auto tmpOp=it->getOwner();
                        tmpOp->replaceUsesOfWith(tmpOp->getOperand(0), (*it_1)->getOperand(0));
                        userOps = (*it_2)->getUses();

                        it=userOps.begin();
                    }
                    

                    // delete add2 mm2 ... add1 mm1
                    auto tmpOp1 = *(it_2);
                    while(tmpOp1!=*it_1){
                        auto tmpOp2 = tmpOp1->getOperand(0).getDefiningOp();
                        auto usersOp = tmpOp1->getUses();
                        if(usersOp.begin()==usersOp.end()){
                            tmpOp1->erase();
                        }
                        tmpOp1=tmpOp2;
                    }
                    auto usersOp = tmpOp1->getUses();
                    if(usersOp.begin()==usersOp.end()){
                        tmpOp1->erase(); // delete mm1
                    }
                    ++i; // Note: jump next mm + add Op
                }
            }
        }
    }
  }
  
  // todo
//   for(int i=0;i<numOfOp;i++){
//     if(i+2<numOfOp){
//         if(mmOpWeightShape[i][0]==mmOpWeightShape[i+2][1] && // m*m
//            mmOpWeightShape[i][1]==mmOpWeightShape[i+1][0] && // n*n
//            mmOpWeightShape[i][1]==mmOpWeightShape[i+1][0] && // s*s
//         )
//     }
    
//   }
    
  // erase the ops which difficultly process in the process
  opWorklist.clear();

  f->walk([&](mlir::Operation *op){ // all Ops
      opWorklist.push_back(op);
  });
  for(auto it_3=opWorklist.begin();it_3!=opWorklist.end();it_3++){
    auto op = *(it_3);
    // relu relu ... relu
    if(isa<AtenReluOp>(op)){
        if(isa<AtenReluOp>(op->getOperand(0).getDefiningOp())){
            auto tmpOp = dyn_cast<AtenReluOp>(op->getOperand(0).getDefiningOp());
            op->replaceUsesOfWith(op->getOperand(0), tmpOp->getOperand(0));
            tmpOp->erase();
        }
    }
    if(isa<ConstantIntOp, PrimListConstructOp, ValueTensorLiteralOp>(op)){
        auto usersOp = op->getUses();
        if(usersOp.begin()==usersOp.end()){
            op->erase();
        }
    }
  }
}


namespace {
class AntiInsertLinearPass : public AntiInsertLinearBase<AntiInsertLinearPass> {
public:
  AntiInsertLinearPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    antiInsertLinear(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createAntiInsertLinearPass() {
  return std::make_unique<AntiInsertLinearPass>();
}