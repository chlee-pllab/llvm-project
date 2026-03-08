//===--- ExpandLargeDivRem.cpp - Expand large div/rem ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands div/rem instructions with a bitwidth above a threshold
// into a call to auto-generated functions.
// This is useful for targets like x86_64 that cannot lower divisions
// with more than 128 bits or targets like x86_32 that cannot lower divisions
// with more than 64 bits.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandLargeDivRem.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Alignment.h"
#include "llvm/IR/Attributes.h"

using namespace llvm;

static cl::opt<unsigned>
    ExpandDivRemBits("expand-div-rem-bits", cl::Hidden,
                     cl::init(llvm::IntegerType::MAX_INT_BITS),
                     cl::desc("div and rem instructions on integers with "
                              "more than <N> bits are expanded."));

static bool isConstantPowerOfTwo(llvm::Value *V, bool SignedOp) {
  auto *C = dyn_cast<ConstantInt>(V);
  if (!C)
    return false;

  APInt Val = C->getValue();
  if (SignedOp && Val.isNegative())
    Val = -Val;
  return Val.isPowerOf2();
}

static bool isSigned(unsigned int Opcode) {
  return Opcode == Instruction::SDiv || Opcode == Instruction::SRem;
}

static void scalarize(BinaryOperator *BO,
                      SmallVectorImpl<BinaryOperator *> &Replace) {
  VectorType *VTy = cast<FixedVectorType>(BO->getType());

  IRBuilder<> Builder(BO);

  unsigned NumElements = VTy->getElementCount().getFixedValue();
  Value *Result = PoisonValue::get(VTy);
  for (unsigned Idx = 0; Idx < NumElements; ++Idx) {
    Value *LHS = Builder.CreateExtractElement(BO->getOperand(0), Idx);
    Value *RHS = Builder.CreateExtractElement(BO->getOperand(1), Idx);
    Value *Op = Builder.CreateBinOp(BO->getOpcode(), LHS, RHS);
    Result = Builder.CreateInsertElement(Result, Op, Idx);
    if (auto *NewBO = dyn_cast<BinaryOperator>(Op)) {
      NewBO->copyIRFlags(Op, true);
      Replace.push_back(NewBO);
    }
  }
  BO->replaceAllUsesWith(Result);
  BO->dropAllReferences();
  BO->eraseFromParent();
}

static bool addVP(Function &F, const TargetLowering &TLI) {
  bool Modified = false;

  SmallVector<CallInst*, 4> MaskedLoads;
  SmallVector<CallInst*, 4> MaskedStores;
  SmallVector<Instruction*, 8> InstructionsToDelete;

  for (auto &I : instructions(F)) {
    if (auto *IE = dyn_cast<InsertElementInst>(&I)) {
        InstructionsToDelete.push_back(IE);
    }
    if (auto *SV = dyn_cast<ShuffleVectorInst>(&I)) {
        InstructionsToDelete.push_back(SV);
    }
    if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
      if (BO->getOpcode() == Instruction::Or) {
        InstructionsToDelete.push_back(BO);
      }
    }
    if (auto *ICmp = dyn_cast<ICmpInst>(&I)) {
      InstructionsToDelete.push_back(ICmp);
    }
  }
  for (auto &I : instructions(F)) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      Function *Callee = CI->getCalledFunction();
      if (!Callee) continue;
      StringRef Name = Callee->getName();
      if (Name.starts_with("llvm.masked.load.")) {
        MaskedLoads.push_back(CI);
        dbgs()<<"masked.load\n";
      } else if (Name.starts_with("llvm.masked.store.")) {
        MaskedStores.push_back(CI);
        dbgs()<<"masked.store\n";
      }
    }
  }

  Module *M = F.getParent();
  if (!M) return Modified;
  LLVMContext &Ctx = F.getContext();
  Constant *TrueMask = ConstantVector::getSplat(
      ElementCount::getFixed(128),
      ConstantInt::getTrue(Ctx));
  FunctionCallee GetVectorLengthFunc = M->getOrInsertFunction(
      "llvm.experimental.get.vector.length.i64",
      FunctionType::get(Type::getInt32Ty(Ctx),
                       {Type::getInt64Ty(Ctx),
                        Type::getInt32Ty(Ctx),
                        Type::getInt1Ty(Ctx)},
                       false));

  for (CallInst *CI : MaskedLoads) {
    LLVMContext &Ctx = CI->getContext();
    IRBuilder<> Builder(CI);

    Value *Ptr = CI->getArgOperand(0);
    Value *Alignment = CI->getArgOperand(1);
    //Value *PassThru = CI->getArgOperand(3);

    Type *ReturnType = CI->getType();

    VectorType *VecTy = cast<VectorType>(ReturnType);
    ElementCount EC = VecTy->getElementCount();
    if (!EC.isFixed()) {
      continue;
    }
    uint64_t NumElements = EC.getFixedValue();

    Value *ActiveElementCount = ConstantInt::get(Type::getInt64Ty(Ctx), NumElements);
    CallInst *VL = Builder.CreateCall(GetVectorLengthFunc,
                                      {ActiveElementCount,
                                       ConstantInt::get(Type::getInt32Ty(Ctx), 128),
                                       ConstantInt::getTrue(Ctx)});
    VL->setTailCall(true);
    VL->setDebugLoc(CI->getDebugLoc());

    std::string VPLoadName = "llvm.vp.load.v128f32.p0";

    Type *VPLoadParams[] = {
        Ptr->getType(),                    // pointer
        TrueMask->getType(),                  // mask
        Type::getInt32Ty(Ctx)             // vector length
    };

    FunctionCallee VPLoadFunc = M->getOrInsertFunction(
        VPLoadName,
        FunctionType::get(ReturnType, VPLoadParams, false));

    CallInst *VPLoad = Builder.CreateCall(VPLoadFunc,
                                         {Ptr, TrueMask, VL});
    VPLoad->setTailCall(true);
    VPLoad->setDebugLoc(CI->getDebugLoc());

    if (auto *AlignConst = dyn_cast<ConstantInt>(Alignment)) {
      MaybeAlign AlignVal(AlignConst->getZExtValue());
      if (AlignVal) {
        VPLoad->addParamAttr(0, Attribute::getWithAlignment(Ctx, *AlignVal));
      }
    }

    CI->replaceAllUsesWith(VPLoad);
    CI->eraseFromParent();
    Modified = true;
  }

  for (CallInst *CI : MaskedStores) {
    LLVMContext &Ctx = CI->getContext();
    IRBuilder<> Builder(CI);

    Value *ValueToStore = CI->getArgOperand(0);
    Value *Ptr = CI->getArgOperand(1);
    Value *Alignment = CI->getArgOperand(2);

    Type *ValueType = ValueToStore->getType();

    VectorType *VecTy = cast<VectorType>(ValueType);
    ElementCount EC = VecTy->getElementCount();
    if (!EC.isFixed()) {
      continue;
    }
    uint64_t NumElements = EC.getFixedValue();

    Value *ActiveElementCount = ConstantInt::get(Type::getInt64Ty(Ctx), NumElements);
    CallInst *VL = Builder.CreateCall(GetVectorLengthFunc,
                                      {ActiveElementCount,
                                       ConstantInt::get(Type::getInt32Ty(Ctx), 128),
                                       ConstantInt::getTrue(Ctx)});
    VL->setTailCall(true);
    VL->setDebugLoc(CI->getDebugLoc());

    std::string VPStoreName = "llvm.vp.store.v128f32.p0";

    Type *VPStoreParams[] = {
        ValueToStore->getType(),           // value to store
        Ptr->getType(),                   // pointer
        TrueMask->getType(),                  // mask
        Type::getInt32Ty(Ctx)             // vector length
    };

    FunctionCallee VPStoreFunc = M->getOrInsertFunction(
        VPStoreName,
        FunctionType::get(Type::getVoidTy(Ctx), VPStoreParams, false));

    CallInst *VPStore = Builder.CreateCall(VPStoreFunc,
                                          {ValueToStore, Ptr, TrueMask, VL});
    VPStore->setTailCall(true);
    VPStore->setDebugLoc(CI->getDebugLoc());

    if (auto *AlignConst = dyn_cast<ConstantInt>(Alignment)) {
      MaybeAlign AlignVal(AlignConst->getZExtValue());
      if (AlignVal) {
        VPStore->addParamAttr(1, Attribute::getWithAlignment(Ctx, *AlignVal));
      }
    }

    CI->eraseFromParent();
    Modified = true;
  }
  for (auto &I : instructions(F)) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      Function *Callee = CI->getCalledFunction();
      if (Callee && Callee->getName() == "llvm.experimental.get.vector.length.i64") {
        if (auto *Const = dyn_cast<ConstantInt>(CI->getArgOperand(1))) {
          if (Const->getZExtValue() == 16) {
            CI->setArgOperand(1, ConstantInt::get(Type::getInt32Ty(Ctx), 128));
          }
        }
        CI->setTailCall(true);
      }
      if (Callee && Callee->getName().starts_with("llvm.vp.")) {
        CI->setTailCall(true);
      }
    }
  }
  for (Instruction *I : InstructionsToDelete) {
    if (I && !I->use_empty()) {
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    }
    if (I) {
      I->eraseFromParent();
    }
  }

  return Modified;
}

static bool runImpl(Function &F, const TargetLowering &TLI) {
  addVP(F, TLI);

  SmallVector<BinaryOperator *, 4> Replace;
  SmallVector<BinaryOperator *, 4> ReplaceVector;
  bool Modified = false;

  unsigned MaxLegalDivRemBitWidth = TLI.getMaxDivRemBitWidthSupported();
  if (ExpandDivRemBits != llvm::IntegerType::MAX_INT_BITS)
    MaxLegalDivRemBitWidth = ExpandDivRemBits;

  if (MaxLegalDivRemBitWidth >= llvm::IntegerType::MAX_INT_BITS)
    return false;

  for (auto &I : instructions(F)) {
    switch (I.getOpcode()) {
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem: {
      // TODO: This pass doesn't handle scalable vectors.
      if (I.getOperand(0)->getType()->isScalableTy())
        continue;

      auto *IntTy = dyn_cast<IntegerType>(I.getType()->getScalarType());
      if (!IntTy || IntTy->getIntegerBitWidth() <= MaxLegalDivRemBitWidth)
        continue;

      // The backend has peephole optimizations for powers of two.
      // TODO: We don't consider vectors here.
      if (isConstantPowerOfTwo(I.getOperand(1), isSigned(I.getOpcode())))
        continue;

      if (I.getOperand(0)->getType()->isVectorTy())
        ReplaceVector.push_back(&cast<BinaryOperator>(I));
      else
        Replace.push_back(&cast<BinaryOperator>(I));
      Modified = true;
      break;
    }
    default:
      break;
    }
  }

  while (!ReplaceVector.empty()) {
    BinaryOperator *BO = ReplaceVector.pop_back_val();
    scalarize(BO, Replace);
  }

  if (Replace.empty())
    return false;

  while (!Replace.empty()) {
    BinaryOperator *I = Replace.pop_back_val();

    if (I->getOpcode() == Instruction::UDiv ||
        I->getOpcode() == Instruction::SDiv) {
      expandDivision(I);
    } else {
      expandRemainder(I);
    }
  }

  return Modified;
}

namespace {
class ExpandLargeDivRemLegacyPass : public FunctionPass {
public:
  static char ID;

  ExpandLargeDivRemLegacyPass() : FunctionPass(ID) {
    initializeExpandLargeDivRemLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    auto *TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    auto *TLI = TM->getSubtargetImpl(F)->getTargetLowering();
    return runImpl(F, *TLI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
} // namespace

PreservedAnalyses ExpandLargeDivRemPass::run(Function &F,
                                             FunctionAnalysisManager &FAM) {
  const TargetSubtargetInfo *STI = TM->getSubtargetImpl(F);
  return runImpl(F, *STI->getTargetLowering()) ? PreservedAnalyses::none()
                                               : PreservedAnalyses::all();
}

char ExpandLargeDivRemLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(ExpandLargeDivRemLegacyPass, "expand-large-div-rem",
                      "Expand large div/rem", false, false)
INITIALIZE_PASS_END(ExpandLargeDivRemLegacyPass, "expand-large-div-rem",
                    "Expand large div/rem", false, false)

FunctionPass *llvm::createExpandLargeDivRemPass() {
  return new ExpandLargeDivRemLegacyPass();
}
