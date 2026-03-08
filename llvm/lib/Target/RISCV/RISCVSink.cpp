//===-- RISCVSink.cpp - RISC-V Sink ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVSink.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"

using namespace llvm;

#define DEBUG_TYPE "riscvsink"
#define RISCV_INSERT_VSETVLI_NAME "RISC-V Sink"

char RISCVSink::ID = 0;
char &llvm::RISCVSinkID = RISCVSink::ID;

INITIALIZE_PASS(RISCVSink, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static cl::opt<bool>
    EnableSinks("sinks", cl::init(false), cl::Hidden,
               cl::desc("Enable riscv sinking"));

RISCVSink::RISCVSink() : MachineFunctionPass(ID) {
  initializeRISCVSinkPass(*PassRegistry::getPassRegistry());
}

void RISCVSink::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  //AU.addRequired<MachineDominatorTreeWrapperPass>();
  //AU.addRequired<MachineCycleInfoWrapperPass>();
  //AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static bool attemptDebugCopyProp(MachineInstr &SinkInst, MachineInstr &DbgMI,
                                 Register Reg) {
  const MachineRegisterInfo &MRI = SinkInst.getMF()->getRegInfo();
  const TargetInstrInfo &TII = *SinkInst.getMF()->getSubtarget().getInstrInfo();

  // Copy DBG_VALUE operand and set the original to undef. We then check to
  // see whether this is something that can be copy-forwarded. If it isn't,
  // continue around the loop.

  const MachineOperand *SrcMO = nullptr, *DstMO = nullptr;
  auto CopyOperands = TII.isCopyInstr(SinkInst);
  if (!CopyOperands)
    return false;
  SrcMO = CopyOperands->Source;
  DstMO = CopyOperands->Destination;

  // Check validity of forwarding this copy.
  bool PostRA = MRI.getNumVirtRegs() == 0;

  // Trying to forward between physical and virtual registers is too hard.
  if (Reg.isVirtual() != SrcMO->getReg().isVirtual())
    return false;

  // Only try virtual register copy-forwarding before regalloc, and physical
  // register copy-forwarding after regalloc.
  bool arePhysRegs = !Reg.isVirtual();
  if (arePhysRegs != PostRA)
    return false;

  // Pre-regalloc, only forward if all subregisters agree (or there are no
  // subregs at all). More analysis might recover some forwardable copies.
  if (!PostRA)
    for (auto &DbgMO : DbgMI.getDebugOperandsForReg(Reg))
      if (DbgMO.getSubReg() != SrcMO->getSubReg() ||
          DbgMO.getSubReg() != DstMO->getSubReg())
        return false;

  // Post-regalloc, we may be sinking a DBG_VALUE of a sub or super-register
  // of this copy. Only forward the copy if the DBG_VALUE operand exactly
  // matches the copy destination.
  if (PostRA && Reg != DstMO->getReg())
    return false;

  for (auto &DbgMO : DbgMI.getDebugOperandsForReg(Reg)) {
    DbgMO.setReg(SrcMO->getReg());
    DbgMO.setSubReg(SrcMO->getSubReg());
  }
  return true;
}

MachineInstr *
RISCVSink::FindInSameSuccToSinkTo(MachineInstr &MI, MachineBasicBlock *MBB,
                                 bool &BreakPHIEdge,
                                 AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs()<<"FindInSameSuccToSinkTo.\n");
  assert(MBB && "Invalid MachineBasicBlock!");

  // loop over all the operands of the specified instruction.  If there is
  // anything we can't handle, bail out.

  // SuccToSinkTo - This is the successor to sink this instruction to, once we
  // decide.
  MachineInstr *SuccToSinkTo = nullptr;
  const MachineOperand &MO = MI.getOperand(0);
  Register Reg = MO.getReg();
  LLVM_DEBUG(dbgs()<<"MO:"<<MO<<"\n");
  LLVM_DEBUG(dbgs()<<"UseMI:\n");
  bool AfterMI = false;
  for (MachineInstr &UseMI : *MBB) {
    if (&UseMI == &MI) {
      AfterMI = true;
      continue;
    }
    if (!AfterMI) {
      continue;
    }
    LLVM_DEBUG(dbgs()<<"  "<<UseMI);
    if (UseMI.getNumOperands() > 0) {
      const MachineOperand &UseOp = UseMI.getOperand(0);
      if (UseOp.isReg() && UseOp.getReg() == Reg) {
        if (UseMI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK) {
          LLVM_DEBUG(dbgs()<<"  Found UseMI: "<<UseMI);
          SuccToSinkTo = &UseMI;
          break;
	}
      }
    }
  }

  return SuccToSinkTo;
}

bool RISCVSink::SinkInSameInstruction(MachineInstr &MI, bool &SawStore,
                                     AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs()<<"SinkInSameInstruction.\n");
  if (!TII->shouldSink(MI)) {
    LLVM_DEBUG(dbgs()<<"should not sink.\n");
    return false;
  }
  if (!MI.isSafeToMove(SawStore)) {
    LLVM_DEBUG(dbgs()<<"not safe to move.\n");
    return false;
  }
  if (MI.isConvergent()) {
    LLVM_DEBUG(dbgs()<<"is convergent.\n");
    return false;
  }
  bool BreakPHIEdge = false;
  MachineBasicBlock *ParentBlock = MI.getParent();
  MachineBasicBlock *SuccToSinkTo = MI.getParent();
  MachineInstr *TargetUser =
      FindInSameSuccToSinkTo(MI, ParentBlock, BreakPHIEdge, AllSuccessors);
  if (!TargetUser)
    return false;
  for (const MachineOperand &MO : MI.all_defs()) {
    Register Reg = MO.getReg();
    if (Reg == 0 || !Reg.isPhysical())
      continue;
    if (SuccToSinkTo->isLiveIn(Reg))
      return false;
  }

  LLVM_DEBUG(dbgs() << "Sink instr " << MI);

  //MachineBasicBlock::iterator InsertPos =
  //    SuccToSinkTo->SkipPHIsAndLabels(SuccToSinkTo->begin());
  MachineBasicBlock::iterator InsertPos = TargetUser->getIterator();
  MachineBasicBlock::iterator CurrPos = MI.getIterator();
  //if (blockPrologueInterferes(SuccToSinkTo, InsertPos, MI, TRI, TII, MRI)) {
  //  LLVM_DEBUG(dbgs() << " *** Not sinking: prologue interference\n");
  //  return false;
  //}
  SmallVector<MIRegs, 4> DbgUsersToSink;
  for (auto &MO : MI.all_defs()) {
    if (!MO.getReg().isVirtual())
      continue;
    auto It = SeenDbgUsers.find(MO.getReg());
    if (It == SeenDbgUsers.end())
      continue;
    auto &Users = It->second;
    for (auto &User : Users) {
      MachineInstr *DbgMI = User.getPointer();
      if (User.getInt()) {
        if (!attemptDebugCopyProp(MI, *DbgMI, MO.getReg()))
          DbgMI->setDebugValueUndef();
      } else {
        DbgUsersToSink.push_back(
            {DbgMI, SmallVector<Register, 2>(1, MO.getReg())});
      }
    }
  }
  //performSink(MI, *SuccToSinkTo, InsertPos, DbgUsersToSink);
  SuccToSinkTo->splice(InsertPos, SuccToSinkTo, CurrPos);
  for (MachineOperand &MO : MI.all_uses())
    RegsToClearKillFlags.insert(MO.getReg());
  return true;
}

void RISCVSink::ProcessInSameBlock(MachineFunction &MF) {
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs()<<"ProcessInSameBlock.\n");
    AllSuccsCache AllSuccessors;
    MachineBasicBlock::iterator I = MBB.end();
    --I;
    bool ProcessedBegin, SawStore = false;
    do {
      MachineInstr &MI = *I;
      ProcessedBegin = I == MBB.begin();
      if (!ProcessedBegin)
        --I;
      LLVM_DEBUG(dbgs()<<MI);
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      StringRef Name = TII->getName(MI.getOpcode());
      if (Name.contains("PseudoVMV_V_I")) {
        const MachineOperand &MO = MI.getOperand(0);
	if (MO.isReg() && MO.getReg() == RISCV::V16M8) {
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
          if (SinkInSameInstruction(MI, SawStore, AllSuccessors)) {
            LLVM_DEBUG(dbgs()<<"Sink success. \n");
            break;
          }
	}
      }

    } while (!ProcessedBegin);
    SeenDbgUsers.clear();
    SeenDbgVars.clear();
    CachedRegisterPressure.clear();
  }
}

bool RISCVSink::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableSinks) {
    return false;
  } else {
  }

  LLVM_DEBUG(dbgs() << "******** RISCV Sinks ********\n");
  STI = &MF.getSubtarget();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool EverMadeChange = false;

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  ProcessInSameBlock(MF);

  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();

  return EverMadeChange;
}
