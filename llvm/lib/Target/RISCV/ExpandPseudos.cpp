//===-- ExpandPseudos.cpp - Expand Pseudos ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpandPseudos.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"

using namespace llvm;

#define DEBUG_TYPE "expandpseudos"
#define RISCV_INSERT_VSETVLI_NAME "Expand to Pseudos and Sink pass"

char ExpandPseudos::ID = 0;
char &llvm::ExpandPseudosID = ExpandPseudos::ID;

INITIALIZE_PASS(ExpandPseudos, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static cl::opt<bool>
    Sink("sink", cl::init(false), cl::Hidden,
               cl::desc("Enable sinking"));

ExpandPseudos::ExpandPseudos() : MachineFunctionPass(ID) {
  initializeExpandPseudosPass(*PassRegistry::getPassRegistry());
}

void ExpandPseudos::getAnalysisUsage(AnalysisUsage &AU) const {
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

bool ExpandPseudos::LowerCopy(MachineBasicBlock &MBB, MachineInstr &MI) {
  bool MadeChange = false;
  if (MI.getNumOperands() >= 2 &&
            MI.getOperand(0).isReg() &&
            MI.getOperand(1).isReg()) {

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = MI.getOperand(1).getReg();

    if (DstReg.isVirtual() && SrcReg.isVirtual()) {
      const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);
      const TargetRegisterClass *SrcRC = MRI->getRegClass(SrcReg);
            
      StringRef DstName = TRI->getRegClassName(DstRC);
      StringRef SrcName = TRI->getRegClassName(SrcRC);
 
      if (DstName.contains("VRM8") && SrcName.contains("VRM8")) {
        LLVM_DEBUG(dbgs() << "lowerCopy: "<<MI);
        //LLVM_DEBUG(dbgs() << "DstRC: " << DstName 
        //                  << " SrcRC: " << SrcName << "\n");
        Register VLReg;
        unsigned SEW = 5;
              
        DebugLoc DL = MI.getDebugLoc();
        MachineBasicBlock::iterator MBBI = MI.getIterator();

        for (auto It = MBB.begin(); It != MI.getIterator(); ++It) {
          MachineInstr &PrevMI = *It;
          if (PrevMI.getOpcode() == RISCV::PseudoVMV_V_I_M8 &&
            PrevMI.getOperand(0).isReg() &&
            PrevMI.getOperand(0).getReg() == SrcReg &&
            PrevMI.getNumOperands() >= 2 &&
            PrevMI.getOperand(2).isImm() &&
            PrevMI.getOperand(2).getImm() == 0) {
                  
            if (PrevMI.getNumOperands() >= 4) {
              if (PrevMI.getOperand(3).isReg())
                 VLReg = PrevMI.getOperand(3).getReg();
            }
            break;
          }
        }
              
        LLVM_DEBUG(dbgs() << "Replacing COPY with VMV: SEW=" << SEW << "\n");
              
        auto NewMI = BuildMI(MBB, MBBI, DL, 
                                  TII->get(RISCV::PseudoVMV_V_I_M8))
          .addReg(DstReg, RegState::Define)
          .addReg(DstReg, RegState::Undef)
          .addImm(0)
          .addReg(VLReg)
          .addImm(SEW)
          .addImm(0);
              
        //for (unsigned i = 2; i < MI.getNumOperands(); ++i) {
        //  NewMI.add(MI.getOperand(i));
        //}
        NewMI->setDebugLoc(DL);
        MI.eraseFromParent();
        MadeChange = true;
      }
    }
  }
  return MadeChange;
}

MachineInstr *
ExpandPseudos::FindInSameSuccToSinkTo(MachineInstr &MI, MachineBasicBlock *MBB,
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
    if (UseMI.getOpcode() == RISCV::PseudoVFADD_VV_M8_E32) {
      for (unsigned i = 0; i < UseMI.getNumOperands(); i++) {
        const MachineOperand &UseOp = UseMI.getOperand(i);
        if (UseOp.isReg() && UseOp.getReg() == Reg) {
          LLVM_DEBUG(dbgs()<<"  Found UseMI: "<<UseMI);
          LLVM_DEBUG(dbgs() << "  "<<UseOp<<" is used as operand " << i << "\n");
          SuccToSinkTo = &UseMI;
          return SuccToSinkTo;
	}
      }
    }
  }
  return SuccToSinkTo;
}

bool ExpandPseudos::SinkInSameInstruction(MachineInstr &MI, bool &SawStore,
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

void ExpandPseudos::ProcessInSameBlock(MachineFunction &MF) {
  int LoadTime256 = 0;
  int LoadTime384 = 0;
  DenseSet<Register> SinkRegs;
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs()<<"ProcessInSameBlock.\n");
    AllSuccsCache AllSuccessors;
    //MachineBasicBlock::iterator I = MBB.end();
    //--I;
    bool ProcessedBegin, SawStore = false;
    for (auto It = MBB.begin(); It != MBB.end(); ){
      MachineInstr &MI = *It;
      bool Sunk = false;
      auto NextIt = std::next(It);
      //ProcessedBegin = I == MBB.begin();
      //if (!ProcessedBegin)
      //  --I;
      LLVM_DEBUG(dbgs()<<MI);
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      StringRef Name = TII->getName(MI.getOpcode());
      if (Name.contains("PseudoVMV_V_I")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &MO = MI.getOperand(1);
        if (Dst.isReg() && MO.isReg() && Dst.getReg() == MO.getReg()) {
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
          if (SinkInSameInstruction(MI, SawStore, AllSuccessors)) {
            LLVM_DEBUG(dbgs()<<"Sink success. \n");
            Sunk = true;
          }
        }
      }
      else if (Name.contains("PseudoVLE32_V_M8")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &Src = MI.getOperand(1);
        if (Dst.isReg() && Src.isReg()) {
          Register DstReg = Dst.getReg();
          //Register SrcReg = Src.getReg();
	  if (!SinkRegs.count(DstReg) && MI.memoperands_begin() != MI.memoperands_end()) {
            const MachineMemOperand *MMO = *MI.memoperands_begin();
            if (const Value *PtrVal = MMO->getValue()) {
              int64_t Offset = MMO->getOffset();
              if (Offset == 256) {
                LLVM_DEBUG(dbgs() << "Matched 3rd VLE32 load: offset=" << Offset << "\n");
		++LoadTime256;
                if (1 <= LoadTime256 && LoadTime256 <= 4) {
		  SinkRegs.insert(DstReg);
                  LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
                  if (SinkInSameInstruction(MI, SawStore, AllSuccessors)) {
                    LLVM_DEBUG(dbgs()<<"Sink success. \n");
                    Sunk = true;
                  }
		}
              }
              if (Offset == 384) {
                LLVM_DEBUG(dbgs() << "Matched 4th VLE32 load: offset=" << Offset << "\n");
		++LoadTime384;
                if (1 <= LoadTime384 && LoadTime384 <= 4) {
		  SinkRegs.insert(DstReg);
                  LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
                  if (SinkInSameInstruction(MI, SawStore, AllSuccessors)) {
                    LLVM_DEBUG(dbgs()<<"Sink success. \n");
                    Sunk = true;
                  }
		}
              }
            }
          } 
	}
      }
      if (Sunk) {
        It = NextIt;
      } else {
        ++It;
      }
    }// while (!ProcessedBegin);
    SeenDbgUsers.clear();
    SeenDbgVars.clear();
    CachedRegisterPressure.clear();
  }
}

bool ExpandPseudos::runSink(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "******** RISCV Sink ********\n");

  bool EverMadeChange = false;

  ProcessInSameBlock(MF);

  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();

  return EverMadeChange;
}

bool ExpandPseudos::runOnMachineFunction(MachineFunction &MF) {
  if (!Sink) {
    dbgs()<<"RISCV Sink disabled.\n";
    return false;
  } else {
    dbgs()<<"RISCV Sink enabled.\n";
  }

  LLVM_DEBUG(dbgs() << "******** Expand Pseudos ********\n");
  STI = &MF.getSubtarget();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      // Only expand pseudos.
      if (!MI.isPseudo())
        continue;

      // Give targets a chance to expand even standard pseudos.
      if (TII->expandPostRAPseudo(MI)) {
        LLVM_DEBUG(dbgs() << "expandPseudo: "<<MI);
        MadeChange = true;
        continue;
      }

      // Expand standard pseudos.
      switch (MI.getOpcode()) {
      //case TargetOpcode::SUBREG_TO_REG:
      //  MadeChange |= LowerSubregToReg(&MI);
      //  break;
      case TargetOpcode::COPY:
        MadeChange = LowerCopy(MBB, MI);
        if (MadeChange) {
          LLVM_DEBUG(dbgs()<<"lowerCopy success.\n");
	}
        break;
      case TargetOpcode::DBG_VALUE:
        continue;
      case TargetOpcode::INSERT_SUBREG:
      case TargetOpcode::EXTRACT_SUBREG:
        llvm_unreachable("Sub-register pseudos should have been eliminated.");
      }
    }
  }

  MadeChange |= runSink(MF);

  return MadeChange;
}
