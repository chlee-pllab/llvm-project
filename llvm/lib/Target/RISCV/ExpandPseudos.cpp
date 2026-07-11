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
#include "llvm/IR/Function.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/CodeGen/MachineFunction.h"

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

static cl::opt<bool>
    a("a", cl::init(false), cl::Hidden,
               cl::desc("Enable affine"));

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
        unsigned SEW = 5;
        DebugLoc DL = MI.getDebugLoc();
        MachineBasicBlock::iterator MBBI = MI.getIterator();
        Register VLReg;
        Register VXReg;
        unsigned NewOpCode = 0;
        bool isVI = false;
	auto CreateNewVMV = [&](unsigned OpCode, bool isVI,
                                MachineInstr &PrevMI) -> MachineInstr * {
          MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL,
                                            TII->get(OpCode));
          MIB.addReg(DstReg, RegState::Define);
          MIB.addReg(DstReg, RegState::Undef);
	  if (isVI)
            MIB.addImm(0);
	  else
            MIB.addReg(VXReg);
          MIB.addReg(VLReg);
          MIB.addImm(SEW);
          MIB.addImm(0);
          MIB.copyImplicitOps(PrevMI);
          return MIB;
        };
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
	    NewOpCode = RISCV::PseudoVMV_V_I_M8;
            isVI = true;
          }
	  else if (PrevMI.getOpcode() == RISCV::PseudoVMV_V_X_M8 &&
            PrevMI.getOperand(0).isReg() &&
            PrevMI.getOperand(0).getReg() == SrcReg &&
            PrevMI.getNumOperands() >= 2 &&
            PrevMI.getOperand(2).isReg()) {
            if (PrevMI.getNumOperands() >= 4) {
              if (PrevMI.getOperand(2).isReg())
                 VXReg = PrevMI.getOperand(2).getReg();
              if (PrevMI.getOperand(3).isReg())
                 VLReg = PrevMI.getOperand(3).getReg();
            }
	    NewOpCode = RISCV::PseudoVMV_V_X_M8;
            isVI = false;
          }
	  if (NewOpCode) {
            LLVM_DEBUG(dbgs() << "Replacing COPY with VMV: SEW=" << SEW << "\n");
            auto NewMI = CreateNewVMV(NewOpCode, isVI, PrevMI);
            NewMI->setDebugLoc(DL);
            MI.eraseFromParent();
            return true;
	  }
        }
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
    //LLVM_DEBUG(dbgs()<<"  "<<UseMI);
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
    if (UseMI.getOpcode() == RISCV::PseudoVFMAX_VV_M8_E32) {
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
  MachineBasicBlock::iterator InsertPos = TargetUser->getIterator();
  MachineBasicBlock::iterator CurrPos = MI.getIterator();
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

bool ExpandPseudos::SinkInSameInstructionGroup(
    SmallVectorImpl<MachineInstr *> &InstrsToSink, bool &SawStore,
    AllSuccsCache &AllSuccessors) {
  LLVM_DEBUG(dbgs() << "SinkInSameInstructionGroup with "
                    << InstrsToSink.size() << " instructions.\n");
  if (InstrsToSink.empty())
    return false;
  MachineInstr &PrimaryMI = *InstrsToSink[0];
  for (MachineInstr *MI : InstrsToSink) {
    if (!TII->shouldSink(*MI)) {
      LLVM_DEBUG(dbgs() << "should not sink: " << *MI);
      return false;
    }
    if (!MI->isSafeToMove(SawStore)) {
      LLVM_DEBUG(dbgs() << "not safe to move: " << *MI);
      return false;
    }
    if (MI->isConvergent()) {
      LLVM_DEBUG(dbgs() << "is convergent: " << *MI);
      return false;
    }
  }
  bool BreakPHIEdge = false;
  MachineBasicBlock *ParentBlock = PrimaryMI.getParent();
  MachineBasicBlock *SuccToSinkTo = PrimaryMI.getParent();
  MachineInstr *TargetUser =
      FindInSameSuccToSinkTo(PrimaryMI, ParentBlock, BreakPHIEdge, AllSuccessors);
  if (!TargetUser)
    return false;
  for (MachineInstr *MI : InstrsToSink) {
    for (const MachineOperand &MO : MI->all_defs()) {
      Register Reg = MO.getReg();
      if (Reg == 0 || !Reg.isPhysical())
        continue;
      if (SuccToSinkTo->isLiveIn(Reg))
        return false;
    }
  }
  LLVM_DEBUG(dbgs() << "Sinking group of instructions\n");
  for (MachineInstr *MI : InstrsToSink) {
    LLVM_DEBUG(dbgs() << "  " << *MI);
  }
  MachineBasicBlock::iterator InsertPos = TargetUser->getIterator();
  for (auto It = InstrsToSink.begin(); It != InstrsToSink.end(); ++It) {
    MachineInstr *MI = *It;
    MachineBasicBlock::iterator CurrPos = MI->getIterator();
    SuccToSinkTo->splice(InsertPos, SuccToSinkTo, CurrPos);
    InsertPos = MI->getIterator();
  }
  SmallVector<MIRegs, 4> DbgUsersToSink;
  for (MachineInstr *MI : InstrsToSink) {
    for (auto &MO : MI->all_defs()) {
      if (!MO.getReg().isVirtual())
        continue;
      auto It = SeenDbgUsers.find(MO.getReg());
      if (It == SeenDbgUsers.end())
        continue;
      auto &Users = It->second;
      for (auto &User : Users) {
        MachineInstr *DbgMI = User.getPointer();
        if (User.getInt()) {
          if (!attemptDebugCopyProp(*MI, *DbgMI, MO.getReg()))
            DbgMI->setDebugValueUndef();
        } else {
          DbgUsersToSink.push_back(
              {DbgMI, SmallVector<Register, 2>(1, MO.getReg())});
        }
      }
    }
    for (MachineOperand &MO : MI->all_uses())
      RegsToClearKillFlags.insert(MO.getReg());
  }
  return true;
}

void ExpandPseudos::ProcessInSameBlock(MachineFunction &MF) {
  int64_t BlockSize = 128;
  for (const auto &BB : MF.getFunction()) {
    for (const auto &I : BB) {
      if (auto *VTy = dyn_cast<FixedVectorType>(I.getType())) {
        BlockSize = VTy->getNumElements();
      }
    }
  }
  LLVM_DEBUG(dbgs()<<"BlockSize: "<<BlockSize<<"\n");
  int SinkVMVTime = 0;
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
      //LLVM_DEBUG(dbgs()<<MI);
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      StringRef Name = TII->getName(MI.getOpcode());
      if (Name.contains("PseudoVMV_V_I")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &MO = MI.getOperand(1);
        ++SinkVMVTime;
        if (SinkVMVTime <= 2 && Dst.isReg() && MO.isReg() && Dst.getReg() == MO.getReg()) {
          LLVM_DEBUG(dbgs()<<"Prepare to sink "<<MI);
          if (SinkInSameInstruction(MI, SawStore, AllSuccessors)) {
            LLVM_DEBUG(dbgs()<<"Sink success. \n");
            Sunk = true;
          }
        }
      }
      if (Name.contains("PseudoVLE32_V_M8")) {
        const MachineOperand &Dst = MI.getOperand(0);
        const MachineOperand &Src = MI.getOperand(1);
        if (Dst.isReg() && Src.isReg()) {
          Register DstReg = Dst.getReg();
          //Register SrcReg = Src.getReg();
	  if (!SinkRegs.count(DstReg) && MI.memoperands_begin() != MI.memoperands_end()) {
            const MachineMemOperand *MMO = *MI.memoperands_begin();
            if (const Value *PtrVal = MMO->getValue()) {
              int64_t Offset = MMO->getOffset();
              bool ShouldSink = false;
              if (Offset >= BlockSize * 4 / 2) {
                LLVM_DEBUG(dbgs() << "Matched 3rd VLE32 load: offset=" << Offset << "\n");
		++LoadTime256;
                if (1 <= LoadTime256) {
                  ShouldSink = true;
		}
              }
              else if (Offset == 96 * 4) {
                LLVM_DEBUG(dbgs() << "Matched 4th VLE32 load: offset=" << Offset << "\n");
		++LoadTime384;
                if (1 <= LoadTime384) {
                  ShouldSink = true;
		}
              }
	      if (ShouldSink) {
                SinkRegs.insert(DstReg);
                LLVM_DEBUG(dbgs()<<"!Prepare to sink Group "<<MI);
                SmallVector<MachineInstr*, 4> InstructionsToSink;
                InstructionsToSink.push_back(&MI);
                Register LDReg = MI.getOperand(0).getReg();
                Register MaskReg = 0;
		for (MachineOperand &MO : MI.all_uses()) {
                  if (MO.isReg() && !MO.getReg().isVirtual()) {
                    LLVM_DEBUG(dbgs() <<"  Found MaskReg: " <<MO<<"\n");
		    MaskReg = MO.getReg();
                    break;
                  }
                }
                if (MaskReg) {
                  MachineInstr *MaskDef = nullptr;
                  for (auto It = MI.getIterator(); It != MI.getParent()->begin(); --It) {
                    MachineInstr &Candidate = *std::prev(It);
                    LLVM_DEBUG(dbgs() <<"  Reverse to find Mask: " <<Candidate);
                    for (MachineOperand &DefMO : Candidate.all_defs()) {
                      if (DefMO.isReg() && DefMO.getReg() == MaskReg) {
                        MaskDef = &Candidate;
                        break;
                      }
                    }
                    if (MaskDef)
                      break;
                  }
                  StringRef Name = TII->getName(MaskDef->getOpcode());
                  if (MaskDef && Name.contains("COPY")) {
                    LLVM_DEBUG(dbgs() <<"  Found Mask setup: " <<*MaskDef);
                    InstructionsToSink.push_back(MaskDef);
                  }
                }
		for (MachineInstr &UseMI : MRI->def_instructions(LDReg)) {
                  if (&UseMI == &MI) continue;
                  StringRef Name = TII->getName(UseMI.getOpcode());
                  if (Name.contains("PseudoVMV_V_I_M8")) {
                    LLVM_DEBUG(dbgs() <<"  Found VMV setup: " <<UseMI);
                    InstructionsToSink.push_back(&UseMI);
                    break;
                  }
                }
                if (SinkInSameInstructionGroup(InstructionsToSink, SawStore, AllSuccessors)) {
                  LLVM_DEBUG(dbgs()<<"Sink Group success. \n");
                  Sunk = true;
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

void ExpandPseudos::ProcessInSameAffine(MachineFunction &MF) {
  for (auto &MBB : MF) {
    LLVM_DEBUG(dbgs() << "ProcessInSameAffine.\n");
    for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
      MachineInstr &MI = *I;
      LLVM_DEBUG(dbgs() << MI);
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      if (MI.getOpcode() == RISCV::PseudoVID_V_M8) {
        MachineOperand &VIDDest = MI.getOperand(0);
        MachineOperand &VIDVL = MI.getOperand(2);
        MachineInstr *VOR_Orig = nullptr;
        Register VIDReg = VIDDest.getReg();
        Register VLReg = VIDVL.getReg();
        SmallVector<MachineInstr *, 8> Uses;
        for (MachineInstr &UseMI : MRI->use_instructions(VIDReg)) {
          if (&UseMI == &MI) continue;
          if (UseMI.getOpcode() == RISCV::PseudoVOR_VX_M8) {
            MachineOperand &SrcReg = UseMI.getOperand(2);
            if (SrcReg.isReg() && SrcReg.getReg() == VIDReg) {
              LLVM_DEBUG(dbgs() <<"  Imm is VOR: " <<UseMI);
              VOR_Orig = &UseMI;
            }
          } else if (UseMI.getOpcode() == RISCV::PseudoVADD_VX_M8) {
            LLVM_DEBUG(dbgs() <<"  Use VID result: " <<UseMI);
            Uses.push_back(&UseMI);
	  }
        }
        if (!VOR_Orig) {
          LLVM_DEBUG(dbgs() << "  No VOR_Orig found, skipping\n");
          continue;
        }
        MBB.splice(std::next(MI.getIterator()), &MBB, VOR_Orig->getIterator());
        for (MachineInstr *UseMI : Uses) {
          MachineOperand &SrcReg = UseMI->getOperand(2);
          MachineOperand &ImmReg = UseMI->getOperand(3);
          MachineInstr *ImmDef = MRI->getUniqueVRegDef(ImmReg.getReg());
          if (ImmDef && ImmDef->getOpcode() == RISCV::ADDI) {
           int64_t ImmValue = ImmDef->getOperand(2).getImm();
            LLVM_DEBUG(dbgs() <<"  Imm is: " <<*UseMI);
            MachineInstr *VADD = UseMI;
            MachineInstr *VOR_ForVADD = nullptr;
            Register VADDDest = VADD->getOperand(0).getReg();
            for (MachineInstr &UseMI : MRI->use_instructions(VADDDest)) {
              if (UseMI.getOpcode() == RISCV::PseudoVOR_VX_M8) {
                MachineOperand &SrcReg = UseMI.getOperand(2);
                LLVM_DEBUG(dbgs() <<"  Found VOR: " <<UseMI);
                VOR_ForVADD = &UseMI;
                break;
              }
            }
            if (!VOR_ForVADD) continue;
            Register BaseReg = VOR_Orig->getOperand(0).getReg();
	    MachineInstr *NewVADD = nullptr;
	    auto CreateNewVADD = [&](MachineInstr *OldVOR, MachineInstr *OldVADD) -> MachineInstr * {
              MachineInstrBuilder MIB = BuildMI(MBB, *OldVOR, OldVOR->getDebugLoc(),
                                                TII->get(RISCV::PseudoVADD_VX_M8));
              MIB.addReg(OldVOR->getOperand(0).getReg(), RegState::Define);
              MIB.addReg(OldVOR->getOperand(1).getReg(), RegState::Undef);
              MIB.addReg(BaseReg);
              MIB.addReg(OldVADD->getOperand(3).getReg());
              MIB.addReg(VLReg);
              MIB.addImm(5);
              MIB.addImm(1);
              MIB.copyImplicitOps(*OldVOR);
              return MIB;
            };
            NewVADD = CreateNewVADD(VOR_ForVADD, VADD);
            Register VORDest = VOR_ForVADD->getOperand(0).getReg();
            for (auto &UseMI : MRI->use_instructions(VORDest)) {
              if (UseMI.getOpcode() == RISCV::PseudoVMSLT_VX_M8) {
                MachineOperand &SrcReg = UseMI.getOperand(1);
                if (SrcReg.isReg() && SrcReg.getReg() == VORDest) {
                  LLVM_DEBUG(dbgs() <<"  Found VMSLT: " <<UseMI);
	          UseMI.getOperand(1).setReg(NewVADD->getOperand(0).getReg());
                  break;
                }
              }
            }
            VOR_ForVADD->eraseFromParent();
            VADD->eraseFromParent();
          }
	}
        LLVM_DEBUG(dbgs() << "Transformed pattern from A to B\n");
      }
    }
  }
}

bool ExpandPseudos::runSink(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "******** RISCV Sink ********\n");

  bool EverMadeChange = false;

  if (Sink)
    ProcessInSameBlock(MF);
  if (a) {
    dbgs()<<"RISCV Affine enabled.\n";
    ProcessInSameAffine(MF);
  }

  for (auto I : RegsToClearKillFlags)
    MRI->clearKillFlags(I);
  RegsToClearKillFlags.clear();

  return EverMadeChange;
}

bool ExpandPseudos::runOnMachineFunction(MachineFunction &MF) {
  if (!Sink && !a) {
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
