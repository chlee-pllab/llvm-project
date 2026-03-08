//===-- UniformityBeforePhiWrapper.cpp - RISC-V Uniformity Before PHI ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UniformityBeforePhiWrapper.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

#define DEBUG_TYPE "uniformity-before-phi-wrapper"
#define RISCV_INSERT_VSETVLI_NAME "RISC-V Uniformity Wrapper pass"

char UniformityBeforePhiWrapper::ID = 0;
char &llvm::UniformityBeforePhiWrapperID = UniformityBeforePhiWrapper::ID;

UniformityBeforePhiWrapper::UniformityBeforePhiWrapper() : MachineFunctionPass(ID) {
  initializeUniformityBeforePhiWrapperPass(*PassRegistry::getPassRegistry());
}

void UniformityBeforePhiWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineUniformityAnalysisPass>();
  //AU.addRequired<MachineDominatorTreeWrapperPass>();
  //AU.addRequired<MachineCycleInfoWrapperPass>();
  //AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void UniformityBeforePhiWrapper::PrintRegDiv(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  //MachineRegisterInfo &MRI = MF.getRegInfo();
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
    if (RegType == "Divergent") return DivergentRegs;
    return NondivergRegs;
  }();
  dbgs()<<RegType<<" Registers: {";
  for (const auto &Reg : TargetRegs) {
    //Register Reg = MO.getReg();
    dbgs() << " "<< printReg(Reg, TRI);
  }
  dbgs()<<"}\n";
}

void UniformityBeforePhiWrapper::PrintDiv(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs) {
  for (MachineBasicBlock &MBB : MF) {
    dbgs() << "Return Print: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    for (MachineInstr &MI : MBB) {
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        bool s = findRegDiv(MI, DivergentRegs, NondivergRegs, i);
        bool a = s ? false : findRegDiv(MI, DivergentRegs, NondivergRegs, i, "Divergent");
        //if (!s && !a && !v) continue;
        //dbgs()<<"(s, a, v) = "<<"("<<s<<", "<<a<<", "<<v<<")\n";
        StringRef RegType = [&]() -> StringRef {
          if (s) { return "(U)"; }
          if (a) { return "(D)"; }
          return "";
        }();
        const MachineOperand &MO = MI.getOperand(i);
        if (!(MI.getOpcode() == RISCV::PseudoVSE32_V_M8) && !MI.isBranch()) {
          if (i == 0) dbgs()<<"  "<<MO<<RegType<<" = "<<TII->getName(MI.getOpcode());
          else if (i == 1) dbgs()<<" "<<MO<<RegType;
          else dbgs()<<", "<<MO<<RegType;
        } else {
          if (i == 0) dbgs()<<"  "<<TII->getName(MI.getOpcode());
          else dbgs()<<",";
          dbgs()<<" "<<MO<<RegType;
        }
      }
      dbgs()<<"\n";
    }
  }
  dbgs()<<"--------------------------\n";
  dbgs()<<"UniformRegs (size: "<<NondivergRegs.size()<<")\n";
  dbgs()<<"DivergentRegs (size: "<<DivergentRegs.size()<<")\n";
  PrintRegDiv(MF, DivergentRegs, NondivergRegs, "Uniform");
  PrintRegDiv(MF, DivergentRegs, NondivergRegs, "Divergent");
}

void UniformityBeforePhiWrapper::addToRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType) {
  if (MI.getOperand(0).isReg()) {
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Divergent") return DivergentRegs;
      return NondivergRegs;
    }();
    TargetRegs.insert(MI.getOperand(0).getReg());
    dbgs()<<"    -> Add register "<<MI.getOperand(0)<<" to "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
  }
}

bool UniformityBeforePhiWrapper::findRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, int i, StringRef RegType) {
  if (MI.getOperand(i).isReg()) {
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Divergent") return DivergentRegs;
      return NondivergRegs;
    }();
    if (TargetRegs.count(MI.getOperand(i).getReg())) {
      //dbgs() << "    -> Found " <<RegType.str()<<" " << MI.getOperand(i) << " in op"<<i<<" \n";
      return true;
    } else {
      //dbgs() << "    -> Not found "<<RegType.str()<<" " << MI.getOperand(i) << " in op"<<i<<"\n";
      return false;
    }
  }
  return false;
}

void UniformityBeforePhiWrapper::eraseRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsDivMap, SetVector<MachineInstr*> &WorkDivList, StringRef RegType) {
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Divergent") return DivergentRegs;
      return NondivergRegs;
    }();
  TargetRegs.erase(MI.getOperand(0).getReg());
  dbgs()<<"    -> Erase "<<MI.getOperand(0)<<" from "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
  dbgs()<<"  Add to WorkDirList:\n";
  for (MachineInstr *MI : RegToInsDivMap[MI.getOperand(0).getReg()]) {
    WorkDivList.insert(MI);
    dbgs()<<"    -> Add "<<*MI;
  }
}

void UniformityBeforePhiWrapper::collectInsDiv(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsDivMap, StringRef BType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  dbgs() << "--------------------------------\nCollecting register instruction:\n";
  for (auto &MBB : MF) {
    if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return MapDiv: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      //dbgs() <<"  " <<MI;
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        if (MO.isReg()) {
          Register Reg = MO.getReg();
	  /*if (!RegToInsMap.count(Reg)) {
            RegToInsMap[keyReg] = DenseSet<MachineInstr*>();
	  }*/
	  if (i == 0 && !(MI.getOpcode() == RISCV::SW)) continue;
	  if (MI.isCopy()) {
            Register SrcReg = MI.getOperand(1).getReg();
            if (!(SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) && !(SrcReg == RISCV::X0)) {
              DenseSet<MachineInstr*> &insSet = RegToInsDivMap[Reg];
	      insSet.insert(&MI);
              //dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
            }
	  } else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB || MI.getOpcode() == RISCV::AND) {
            Register SrcReg = MI.getOperand(i).getReg();
            if (!(SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) && !(SrcReg == RISCV::X0)) {
              DenseSet<MachineInstr*> &insSet = RegToInsDivMap[Reg];
	      insSet.insert(&MI);
              //dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
            }
	  } else if (MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8) {
            if (i <= 3) {
              DenseSet<MachineInstr*> &insSet = RegToInsDivMap[Reg];
	      insSet.insert(&MI);
              //dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
	    }
          } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
            if (i <= 1) {
              DenseSet<MachineInstr*> &insSet = RegToInsDivMap[Reg];
	      insSet.insert(&MI);
              //dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
	    }
          } else if (MI.getOpcode() == RISCV::SW) {
            DenseSet<MachineInstr*> &insSet = RegToInsDivMap[Reg];
	    insSet.insert(&MI);
            //dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
          } else if (MI.isBranch()) {
	  } else {
            DenseSet<MachineInstr*> &insSet = RegToInsDivMap[Reg];
	    insSet.insert(&MI);
            //dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
	  }
        }
      }
    }
  }
}

void UniformityBeforePhiWrapper::findINC(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &VLs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseSet<Register> &Is, StringRef BType) {
  MachineSSAContext MSSA(&MF);
  MUI = &getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  dbgs() << "--------------------------------\nFinding INC(induction variable):\n";
  for (auto &MBB : MF) {
    if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return INC: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      //dbgs() <<"  " <<MI;
      if (MI.isCopy()) {
        if (MI.getOperand(1).isReg()) {
          dbgs() <<"  " <<MI;
          dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
          Register SrcReg = MI.getOperand(1).getReg();
          if (SrcReg >= RISCV::X10 && SrcReg <= RISCV::X12) {
            ARRs.insert(MI.getOperand(0).getReg());
            dbgs() << "    -> Add ARR "<<MI.getOperand(0)<<" to ARR (size: "<<ARRs.size()<<")\n";
          }
        }
      } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
        dbgs() <<"  " <<MI;
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
        VLs.insert(MI.getOperand(0).getReg());
        dbgs() << "    -> Add VL "<<MI.getOperand(0)<<" to VL (size: "<<VLs.size()<<")\n";
      } else if (MI.getOpcode() == RISCV::PseudoReadVLENB) {
        dbgs() <<"  " <<MI;
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
        VLs.insert(MI.getOperand(0).getReg());
        dbgs() << "    -> Add VL "<<MI.getOperand(0)<<" to VL (size: "<<VLs.size()<<")\n";
      } else if (MI.getOpcode() == RISCV::SLLI) {
        dbgs() <<"  " <<MI;
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
        if (VLs.count(MI.getOperand(1).getReg())) {
          //VLs.erase(MI.getOperand(1).getReg());
          VLs.insert(MI.getOperand(0).getReg());
          //dbgs() << "    -> Erase VL "<<MI.getOperand(1)<<"\n";
          dbgs() << "    -> Add VL "<<MI.getOperand(0)<<" to VL (size: "<<VLs.size()<<")\n";
        }
      } else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB) {
        dbgs() << "  Potential increment instruction:";
        dbgs() <<"  " <<MI;
        if (BType == "Body" && MI.getOperand(2).isReg()) {
          if (VLs.count(MI.getOperand(2).getReg())) {
            dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          } else if (MI.getOperand(1).isReg() && ARRs.count(MI.getOperand(1).getReg())) {
            AIs.insert(MI.getOperand(0).getReg());
            dbgs() << "    -> Add AI "<<MI.getOperand(0)<<" to AI (size: "<<AIs.size()<<")\n";
          }
        }
      } else if (MI.getOpcode() == RISCV::PseudoVADD_VX_M8) {
        dbgs() << "  Potential increment instruction:";
        dbgs() <<"  " <<MI;
        if (BType == "Body" && MI.getOperand(3).isReg() && VLs.count(MI.getOperand(3).getReg())) {
          dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
          addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
        }
      } else if (MI.getOpcode() == RISCV::LW) {
        //dbgs() <<"  " <<MI;
        //dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
        //ARRs.insert(MI.getOperand(1).getReg());
        //dbgs() << "    -> Add ARR "<<MI.getOperand(1)<<" to ARR (size: "<<ARRs.size()<<")\n";
      } else if (MI.getOpcode() == RISCV::ADDI) {
        dbgs() << "  Potential scalar increment instruction:";
        dbgs() <<"  " <<MI;
        if (BType == "Body" && MI.getOperand(1).isReg()) {
          Register Reg = MI.getOperand(1).getReg();
          if (ARRs.count(MI.getOperand(1).getReg())) {
            dbgs() << "    -> Found ARR " <<MI.getOperand(0)<<"\n";
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          } else if (AIs.count(MI.getOperand(1).getReg())) {
            AIs.insert(MI.getOperand(0).getReg());
            dbgs() << "    -> Add AI "<<MI.getOperand(0)<<" to AI (size: "<<AIs.size()<<")\n";
          } else if (Is.count(MI.getOperand(1).getReg())) {
            dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          }
        } else if (MI.getOperand(1).isReg()) {
          if (MI.getOperand(0).getReg() == MI.getOperand(1).getReg()) {
            dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          }
        }
      } else if (MI.isPHI()) {
        dbgs() <<"  " <<MI;
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
        const MachineOperand &ValMO = MI.getOperand(1);
        const MachineOperand &BBMO = MI.getOperand(2);
        if (ValMO.isReg() && BBMO.isMBB()) {
          dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
          addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          Is.insert(MI.getOperand(0).getReg());
          dbgs() << "    -> Add I "<<MI.getOperand(0)<<" to I (size: "<<Is.size()<<")\n";
        }
      } else if (MI.getOpcode() == RISCV::SLLIW) {
        dbgs() <<"  " <<MI;
        if (MI.getOperand(1).isReg()) {
          dbgs() << "    -> Found Pid " <<MI.getOperand(1)<<"\n";
          DivergentRegs.insert(MI.getOperand(1).getReg());
          dbgs()<<"    -> Add register "<<MI.getOperand(1)<<" to Divergent (size: "<<DivergentRegs.size()<<")\n";
        }
      }
    }
  }
}

void UniformityBeforePhiWrapper::findDivergent(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &DivergentArrs, DenseSet<Register> &NondivergArrs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsDirMap, SetVector<MachineInstr*> &WorkDirList, DenseSet<Register> &AIs, StringRef BType) {
  /*const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (auto &[Reg, insSet] : RegToInsDirMap) {
    if (!AffineRegs.count(Reg) && !ScalarRegs.count(Reg)) {
      VectorRegs.insert(Reg);
    } else {
      for (MachineInstr *MI : insSet) {
        WorkDirList.insert(MI);
        dbgs()<<"    -> Add "<<*MI;
      }
    }
  }
  dbgs()<<"--------------------------------\nFinding Type:\n";
  dbgs()<<"Return RegTypeDetection:\n";
  //PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, "Scalar");
  //PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, "Affine");
  //PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, "Vector");
  while (!WorkDirList.empty()) {
    MachineInstr *Cur = WorkDirList.front();
    MachineInstr &MI = *Cur;
    WorkDirList.erase(WorkDirList.begin());
    dbgs() <<"  " <<MI;
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    if (MI.isCopy()) {
      if (MI.getOperand(1).isReg()) {
        bool a0 = findRegDiv(MI,DivergentRegs,NondivergRegs,0,"Divergent");
        //bool v0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Vector");
	if (!a0) {
          Register SrcReg = MI.getOperand(1).getReg();
          if (findRegDiv(MI,DivergentRegs,NondivergRegs)) addToRegDiv(MI, DivergentRegs, NondivergRegs);
	}
      }
    } else if (MI.getOpcode() == RISCV::LUI) {
      addToRegDiv(MI, DivergentRegs, NondivergRegs);
    } else if (MI.getOpcode() == RISCV::ADDI || MI.getOpcode() == RISCV::SLLI || MI.getOpcode() == RISCV::SRLI) {
      bool a0 = findRegDiv(MI,DivergentRegs,NondivergRegs,0,"Divergent");
      bool s0 = findRegDiv(MI,DivergentRegs,NondivergRegs,0,"Nondiverg");
      if (!a0) {
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" constant "<<MI.getOperand(2)<<"\n";
        if (findRegDiv(MI,DivergentRegs,NondivergRegs,1,"Divergent")) {
          addToRegDiv(MI,DivergentRegs,NondivergRegs,"Divergent");
	} else if(findRegDiv(MI,ScalarRegs,AffineRegs,VectorRegs,1,"Divergent")) {
          addToRegDiv(MI,DivergentRegs,NondivergRegs,"Divergent");
	  eraseRegDiv(MI, DivergentRegs, NondivergRegs, RegToInsDirMap, WorkDirList);
	} else {
          addToRegDiv(MI, DivergentRegs, NondivergRegs);
          if (!s0) {
            eraseRegDiv(MI, DivergentRegs, NondivergRegs, RegToInsDivMap, WorkDivList);
          }
        }
      }
    } else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB || MI.getOpcode() == RISCV::AND ||
        MI.getOpcode() == RISCV::PseudoVADD_VV_M8 || MI.getOpcode() == RISCV::PseudoVADD_VX_M8 ||
        MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8 || MI.getOpcode() == RISCV::PseudoVSLIDEUP_VX_MF4) {
      auto processADD = [&](int i, bool c=false, bool add=false) {
        bool s0 = findRegDiv(MI,DivergentRegs,NondivergRegs,0);
        //bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        bool s1 = findRegDiv(MI, DivergentRegs, NondivergRegs, i);
        //bool a1 = s1 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Affine");
        //bool v1 = (s1 || a1) ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Vector");
        bool s2 = findReg(MI, DivergentRegs, NondivergRegs, i + 1);
        //bool a2 = s2 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1, "Affine");
        //bool v2 = (s2 || a2) ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1, "Vector");
        Register X1 = (MI.getOperand(i).getReg() == RISCV::X0);
        Register X2 = (MI.getOperand(i + 1).getReg() == RISCV::X0);
        if ((!s1 && !a1 && !X1) || (!s2 && !a2 && !X2)) {
          if (!a0 && !s0) addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Vector");
        } else if ((!s1 && a1) && (!s2 && a2) && c) {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Vector");
        } else if ((!s1 && a1) || (!s2 && a2)) {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
	  if (!a0) eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
        } else {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
	  if (!s0) eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
        }
      };
      if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB)
        processADD(1, false, true);
      else if (MI.getOpcode() == RISCV::AND)
        processADD(1, true);
      else if (MI.getOpcode() == RISCV::PseudoVADD_VV_M8 || MI.getOpcode() == RISCV::PseudoVADD_VX_M8 || MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8 || MI.getOpcode() == RISCV::PseudoVSLIDEUP_VX_MF4)
        processADD(2, false);
    } else if (MI.getOpcode() == RISCV::PseudoVOR_VI_M8) {
      bool s0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0);
      bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
      bool s1 = findReg(MI, ScalarRegs, AffineRegs, VectorRegs, 2);
      bool a1 = s1 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, 2, "Affine");
      bool s2 = findReg(MI, ScalarRegs, AffineRegs, VectorRegs, 3);
      bool a2 = s2 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, 3, "Affine");
      bool i2 = (MI.getOperand(3).isImm());
      Register X1 = (MI.getOperand(2).getReg() == RISCV::X0);
      Register X2 = i2 ? false : (MI.getOperand(3).getReg() == RISCV::X0);
      if ((!s1 && !a1 && !X1) || (!s2 && !a2 && !X2 && !i2)) {
        if (!a0 && !s0) addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Vector");
      } else if ((!s1 && a1)) {
        addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
	if (!a0) eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
      } else {
        addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
	if (!s0) eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
      }
    } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
    } else if (MI.getOpcode() == RISCV::LW || MI.getOpcode() == RISCV::PseudoVLE32_V_M8 || MI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK) {
      if (MI.getOperand(2).isReg()) {
        bool s0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0);
        bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        bool s1 = findReg(MI, ScalarArrs, AffineArrs, VectorArrs, 2);
        bool a1 = findReg(MI, ScalarArrs, AffineArrs, VectorArrs, 2, "Affine");
        if (AIs.count(MI.getOperand(2).getReg())) {
          if (!a0 && !s0) {
            if (s1) {
		    dbgs()<<"!!!!!!!!!!!!!s1\n\n\n";
              addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
              eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
	    } else if (a1) {
		    dbgs()<<"!!!!!!!!!!!!!a1\n\n\n";
              addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
	      eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
	    }
	  }
	}
      }
    } else if (MI.getOpcode() == RISCV::PseudoVID_V_M8) {
      if (MI.getOperand(2).isImm()) {
        dbgs() << "    -> op2 constant "<<MI.getOperand(2)<<"\n";
        bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        if (!a0) addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
      }
    } else if (MI.getOpcode() == RISCV::PseudoReadVLENB) {
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
    } else if (MI.getOpcode() == RISCV::PseudoVMSLTU_VX_M8) {
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Scalar");
    }
    dbgs()<<"-------------------------\n";
  }
  dbgs()<<"ScalarRegs (size: "<<ScalarRegs.size()<<")\n";
  dbgs()<<"AffineRegs (size: "<<AffineRegs.size()<<")\n";
  dbgs()<<"VectorRegs (size: "<<VectorRegs.size()<<")\n";
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, "Scalar");
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, "Affine");
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, "Vector");
  dbgs()<<"-----------------------------------\n";*/
}

void UniformityBeforePhiWrapper::detectDivergent(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &AIs, StringRef BType) {
  MachineSSAContext MSSA(&MF);
  MUI = &getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  dbgs() << "--------------------------------\nFinding Divergent:\n";
  for (auto &MBB : MF) {
    if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return Formal: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      dbgs() <<"  " <<MI;
      if (MI.isCopy()) {
        if (MI.getOperand(1).isReg()) {
          bool a0 = findRegDiv(MI,DivergentRegs,NondivergRegs,0,"Divergent");
          if (!a0) {
            Register SrcReg = MI.getOperand(1).getReg();
	    if (SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) {
              addToRegDiv(MI, DivergentRegs, NondivergRegs);
            } else if (SrcReg == RISCV::X0) {
              addToRegDiv(MI, DivergentRegs, NondivergRegs);
            } else {
              if (findRegDiv(MI,DivergentRegs,NondivergRegs)) addToRegDiv(MI, DivergentRegs, NondivergRegs);
            }
          }
        }
      } else if (MI.getOpcode() == RISCV::LUI) {
        if (MI.getOperand(1).isImm()) {
          addToRegDiv(MI, DivergentRegs, NondivergRegs);
        }
      } else if (MI.getOpcode() == RISCV::ADDI || MI.getOpcode() == RISCV::SLLI || MI.getOpcode() == RISCV::SRLI || MI.getOpcode() == RISCV::SLLIW) {
        bool a0 = findRegDiv(MI,DivergentRegs,NondivergRegs,0,"Divergent");
        if (!a0 /*|| BType != "Entry"*/) {
          dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" constant "<<MI.getOperand(2)<<"\n";
          if (findRegDiv(MI,DivergentRegs,NondivergRegs,1,"Divergent")) addToRegDiv(MI,DivergentRegs,NondivergRegs,"Divergent");
          else addToRegDiv(MI, DivergentRegs, NondivergRegs);
        }
      } else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB || MI.getOpcode() == RISCV::AND ||
          MI.getOpcode() == RISCV::PseudoVADD_VV_M8 || MI.getOpcode() == RISCV::PseudoVADD_VX_M8 ||
          MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8 || MI.getOpcode() == RISCV::PseudoVSLIDEUP_VX_MF4 ||
	  MI.getOpcode() == RISCV::PseudoVOR_VX_M8 || MI.getOpcode() == RISCV::PseudoVFADD_VV_M8_E32) {
        auto processADD = [&](int i, bool c=false) {
          bool s1 = findRegDiv(MI, DivergentRegs, NondivergRegs, i);
          //bool a1 = s1 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Affine");
          bool s2 = findRegDiv(MI, DivergentRegs, NondivergRegs, i + 1);
          //bool a2 = s2 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1, "Affine");
          Register X1 = (MI.getOperand(i).getReg() == RISCV::X0);
          Register X2 = (MI.getOperand(i + 1).getReg() == RISCV::X0);
          if ((!s1 && !X1) || (!s2  && !X2)) {
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          } else if ((!s1) && (!s2) && c) {
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          } else if ((!s1) || (!s2)) {
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
          } else {
            addToRegDiv(MI, DivergentRegs, NondivergRegs);
          }
        };
        if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB)
          processADD(1);
        else if (MI.getOpcode() == RISCV::AND)
          processADD(1, true);
        else if (MI.getOpcode() == RISCV::PseudoVADD_VV_M8 || MI.getOpcode() == RISCV::PseudoVADD_VX_M8 || MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8 || MI.getOpcode() == RISCV::PseudoVSLIDEUP_VX_MF4 || MI.getOpcode() == RISCV::PseudoVOR_VX_M8 || MI.getOpcode() == RISCV::PseudoVFADD_VV_M8_E32)
          processADD(2);
        if (MI.getOpcode() == RISCV::PseudoVOR_VX_M8) {
          bool s0 = findRegDiv(MI,DivergentRegs,NondivergRegs,1);
          bool a0 = findRegDiv(MI,DivergentRegs,NondivergRegs,1,"Divergent");
          if (!a0 && !s0) {
            DivergentRegs.insert(MI.getOperand(1).getReg());
            dbgs()<<"    -> Add register "<<MI.getOperand(1)<<" to Divergent (size: "<<DivergentRegs.size()<<")\n";
          }
	}
      } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
        if (MI.getOperand(2).isImm()) {
          dbgs() << "    -> op2 constant "<<MI.getOperand(2)<<"\n";
          addToRegDiv(MI, DivergentRegs, NondivergRegs);
        }
      } else if (MI.getOpcode() == RISCV::LW || MI.getOpcode() == RISCV::PseudoVLE32_V_M8 || MI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK) {
        if (MI.getOperand(1).isFI()) {
          addToRegDiv(MI, DivergentRegs, NondivergRegs);
        } else if (MI.getOperand(1).isFI()) {
          addToRegDiv(MI,DivergentRegs,NondivergRegs,"Divergent");
        } else {
          addToRegDiv(MI, DivergentRegs, NondivergRegs, "Divergent");
	}
	if (MI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK) {
          bool s0 = findRegDiv(MI,DivergentRegs,NondivergRegs,0);
          if (s0) {
            NondivergRegs.erase(MI.getOperand(0).getReg());
            dbgs()<<"    -> Erase "<<MI.getOperand(0)<<" from Uniform (size: "<<NondivergRegs.size()<<")\n";
          }
        }
      } else if (MI.getOpcode() == RISCV::SW || MI.getOpcode() == RISCV::PseudoVSE32_V_M8 || MI.getOpcode() == RISCV::PseudoVSE32_V_M8_MASK) {
        //dbgs() << "    -> op2 "<<MI.getOperand(2)<<"\n";
      } else if (MI.getOpcode() == RISCV::PseudoVID_V_M8 || MI.getOpcode() == RISCV::PseudoVMV_V_I_M8) {
        //dbgs() << "    -> op2 constant "<<MI.getOperand(2)<<"\n";
        addToRegDiv(MI, DivergentRegs, NondivergRegs);
      } else if (MI.getOpcode() == RISCV::PseudoReadVLENB) {
        //dbgs() << "    -> op0 "<<MI.getOperand(0)<<"\n";
        addToRegDiv(MI, DivergentRegs, NondivergRegs);
      } else if (MI.getOpcode() == RISCV::PseudoVMSLTU_VX_M8 || MI.getOpcode() == RISCV::PseudoVMSLT_VX_M8) {
        //dbgs() << "    -> op0 "<<MI.getOperand(0)<<"\n";
        addToRegDiv(MI, DivergentRegs, NondivergRegs);
      }
    }
    dbgs()<<"-------------------------\n";
  }
}

bool UniformityBeforePhiWrapper::runOnMachineFunction(MachineFunction &MF) {
  DenseSet<Register> DivergentRegs;
  DenseSet<Register> NondivergRegs;
  DenseSet<Register> ScalarArrs;
  DenseSet<Register> AffineArrs;
  DenseSet<Register> VectorArrs;
  DenseSet<Register> VLs;
  DenseSet<Register> ARRs;
  DenseSet<Register> AIs;
  DenseSet<Register> Is;
  DenseMap<Register, DenseSet<MachineInstr*>> RegToInsDivMap;

  MachineSSAContext MSSA(&MF);
  MUI = &getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  //collectInsDiv(MF, DivergentRegs, NondivergRegs, RegToInsDivMap);
  //findINC(MF,DivergentRegs,NondivergRegs,VLs,ARRs,AIs,Is);
  //detectDivergent(MF,DivergentRegs,NondivergRegs,AIs);
  //PrintDiv(MF, DivergentRegs, NondivergRegs);
  //auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  //auto &MCI = getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  //MUI = computeMachineUniformityInfo(MF, MCI, MDT, /*IgnoreCycles=*/false);
  /*dbgs() << "=== Uniformity Analysis ===\n";
  for (auto &MBB : MF) {
    dbgs() << "Return Map: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      dbgs() <<"  " <<MI;
      if (MI.isPHI()) {
        dbgs()<<"  -> MemPHI\n";
        if (MSSA.isConstantOrUndefValuePhi(MI)) {
          dbgs() << "  [Constant or undef PHI]\n";
        }
        for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
          const MachineOperand &ValMO = MI.getOperand(i);
          const MachineOperand &BBMO = MI.getOperand(i + 1);
          if (ValMO.isReg() && BBMO.isMBB()) {
            dbgs() << "  -> Operand: " << printReg(ValMO.getReg(),TRI)
                   << " from bb." << BBMO.getMBB()->getNumber() << "\n";
          }
        }
      }
      //dbgs() <<"  -> is divergent: "<<MUI->isDivergent(&MI)<<"\n";
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        if (MO.isReg()) {
          Register Reg = MO.getReg();
	  if (!Reg.isValid() || MRI.reg_nodbg_empty(Reg)) {
            dbgs() << "    -> Invalid or empty register: " <<printReg(Reg,TRI)<< "\n";
            continue;
          }
          //dbgs()<<"    -> "<<MO<<" is divergent: "<<MUI->isDivergent(Reg)<<"\n";
	  if (MRI.hasOneDef(Reg)) {
	    const MachineBasicBlock *DefBlock = MSSA.getDefBlock(Reg);
            if (!DefBlock)
              continue;
	    MachineInstr *DefMI = MRI.getVRegDef(Reg);
            dbgs()<<"   -> "<<printReg(Reg,TRI)<<" : bb."<<DefBlock->getNumber()<<", "<<*DefMI;
	  }
	  //for (const MachineInstr &UseMI : MRI.use_instructions(Reg)) {
          //  dbgs()<<"   -> "<<MO<<" use in "<<UseMI<<"\n";
          //}
	}
      }
    }
  }*/
  return false;
}

MachineUniformityInfo *UniformityBeforePhiWrapper::getUniformityInfo() {
  //assert(MUI && "UniformityInfo not computed");
  return MUI;
}

INITIALIZE_PASS(UniformityBeforePhiWrapper, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

//static RegisterPass<UniformityBeforePhiWrapper>
//    X("riscv-uniformity-before-phi",
//      "RISC-V Uniformity analysis (before PHI elimination)");
