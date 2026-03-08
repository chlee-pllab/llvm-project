//===-- UniformityBeforePhiWrapper.h - RISC-V Uniformity Before PHI ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares a pass that computes MachineUniformityInfo *before*
/// PHI elimination, so it can be safely used by later passes like register
/// pressure analysis while the function is still in SSA form.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_UNIFORMITYBEFOREPHIWRAPPER_H
#define LLVM_LIB_TARGET_RISCV_UNIFORMITYBEFOREPHIWRAPPER_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"

namespace llvm {

class UniformityBeforePhiWrapper : public MachineFunctionPass {
  MachineUniformityInfo *MUI;

public:
  static char ID;

  UniformityBeforePhiWrapper();

  StringRef getPassName() const override {
    return "RISC-V Uniformity analysis (before PHI elimination)";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineUniformityInfo *getUniformityInfo();

private:
  void PrintRegDiv(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType);
  void PrintDiv(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs);
  void addToRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType="Nondiverg");
  bool findRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, int i=1, StringRef RegType="Nondiverg");
  void eraseRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsDivMap, SetVector<MachineInstr*> &WorkDivList, StringRef RegType="Nondiverg");
  void collectInsDiv(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsDivMap, StringRef BType="Entry");
  void findINC(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &VLs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseSet<Register> &Is, StringRef BType="Entry");
  void findDivergent(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &DivergentArrs, DenseSet<Register> &NondivergArrs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsDirMap, SetVector<MachineInstr*> &WorkDirList, DenseSet<Register> &AIs, StringRef BType="Entry");
  void detectDivergent(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &AIs, StringRef BType="Entry");
};

extern char &UniformityBeforePhiWrapperID;

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_UNIFORMITYBEFOREPHIWRAPPER_H
