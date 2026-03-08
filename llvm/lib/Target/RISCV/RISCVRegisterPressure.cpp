#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "UniformityBeforePhiWrapper.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
//#include "llvm/Analysis/UniformityAnalysis.h"
//#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAContext.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-register-pressure"
#define RISCV_INSERT_VSETVLI_NAME "RISCV-V Register Pressure pass"

namespace {
struct RegisterPressureHotSpot {
  MachineBasicBlock::const_iterator BeginPos;
  MachineBasicBlock::const_iterator EndPos;
  bool BeginClosed = false;
  bool EndClosed = false;
  bool BeginIsClosed() { return BeginClosed; }
  bool EndIsClosed() { return EndClosed; }
  bool IsClosed() { return BeginIsClosed() && EndIsClosed(); }
  void Init() {BeginClosed = EndClosed = false;}
};
static cl::opt<bool> EnableRegisterPressureOpt("riscv-enable-register-pressure-opt",                                                                                                                               
                                              cl::init(true), cl::Hidden);

class RISCVRegisterPressure : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;

public:
  enum DivergenceState {
    UNKNOWN = 0,
    NON_DIVERGENT = 1,
    DIVERGENT = 2
  };

  struct DRGNode {
    Value *IRValue = nullptr;
    MemoryAccess *MemAccess = nullptr;
    Instruction *I = nullptr;
    Register *Reg = nullptr;

    DRGNode(Value *V = nullptr) : IRValue(V) {}
    DRGNode(MemoryAccess *MA) : MemAccess(MA) {}
    DRGNode(Instruction *Ins) : I(Ins) {}
    DRGNode(Register *RegVal) : Reg(RegVal) {}

    bool isValid() const { return IRValue != nullptr || MemAccess != nullptr; }

    size_t getHash() const {
      if (IRValue)
        return llvm::hash_value(IRValue);
      if (MemAccess)
        return llvm::hash_value(MemAccess);
      return 0;
    }

    bool operator==(const DRGNode &Other) const {
      return IRValue == Other.IRValue && MemAccess == Other.MemAccess;
    }

    bool operator!=(const DRGNode &Other) const {
      return !(*this == Other);
    }

    void print(llvm::raw_ostream &OS, MachineFunction &MF) const {
      if (IRValue) {
        OS << "IRValue: ";
        IRValue->printAsOperand(OS);
      } else if (MemAccess) {
        OS << "MemoryAccess: ";
        MemAccess->print(OS);
      } else if (I) {
        OS << "I: ";
	I->print(OS);
      } else if (Reg) {
        const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
        OS << "Reg: ";
        printReg(*Reg, TRI);
      } else {
        OS << "InvalidDRGNode";
      }
    }
  };

  static char ID;

  RISCVRegisterPressure() : MachineFunctionPass(ID) {
    initializeRISCVRegisterPressurePass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    //AU.addRequired<MachineUniformityAnalysisPass>();
    //AU.addRequired<UniformityBeforePhiWrapper>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }
private:
  DRGNode getDRGNode(Value *V) { return DRGNode(V); }
  DRGNode getDRGNode(MemoryAccess *MA) { return DRGNode(MA); }
  DRGNode getDRGNode(Instruction *Ins) { return DRGNode(Ins); }
  //MachineUniformityInfo *MUI;
  const MachineLoopInfo *MLI;
  LiveIntervals *LIS;
  MemorySSA *MSSA;
  //MachineSSAContext *MSSA;

  DenseSet<Register> ScalarArrPars;
  DenseSet<Register> AffineArrPars;
  DenseSet<Register> VectorArrPars;

  bool enableRegisterPressureOpt() const;
  // get Loop MDNode
  MDNode *getLoopID(MachineLoop *ML) const;
  // get loop bool attribute
  std::optional<bool> getOptionalBoolLoopAttribute(MDNode *LoopID, StringRef Name);
  bool isVLoad(const MachineInstr &MI);
  bool hasHotSpotInLoop(MachineFunction &MF, MachineLoop *ML, RegisterPressureHotSpot &P);
  bool trySinkLoad(MachineFunction &MF, RegisterPressureHotSpot &P, std::vector<Register> &Regs, Register TargetReg);
  void PrintReg(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef RegType);
  void Print(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs);
  void PrintDiv(MachineFunction &MF, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs);
  void analyzePrint(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs);
  void analyzePrintDiv(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs);
  void addToArr(MachineFunction &MF, Register &ARR, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, StringRef RegType="Scalar");
  bool findArr(Register &ARR, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, StringRef RegType="Scalar");
  void addToReg(MachineInstr &MI, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, StringRef RegType="Scalar");
  bool findReg(MachineInstr &MI, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, int i=1, StringRef RegType="Scalar");
  void eraseReg(MachineInstr &MI, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseMap<Register,DenseSet<MachineInstr*>> &RegToInsMap, SetVector<MachineInstr*> &WorkList, StringRef RegType="Vector");
  void addToRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType="Uniform");
  bool findRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, int i=1, StringRef RegType="Uniform");
  void eraseRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType="Uniform");
  void addToRegPBD(MachineFunction &MF, Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef indent="  ", StringRef RegType="Uniform");
  bool findRegPBD(Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef RegType="Uniform");
  void eraseRegPBD(MachineFunction &MF, Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef RegType="Uniform");
  void collectIns(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsMap, StringRef BType="Entry");
  void collectDstIns(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseMap<Register, DenseSet<MachineInstr*>> &DstToInsMap, StringRef BType="Entry");
  void collectAIIns(MachineBasicBlock *MBB, DenseMap<Register, DenseSet<MachineInstr*>> &AIToInsMap, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, StringRef BType="Entry");
  void pasteArrType(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseMap<Register, DenseSet<MachineInstr*>> &AIToInsMap, StringRef BType="Entry");
  void findAffine(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &VLs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, StringRef BType="Entry");
  void findINC(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &VLs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseSet<Register> &Is, StringRef BType="Entry");
  void findScalar(MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, StringRef BType="Entry");
  void findVector(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsMap, SetVector<MachineInstr*> &WorkList, DenseSet<Register> &AIs, StringRef BType="Entry");
  void detectArrType(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseMap<Register, DenseSet<MachineInstr*>> &AIToInsMap, SetVector<MachineInstr*> InsList, StringRef BType="Entry");
  void recordArrType(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, StringRef BType="Entry");
  void detectRegType(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, StringRef BType="Entry");
  void collectMIToI(MachineFunction &MF, DenseMap<MachineInstr*, DenseSet<Instruction*>> &MIToIMap);
  void DetectAndPrintDivergence(MachineFunction &MF,StringRef BType="Entry");
  void PBDA(MachineFunction &MF,DenseSet<Register> &DivergRegs,DenseSet<Register> &UniformRegs,DenseSet<Register> &visit, DenseMap<Register, DenseSet<MachineInstr*>> &DstToInsMap, DenseMap<MachineInstr*, DenseSet<Instruction*>> &MIToIMap);
  void BuildPointsToOrAliased(DRGNode &u,DenseSet<DRGNode> &visitBuild, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates, StringRef indent);
  void BuildDRG(MachineFunction &MF, Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, DenseSet<Register> &visit, DenseMap<Register, DenseSet<MachineInstr*>> &DstToInsMap, DenseMap<MachineInstr*, DenseSet<Instruction*>> &MIToIMap, StringRef indent="  ");
  void BuildGate(DRGNode &u,DenseSet<DRGNode> &visitBuild, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates, StringRef indent);
  void BuildChi(DRGNode &u,DenseSet<DRGNode> &visitBuild, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates, StringRef indent);
  bool DivergenceStatePropagation(DRGNode &u, DenseSet<DRGNode> &VisitDRG, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates);
  void analyzeMachineLoop(MachineFunction &MF/*, MachineLoop *ML*/);
};

}

namespace llvm {
template<>
struct DenseMapInfo<RISCVRegisterPressure::DRGNode> {
  static inline RISCVRegisterPressure::DRGNode getEmptyKey() {
    return RISCVRegisterPressure::DRGNode(
      DenseMapInfo<Value*>::getEmptyKey());
  }

  static inline RISCVRegisterPressure::DRGNode getTombstoneKey() {
    return RISCVRegisterPressure::DRGNode(
      DenseMapInfo<Value*>::getTombstoneKey());
  }

  static unsigned getHashValue(const RISCVRegisterPressure::DRGNode &Node) {
    if (Node.IRValue)
      return DenseMapInfo<Value*>::getHashValue(Node.IRValue);
    if (Node.MemAccess)
      return DenseMapInfo<void*>::getHashValue(Node.MemAccess);
    return 0;
  }

  static bool isEqual(const RISCVRegisterPressure::DRGNode &LHS,
                      const RISCVRegisterPressure::DRGNode &RHS) {
    return LHS.IRValue == RHS.IRValue && LHS.MemAccess == RHS.MemAccess;
  }
};
}
char RISCVRegisterPressure::ID=0;
char &llvm::RISCVRegisterPressureID = RISCVRegisterPressure::ID;

INITIALIZE_PASS(RISCVRegisterPressure, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

bool RISCVRegisterPressure::enableRegisterPressureOpt() const {
  return EnableRegisterPressureOpt;
}

MDNode *RISCVRegisterPressure::getLoopID(MachineLoop *ML) const {
  MDNode *LoopID = nullptr;
  if (const auto *MBB = ML->findLoopControlBlock()) {
    // If there is a single latch block, then the metadata
    // node is attached to its terminating instruction.
    const auto *BB = MBB->getBasicBlock();
    if (!BB)
      return nullptr;
    if (const auto *TI = BB->getTerminator())
      LoopID = TI->getMetadata(LLVMContext::MD_loop);
  } else if (const auto *MBB = ML->getHeader()) {
    // There seem to be multiple latch blocks, so we have to
    // visit all predecessors of the loop header and check
    // their terminating instructions for the metadata.
    if (const auto *Header = MBB->getBasicBlock()) {
      // Walk over all blocks in the loop.
      for (const auto *MBB : ML->blocks()) {
        const auto *BB = MBB->getBasicBlock();
        if (!BB)
          return nullptr;
        const auto *TI = BB->getTerminator();
        if (!TI)
          return nullptr;
        MDNode *MD = nullptr;
        // Check if this terminating instruction jumps to the loop header.
        for (const auto *Succ : successors(TI)) {
          if (Succ == Header) {
            // This is a jump to the header - gather the metadata from it.
            MD = TI->getMetadata(LLVMContext::MD_loop);
            break;
          }
        }
        if (!MD)
          return nullptr;
        if (!LoopID)
          LoopID = MD;
        else if (MD != LoopID)
          return nullptr;
      }
    }
  }
  if (LoopID &&
      (LoopID->getNumOperands() == 0 || LoopID->getOperand(0) != LoopID))
    LoopID = nullptr;
  return LoopID;
}
std::optional<bool> RISCVRegisterPressure::getOptionalBoolLoopAttribute(MDNode *LoopID,
                                                       StringRef Name) {
  MDNode *MD = findOptionMDForLoopID(LoopID, Name);
  if (!MD)
    return std::nullopt;
  switch (MD->getNumOperands()) {
  case 1:
    // When the value is absent it is interpreted as 'attribute set'.
    return true;
  case 2:
    if (ConstantInt *IntMD =
            mdconst::extract_or_null<ConstantInt>(MD->getOperand(1).get()))
      return IntMD->getZExtValue();
    return true;
  }
  llvm_unreachable("unexpected number of options");
}
bool RISCVRegisterPressure::isVLoad(const MachineInstr &MI) {
  switch(MI.getOpcode()) {
    default:
      return false;
    case RISCV::VL1RE8_V:
    case RISCV::VL1RE16_V:
    case RISCV::VL1RE32_V:
    case RISCV::VL1RE64_V:
    case RISCV::VL2RE8_V:
    case RISCV::VL2RE16_V:
    case RISCV::VL2RE32_V:
    case RISCV::VL2RE64_V:
    case RISCV::VL4RE8_V:
    case RISCV::VL4RE16_V:
    case RISCV::VL4RE32_V:
    case RISCV::VL4RE64_V:
    case RISCV::VL8RE8_V:
    case RISCV::VL8RE16_V:
    case RISCV::VL8RE32_V:
    case RISCV::VL8RE64_V:
      return true;
  }
}
bool RISCVRegisterPressure::hasHotSpotInLoop(MachineFunction &MF, MachineLoop *ML, RegisterPressureHotSpot &P) {
  P.Init();
  for (const MachineBasicBlock* B: ML->getBlocks()) {
    dbgs()<<"Return enter a block\n";
    B->dump();
    // RegionPressure or IntervalPressure (need LIS)
    //RegionPressure Pressure;
    IntervalPressure Pressure;
    RegPressureTracker RPTracker(Pressure);
    RegisterClassInfo RegClassInfo;
    RegClassInfo.runOnMachineFunction(MF);
    RPTracker.init(&MF, &RegClassInfo, LIS, B, B->begin(),false, false);
    int t = 0;
    while (RPTracker.getPos() != B->end()) {
      dbgs() << t++<<": ===== Pressure At a MI =====\n";
      auto MBI = RPTracker.getPos();
      auto MII = MBI.getInstrIterator();
      RPTracker.advance();
      RPTracker.dump();
      //if (!MII.isEnd())
      if (MBI != B->begin()) {
        auto PMBI = prev_nodbg(MBI, B->begin());
        auto PMII = PMBI.getInstrIterator();
        dbgs() << "RP: prev MI " << *PMII << "\n";
      }
      dbgs() << "RP: cur MI " << *MII << "\n";
      ArrayRef<unsigned> PSet = RPTracker.getRegSetPressureAtPos();
      dbgs()<<"PSet[14]: "<<PSet[14]<<"\n";
      if (!P.BeginIsClosed() && PSet[14] > 32) {
        auto PMBI = prev_nodbg(MBI, B->begin());
        auto PMII = PMBI.getInstrIterator();
        dbgs() << "!BeginPos: prev MI " << *PMII << "\n";
        dbgs() << "!BeginPos: cur MI " << *MII << "\n";
        P.BeginPos = prev_nodbg(MBI, B->begin());;
        P.BeginClosed = true;
        dbgs() << "RP: P.BeginClosed set " << *P.BeginPos.getInstrIterator() << "\n";
      }
      if (P.BeginIsClosed() && !P.EndIsClosed() && PSet[14] <= 32) {
        auto PMBI = prev_nodbg(MBI, B->begin());
        auto PMII = PMBI.getInstrIterator();
        dbgs() << "!BeginPos: prev MI " << *PMII << "\n";
        dbgs() << "!BeginPos: cur MI " << *MII << "\n";
        P.EndPos = MBI;
        P.EndClosed = true;
        dbgs() << "RP: P.EndClosed set " << *P.EndPos.getInstrIterator() << "\n";
        // find a hot spot, so return
        return true;
      }
      dbgs() << "============================\n";
    }

    RPTracker.closeBottom();
    RPTracker.dump();
    dbgs()<<"Return end of block\n";
    //LLVM_DEBUG(dbgs() << "isTopClosed:" << RPTracker.isTopClosed() << "\n");
    //LLVM_DEBUG(dbgs() << "isBottomClosed:" << RPTracker.isBottomClosed() << "\n");
    //RPTracker.closeRegion();
    //std::vector<unsigned> Pr, MPr;
    //RPTracker.getUpwardPressure(&(*(B->begin())), Pr, MPr);
    //dumpRegSetPressure(RPTracker.getRegSetPressureAtPos(), MF.getSubtarget().getRegisterInfo());
  }
  return false;
}

bool RISCVRegisterPressure::trySinkLoad(MachineFunction &MF, RegisterPressureHotSpot &P, std::vector<Register> &Regs, Register TargetReg) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  SlotIndex BeginIndex = LIS->getInstructionIndex(*P.BeginPos), EndIndex = LIS->getInstructionIndex(*P.EndPos);

  dbgs() << "RP: trySinkLoad: TargetReg:" << printReg(TargetReg, TRI, 0, &MRI) << "\n";
  dbgs() << "RP: trySinkLoad: IndexRange: (" << BeginIndex << ", " << EndIndex << ")\n";

  llvm::MachineBasicBlock::iterator MLI;
  MachineBasicBlock *MB = LIS->getInstructionFromIndex(BeginIndex)->getParent();
  //SmallVector<Register, 16> UsedRegs;

  dbgs() << "RP: MII: new\n";
  // for (auto MII = MB->begin(); MII != MB->end(); MII++) {
  //   for (const MachineOperand & MO : MII->operands()) {
  //     if (!MO.isReg() || MO.getReg() == 0)
  //       continue;
  //     Register Reg = MO.getReg();
  //     if (!is_contained(UsedRegs, Reg))
  //       UsedRegs.push_back(Reg);
  //   }
  //   if (MII->definesRegister(TargetReg))
  //     LLVM_DEBUG(dbgs() << "RP: MII DEFINE REG:" << *MII << "\n");
  //   else if (MII->readsVirtualRegister(TargetReg))
  //     LLVM_DEBUG(dbgs() << "RP: MII READ REG:" << *MII << "\n");
  // }
  


  //for (auto MII = MRI.reg_instr_nodbg_begin(TargetReg); MII != MRI.reg_instr_nodbg_end(); MII++) {
  bool prevIsVLoad = false;
  bool hasMLI = false;
  for (auto MII = MB->begin(); MII != MB->end(); MII++) {
    SlotIndex MIIIndex = LIS->getInstructionIndex(*MII);
    dbgs() << "RP: MII:" << *MII << "\n";
    dbgs() << "Sink RP: MII Opcode:" << MII->getOpcode() << "\n";
    dbgs()<<"-------------------------\n";
    if (MII->definesRegister(TargetReg, TRI) && isVLoad(*MII)) {
      if (isVLoad(*MII)) {
        dbgs() << "RP: VLOAD:" << *MII << "\n";
        MLI = MII;
        prevIsVLoad = true;
        hasMLI = true;
      } else {
        prevIsVLoad = false;
        hasMLI = false;
      }
    } else if(MII->readsVirtualRegister(TargetReg)) {
      dbgs() << "RP: check Sink or ReLoad\n";

      if (prevIsVLoad && (MIIIndex > BeginIndex)) {
        // Sink Load
        // LOAD <- MLI
        // ...
        // USE  <- MII
        // ...
        // USE
        // MachineBasicBlock::iterator WhereIter = (MIIIndex < EndIndex) ? LIS->getInstructionFromIndex(MIIIndex)->getIterator() : LIS->getInstructionFromIndex(EndIndex)->getIterator();
        dbgs() << "RP: do sink load\n";
        MachineBasicBlock::iterator WhereIter = LIS->getInstructionFromIndex(MIIIndex)->getIterator();
        auto *MB = MII->getParent();
        dbgs() << "RP: MOVE " << *MLI << "TO " << *MII << "\n";
        MB->splice(WhereIter, MB, MLI->getIterator());

        LIS->handleMove(*MLI);
        return true;
      } else if ((MIIIndex > BeginIndex) && hasMLI) {
        // Reload
        // LOAD <- MLI
        // ...
        // USE
        // ...
        // USE  < - MII
        const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();                                                                                                                                         
        const RISCVInstrInfo *TII = ST.getInstrInfo();
        auto Opcode = MLI->getOpcode();
        DebugLoc DL;
        for (unsigned i = 0; i < MLI->getNumOperands(); i++) {
          dbgs() << "RP: operand: " << i << " " << MLI->getOperand(i) << "\n";
        }
        dbgs() << "RP: do reload\n";

        //auto newReg = MRI.cloneVirtualRegister(MLI->getOperand(0).getReg());
        auto newMI = BuildMI(*MII->getParent(), MII->getIterator(), DL, TII->get(Opcode), MLI->getOperand(0).getReg());

        // auto newMI = BuildMI(*MII->getParent(), MII->getIterator(), DL, TII->get(Opcode), newReg);
        newMI.addUse(MLI->getOperand(1).getReg());
        LIS->InsertMachineInstrInMaps(*newMI.getInstr());

        // LIS->getRegUnit(newReg);
        LIS->removeInterval(MLI->getOperand(0).getReg());
        LIS->removeInterval(MLI->getOperand(1).getReg());

        //LIS->removeRegUnit(MLI->getOperand(1).getReg());
        LIS->getInterval(MLI->getOperand(0).getReg());
        LIS->getInterval(MLI->getOperand(1).getReg());
        //LIS->getInterval(MLI->getOperand(1).getReg());
        //LIS->createAndComputeVirtRegInterval(newReg);
        //LLVM_DEBUG(dbgs() << "RP: newMI" << *newMI << "\n");
        //newMI->addOperand(
        // LIS->repairIntervalsInRange(MB, MB->begin(), MB->end(), UsedRegs);
        //LIS->releaseMemory();
        //LIS->runOnMachineFunction(MF);
        
        return true;
      } else {
        prevIsVLoad = false;
      }
    }
  }

  // iterate over all the def_instr and find a LOAD instr
  /*
  for (auto MII = MRI.reg_instr_nodbg_begin(TargetReg); MII != MRI.reg_instr_nodbg_end(); MII++) {
    auto isLoad = isVLoad(*MII);
    SlotIndex MIIIndex = LIS->getInstructionIndex(*MII);
    LLVM_DEBUG(dbgs() << "RP: trySinkLoad: "<< *MII << isLoad << "\n");

    if (!isLoad)
      continue;

    auto ToIter = LIS->getInstructionFromIndex(EndIndex)->getIterator();
    auto nextMII = std::next(MII);
    if (!nextMII.atEnd() && LIS->getInstructionIndex(*nextMII) < EndIndex) {
      ToIter = (*nextMII).getIterator();
    }

    LLVM_DEBUG(dbgs() << "RP: trySinkLoad: move " << *MII << "to " << *ToIter);
    auto *MB = MII->getParent();
    MB->splice(ToIter, MB, MII->getIterator());

    LIS->handleMove(*MII);
    return true;
  }
  */

  return false;
}

void RISCVRegisterPressure::PrintReg(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef RegType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  //MachineRegisterInfo &MRI = MF.getRegInfo();
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
    if (RegType == "Affine") return AffineRegs;
    if (RegType == "Vector") return VectorRegs;
    if (RegType == "Scalar") return ScalarRegs;
    if (RegType == "Diverg") return DivergRegs;
    return UniformRegs;
  }();
  dbgs()<<RegType<<" Registers: {";
  for (const auto &Reg : TargetRegs) {
    //Register Reg = MO.getReg();
    dbgs() << " "<< printReg(Reg, TRI);
  }
  dbgs()<<"}\n";
}

void RISCVRegisterPressure::Print(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (auto &MBB : MF) {
    dbgs() << "Return Print: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
      bool s = findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i);
      bool a = s ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Affine");
      bool v = (s || a) ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Vector");
      //if (!s && !a && !v) continue;
      //dbgs()<<"(s, a, v) = "<<"("<<s<<", "<<a<<", "<<v<<")\n";
      StringRef RegType = [&]() -> StringRef {
        if (s) { return "(S)"; }
        if (a) { return "(A)"; }
        if (v) { return "(V)"; }
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
}

void RISCVRegisterPressure::PrintDiv(MachineFunction &MF, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (auto &MBB : MF) {
    dbgs() << "Return PrintDiv: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
      bool s = findRegDiv(MI, DivergRegs, UniformRegs, i);
      bool a = s ? false : findRegDiv(MI, DivergRegs, UniformRegs, i, "Diverg");
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
}

void RISCVRegisterPressure::analyzePrint(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs) {
  dbgs() << "--------------------------------\n";
  Print(MF, ScalarRegs, AffineRegs, VectorRegs);
  dbgs()<<"ScalarRegs (size: "<<ScalarRegs.size()<<")\n";
  dbgs()<<"AffineRegs (size: "<<AffineRegs.size()<<")\n";
  dbgs()<<"VectorRegs (size: "<<VectorRegs.size()<<")\n";
  dbgs()<<"--------------------------\n";
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, DivergRegs, UniformRegs, "Scalar");
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, DivergRegs, UniformRegs, "Affine");
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, DivergRegs, UniformRegs, "Vector");
  dbgs()<<"\n";
}

void RISCVRegisterPressure::analyzePrintDiv(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs) {
  dbgs() << "--------------------------------\n";
  PrintDiv(MF, DivergRegs, UniformRegs);
  dbgs()<<"DivergRegs (size: "<<DivergRegs.size()<<")\n";
  dbgs()<<"UniformRegs (size: "<<UniformRegs.size()<<")\n";
  dbgs()<<"--------------------------\n";
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, DivergRegs, UniformRegs, "Diverg");
  PrintReg(MF, ScalarRegs, AffineRegs, VectorRegs, DivergRegs, UniformRegs, "Uniform");
  dbgs()<<"\n";
}

void RISCVRegisterPressure::addToArr(MachineFunction &MF, Register &ARR, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, StringRef RegType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
    if (RegType == "Affine") return AffineArrs;
    if (RegType == "Vector") return VectorArrs;
    return ScalarArrs;
  }();
  TargetRegs.insert(ARR);
  dbgs()<<"    -> Add ARR "<<printReg(ARR, TRI)<<" to "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
}

bool RISCVRegisterPressure::findArr(Register &ARR, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, StringRef RegType) {
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
    if (RegType == "Affine") return AffineArrs;
    if (RegType == "Vector") return VectorArrs;
    return ScalarArrs;
  }();
  if (TargetRegs.count(ARR)) {
    //dbgs() << "    -> Found " <<RegType.str()<<" " << MI.getOperand(i) << " in op"<<i<<" \n";
    return true;
  } else {
    //dbgs() << "    -> Not found "<<RegType.str()<<" " << MI.getOperand(i) << " in op"<<i<<"\n";
    return false;
  }
  return false;
}

void RISCVRegisterPressure::addToReg(MachineInstr &MI, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, StringRef RegType) {
  if (MI.getOperand(0).isReg()) {
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Affine") return AffineRegs;
      if (RegType == "Vector") return VectorRegs;
      return ScalarRegs;
    }();
    TargetRegs.insert(MI.getOperand(0).getReg());
    dbgs()<<"    -> Add register "<<MI.getOperand(0)<<" to "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
  }
}

bool RISCVRegisterPressure::findReg(MachineInstr &MI, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, int i, StringRef RegType) {
  if (MI.getOperand(i).isReg()) {
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Affine") return AffineRegs;
      if (RegType == "Vector") return VectorRegs;
      return ScalarRegs;
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

void RISCVRegisterPressure::eraseReg(MachineInstr &MI, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsMap, SetVector<MachineInstr*> &WorkList, StringRef RegType) {
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Affine") return AffineRegs;
      if (RegType == "Vector") return VectorRegs;
      return ScalarRegs;
    }();
  TargetRegs.erase(MI.getOperand(0).getReg());
  dbgs()<<"    -> Erase "<<MI.getOperand(0)<<" from "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
  dbgs()<<"  Add to WorkList:\n";
  for (MachineInstr *MI : RegToInsMap[MI.getOperand(0).getReg()]) {
    WorkList.insert(MI);
    dbgs()<<"    -> Add "<<*MI;
  }
}

void RISCVRegisterPressure::addToRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType) {
  if (MI.getOperand(0).isReg()) {
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Diverg") return DivergentRegs;
      return NondivergRegs;
    }();
    TargetRegs.insert(MI.getOperand(0).getReg());
    dbgs()<<"    -> Add register "<<MI.getOperand(0)<<" to "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
  }
}

bool RISCVRegisterPressure::findRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, int i, StringRef RegType) {
  if (MI.getOperand(i).isReg()) {
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Diverg") return DivergentRegs;
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

void RISCVRegisterPressure::eraseRegDiv(MachineInstr &MI, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, StringRef RegType) {
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Diverg") return DivergentRegs;
      return NondivergRegs;
    }();
  TargetRegs.erase(MI.getOperand(0).getReg());
  dbgs()<<"    -> Erase "<<MI.getOperand(0)<<" from "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
}

void RISCVRegisterPressure::addToRegPBD(MachineFunction &MF, Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef indent, StringRef RegType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Diverg") return DivergRegs;
      return UniformRegs;
    }();
    TargetRegs.insert(u);
    dbgs()<<indent.str()<<"-> Add register "<<printReg(u, TRI)<<" to "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
}

bool RISCVRegisterPressure::findRegPBD(Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef RegType) {
    DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Diverg") return DivergRegs;
      return UniformRegs;
    }();
    if (TargetRegs.count(u)) {
      //dbgs() << "    -> Found " <<RegType.str()<<" " << MI.getOperand(i) << " in op"<<i<<" \n";
      return true;
    } else {
      //dbgs() << "    -> Not found "<<RegType.str()<<" " << MI.getOperand(i) << " in op"<<i<<"\n";
      return false;
    }
  return false;
}

void RISCVRegisterPressure::eraseRegPBD(MachineFunction &MF, Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, StringRef RegType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  DenseSet<Register>& TargetRegs = [&]() -> DenseSet<Register>& {
      if (RegType == "Diverg") return DivergRegs;
      return UniformRegs;
    }();
  TargetRegs.erase(u);
  dbgs()<<"    -> Erase "<<printReg(u, TRI)<<" from "<<RegType.str()<<" (size: "<<TargetRegs.size()<<")\n";
}

void RISCVRegisterPressure::collectIns(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsMap, StringRef BType) {
  dbgs() << "--------------------------------\nCollecting register instruction:\n";
  for (auto &MBB : MF) {
    if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return Map: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    dbgs() <<"  " <<MI;
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
            DenseSet<MachineInstr*> &insSet = RegToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
          }
	} else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB || MI.getOpcode() == RISCV::AND) {
          Register SrcReg = MI.getOperand(i).getReg();
          if (!(SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) && !(SrcReg == RISCV::X0)) {
            DenseSet<MachineInstr*> &insSet = RegToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
          }
	} else if (MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8) {
          if (i <= 3) {
            DenseSet<MachineInstr*> &insSet = RegToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
	  }
        } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
          if (i <= 1) {
            DenseSet<MachineInstr*> &insSet = RegToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
	  }
        } else if (MI.getOpcode() == RISCV::SW) {
          DenseSet<MachineInstr*> &insSet = RegToInsMap[Reg];
	  insSet.insert(&MI);
          dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
        } else if (MI.isBranch()) {
	} else {
          DenseSet<MachineInstr*> &insSet = RegToInsMap[Reg];
	  insSet.insert(&MI);
          dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
	}
      }
    }
  }
  }
}

void RISCVRegisterPressure::collectDstIns(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseMap<Register, DenseSet<MachineInstr*>> &DstToInsMap, StringRef BType) {
  dbgs() << "--------------------------------\nCollecting dst instruction:\n";
  for (auto &MBB : MF) {
    if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return DstMap: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    dbgs() <<"  " <<MI;
    if (MI.getNumOperands() > 0) {
      const MachineOperand &MO = MI.getOperand(0);
      if (MO.isReg()) {
        Register Reg = MO.getReg();
	/*if (!RegToInsMap.count(Reg)) {
          RegToInsMap[keyReg] = DenseSet<MachineInstr*>();
	}*/
	//if (i == 0 && !(MI.getOpcode() == RISCV::SW)) continue;
	if (MI.isCopy()) {
          Register SrcReg = MI.getOperand(0).getReg();
          if (!(SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) && !(SrcReg == RISCV::X0)) {
            DenseSet<MachineInstr*> &insSet = DstToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
          }
	} else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB || MI.getOpcode() == RISCV::AND) {
          Register SrcReg = MI.getOperand(0).getReg();
          if (!(SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) && !(SrcReg == RISCV::X0)) {
            DenseSet<MachineInstr*> &insSet = DstToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
          }
	} else if (MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8) {
          DenseSet<MachineInstr*> &insSet = DstToInsMap[Reg];
	  insSet.insert(&MI);
          dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
        } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
          DenseSet<MachineInstr*> &insSet = DstToInsMap[Reg];
	  insSet.insert(&MI);
          dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
        } else if (MI.getOpcode() == RISCV::SW) {
          DenseSet<MachineInstr*> &insSet = DstToInsMap[Reg];
	  insSet.insert(&MI);
          dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
        } else if (MI.isBranch()) {
	} else {
          DenseSet<MachineInstr*> &insSet = DstToInsMap[Reg];
	  insSet.insert(&MI);
          dbgs() << "    -> Add I to "<<MO<<" insSet (size: "<<insSet.size()<<")\n";
	}
      }
    }
  }
  }
}

void RISCVRegisterPressure::collectAIIns(MachineBasicBlock *MBB, DenseMap<Register, DenseSet<MachineInstr*>> &AIToInsMap, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, StringRef BType) {
  dbgs() << "Return AI Map: bb."<<MBB->getNumber()<<"." << MBB->getName() << "\n";
  for (MachineInstr &MI : *MBB) {
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    //dbgs() <<"  " <<MI;
    for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
      const MachineOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        Register Reg = MO.getReg();
	if (i >= 1) continue;
	if (MI.isCopy()) {
          Register SrcReg = MI.getOperand(0).getReg();
          if (AIs.count(SrcReg)) {
            DenseSet<MachineInstr*> &insSet = AIToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" AIinsSet (size: "<<insSet.size()<<")\n";
          }
	} else if (MI.getOpcode() == RISCV::ADD) {
          Register SrcReg = MI.getOperand(1).getReg();
          if (ARRs.count(SrcReg)) {
            DenseSet<MachineInstr*> &insSet = AIToInsMap[Reg];
	    insSet.insert(&MI);
            dbgs() << "    -> Add I to "<<MO<<" AIinsSet (size: "<<insSet.size()<<")\n";
          }
        } else if (MI.getOpcode() == RISCV::PseudoVSE32_V_M8) {
	}
      }
    }
  }
}

void RISCVRegisterPressure::pasteArrType(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseMap<Register, DenseSet<MachineInstr*>> &AIToInsMap, StringRef BType) {
  dbgs()<<"Return ArrTypePaste: bb."<<MBB->getNumber()<<"."<<MBB->getName()<<"\n";
  for (MachineInstr &MI : *MBB) {
    dbgs() <<"  " <<MI;
    if (MI.isCopy()) {
      if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
        Register SrcReg = MI.getOperand(1).getReg();
        if (SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) {
          bool s1 = findArr(SrcReg,ScalarArrPars,AffineArrPars,VectorArrPars);
          bool a1 = findArr(SrcReg,ScalarArrPars,AffineArrPars,VectorArrPars,"Affine");
          bool v1 = findArr(SrcReg,ScalarArrPars,AffineArrPars,VectorArrPars,"Vector");
          if (s1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs);
	  else if (a1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs,"Affine");
          else if (v1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs,"Vector");
        }
      }
    } else if (MI.getOpcode() == RISCV::ADD) {
      if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
	Register Reg = MI.getOperand(0).getReg();
        Register SrcReg = MI.getOperand(1).getReg();
        if (AIs.count(Reg) && ARRs.count(SrcReg)) {
          bool s1 = findArr(SrcReg,ScalarArrs,AffineArrs,VectorArrs);
          bool a1 = findArr(SrcReg,ScalarArrs,AffineArrs,VectorArrs,"Affine");
          bool v1 = findArr(SrcReg,ScalarArrs,AffineArrs,VectorArrs,"Vector");
          if (s1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs);
	  else if (a1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs,"Affine");
	  else if (v1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs,"Vector");
	}
      }
    } else if (MI.getOpcode() == RISCV::ADDI) {
      if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
	Register Reg = MI.getOperand(0).getReg();
        Register SrcReg = MI.getOperand(1).getReg();
        if (AIs.count(SrcReg)) {
          bool s1 = findArr(SrcReg,ScalarArrs,AffineArrs,VectorArrs);
          bool a1 = findArr(SrcReg,ScalarArrs,AffineArrs,VectorArrs,"Affine");
          bool v1 = findArr(SrcReg,ScalarArrs,AffineArrs,VectorArrs,"Vector");
          if (s1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs);
	  else if (a1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs,"Affine");
          else if (v1) addToReg(MI,ScalarArrs,AffineArrs,VectorArrs,"Vector");
        }
      }
    } else if (MI.getOpcode() == RISCV::PseudoVLE32_V_M8) {
      if (MI.getOperand(0).isReg() && MI.getOperand(2).isReg()) {
        Register Reg = MI.getOperand(2).getReg();
        if (AIs.count(Reg)) {
          bool s2 = findArr(Reg,ScalarArrs,AffineArrs,VectorArrs);
          bool a2 = findArr(Reg,ScalarArrs,AffineArrs,VectorArrs,"Affine");
          bool v2 = findArr(Reg,ScalarArrs,AffineArrs,VectorArrs,"Vector");
          if (s2) addToReg(MI,ScalarRegs,AffineRegs,VectorRegs);
          else if (a2) addToReg(MI,ScalarRegs,AffineRegs,VectorRegs,"Affine");
          else if (v2) addToReg(MI,ScalarRegs,AffineRegs,VectorRegs,"Vector");
        }
      }
    }
  }
}

void RISCVRegisterPressure::findAffine(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &VLs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, StringRef BType) {
  dbgs() << "--------------------------------\nFinding Affine(induction variable):\n";
  for (auto &MBB : MF) {
    if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return Affine: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
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
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
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
        addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
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
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
	} else if (AIs.count(MI.getOperand(1).getReg())) {
          AIs.insert(MI.getOperand(0).getReg());
          dbgs() << "    -> Add AI "<<MI.getOperand(0)<<" to AI (size: "<<AIs.size()<<")\n";
	}
      } else if (MI.getOperand(1).isReg()) {
        if (MI.getOperand(0).getReg() == MI.getOperand(1).getReg()) {
          dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
	}
      }
    } else if (MI.getOpcode() == RISCV::SLLIW) {
      dbgs() <<"  " <<MI;
      if (MI.getOperand(1).isReg()) {
        dbgs() << "    -> Found Pid " <<MI.getOperand(1)<<"\n";
        AffineRegs.insert(MI.getOperand(1).getReg());
        dbgs()<<"    -> Add register "<<MI.getOperand(1)<<" to Affine (size: "<<AffineRegs.size()<<")\n";
      }
    }
  }
  }
}

void RISCVRegisterPressure::findINC(MachineFunction &MF, DenseSet<Register> &DivergentRegs, DenseSet<Register> &NondivergRegs, DenseSet<Register> &VLs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseSet<Register> &Is, StringRef BType) {
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
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Diverg");
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
          addToRegDiv(MI, DivergentRegs, NondivergRegs, "Diverg");
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
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Diverg");
          } else if (AIs.count(MI.getOperand(1).getReg())) {
            AIs.insert(MI.getOperand(0).getReg());
            dbgs() << "    -> Add AI "<<MI.getOperand(0)<<" to AI (size: "<<AIs.size()<<")\n";
          } else if (Is.count(MI.getOperand(1).getReg())) {
            dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Diverg");
          }
        } else if (MI.getOperand(1).isReg()) {
          if (MI.getOperand(0).getReg() == MI.getOperand(1).getReg()) {
            dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
            addToRegDiv(MI, DivergentRegs, NondivergRegs, "Diverg");
          }
        }
      } else if (MI.isPHI()) {
        dbgs() <<"  " <<MI;
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
        const MachineOperand &ValMO = MI.getOperand(1);
        const MachineOperand &BBMO = MI.getOperand(2);
        if (ValMO.isReg() && BBMO.isMBB()) {
          dbgs() << "    -> Found IndVar " <<MI.getOperand(0)<<"\n";
          addToRegDiv(MI, DivergentRegs, NondivergRegs, "Diverg");
          Is.insert(MI.getOperand(0).getReg());
          dbgs() << "    -> Add I "<<MI.getOperand(0)<<" to I (size: "<<Is.size()<<")\n";
        }
      } else if (MI.getOpcode() == RISCV::SLLIW) {
        dbgs() <<"  " <<MI;
        if (MI.getOperand(1).isReg()) {
          dbgs() << "    -> Found Pid " <<MI.getOperand(1)<<"\n";
          DivergentRegs.insert(MI.getOperand(1).getReg());
          dbgs()<<"    -> Add register "<<MI.getOperand(1)<<" to Diverg (size: "<<DivergentRegs.size()<<")\n";
        }
      }
    }
  }
}

void RISCVRegisterPressure::findScalar(MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, StringRef BType) {
  dbgs() << "Return Scalar: bb."<<MBB->getNumber()<<"." << MBB->getName() << "\n";
  for (MachineInstr &MI : *MBB) {
    dbgs() <<"  " <<MI;
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    //dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
    if (MI.isCopy()) {
      if (MI.getOperand(1).isReg()) {
        bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        bool v0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Vector");
        if (!a0 && !v0) {
          Register SrcReg = MI.getOperand(1).getReg();
          if (SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) {
            addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
          } else if (SrcReg == RISCV::X0) {
	    addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
          } else {
            if (findReg(MI,ScalarRegs,AffineRegs,VectorRegs)) addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
          }
	}
      }
    } else if (MI.getOpcode() == RISCV::LUI) {
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
    } else if (MI.getOpcode() == RISCV::ADDI || MI.getOpcode() == RISCV::SLLI || MI.getOpcode() == RISCV::SRLI) {
      bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
      if (!a0) {
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" constant "<<MI.getOperand(2)<<"\n";
      }
    } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
      if (MI.getOperand(2).isImm()) {
        dbgs() << "    -> op2 constant "<<MI.getOperand(2)<<"\n";
        addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Scalar");
      }
    } else if (MI.getOpcode() == RISCV::PseudoVID_V_M8 || MI.getOpcode() == RISCV::PseudoVMV_V_I_M8) {
      if (MI.getOperand(2).isImm()) {
        bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        if (!a0) {
          dbgs() << "    -> op2 constant "<<MI.getOperand(2)<<"\n";
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
	}
      }
    } else if (MI.getOpcode() == RISCV::PseudoReadVLENB) {
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
    } else if (MI.getOpcode() == RISCV::PseudoVMSLTU_VX_M8) {
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Scalar");
    }
    dbgs()<<"-------------------------\n";
  }
}

void RISCVRegisterPressure::findVector(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, DenseMap<Register, DenseSet<MachineInstr*>> &RegToInsMap, SetVector<MachineInstr*> &WorkList, DenseSet<Register> &AIs, StringRef BType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  dbgs() << "Return Vector: bb."<<MBB->getNumber()<<"." << MBB->getName() << "\n";
  for (auto &[Reg, insSet] : RegToInsMap) {
    if (!AffineRegs.count(Reg) && !ScalarRegs.count(Reg)) {
      VectorRegs.insert(Reg);
    } else {
      for (MachineInstr *MI : insSet) {
        WorkList.insert(MI);
        dbgs()<<"    -> Add "<<*MI;
      }
    }
  }
  dbgs()<<"--------------------------------\nFinding Type:\n";
  dbgs()<<"Return RegTypeDetection:\n";
  while (!WorkList.empty()) {
    MachineInstr *Cur = WorkList.front();
    MachineInstr &MI = *Cur;
    WorkList.erase(WorkList.begin());
    dbgs() <<"  " <<MI;
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    if (MI.isCopy()) {
      if (MI.getOperand(1).isReg()) {
        bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        bool v0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Vector");
	if (!a0 && !v0) {
          Register SrcReg = MI.getOperand(1).getReg();
          if (findReg(MI,ScalarRegs,AffineRegs,VectorRegs)) addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
	}
      }
    } else if (MI.getOpcode() == RISCV::LUI) {
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
    } else if (MI.getOpcode() == RISCV::ADDI || MI.getOpcode() == RISCV::SLLI || MI.getOpcode() == RISCV::SRLI) {
      bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
      bool s0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Scalar");
      if (!a0) {
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" constant "<<MI.getOperand(2)<<"\n";
        if (findReg(MI,ScalarRegs,AffineRegs,VectorRegs,1,"Vector")) {
          addToReg(MI,ScalarRegs,AffineRegs,VectorRegs,"Vector");
	} else if(findReg(MI,ScalarRegs,AffineRegs,VectorRegs,1,"Affine")) {
          addToReg(MI,ScalarRegs,AffineRegs,VectorRegs,"Affine");
	  eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
	} else {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
          if (!s0) {
            eraseReg(MI, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap, WorkList);
          }
        }
      }
    } else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB || MI.getOpcode() == RISCV::AND ||
        MI.getOpcode() == RISCV::PseudoVADD_VV_M8 || MI.getOpcode() == RISCV::PseudoVADD_VX_M8 ||
        MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8 || MI.getOpcode() == RISCV::PseudoVSLIDEUP_VX_MF4) {
      auto processADD = [&](int i, bool c=false, bool add=false) {
        bool s0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0);
        bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        bool s1 = findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i);
        bool a1 = s1 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Affine");
        //bool v1 = (s1 || a1) ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Vector");
        bool s2 = findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1);
        bool a2 = s2 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1, "Affine");
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
  dbgs()<<"-----------------------------------\n";
}

void RISCVRegisterPressure::detectArrType(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, DenseSet<Register> &ARRs, DenseSet<Register> &AIs, DenseMap<Register, DenseSet<MachineInstr*>> &AIToInsMap, SetVector<MachineInstr*> InsList, StringRef BType) {
  dbgs()<<"Return ArrTypeDetection: bb."<<MBB->getNumber()<<"."<<MBB->getName()<<"\n";
  for (MachineInstr &MI : *MBB) {
    dbgs() <<"  " <<MI;
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    if (MI.getOpcode() == RISCV::PseudoVSE32_V_M8) {
      if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
        Register Reg = MI.getOperand(1).getReg();
        if (AIs.count(Reg)) {
          bool s0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0);
          bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
          if (s0) addToArr(MF,Reg,ScalarArrs,AffineArrs,VectorArrs);
          else if (a0) addToArr(MF,Reg,ScalarArrs,AffineArrs,VectorArrs,"Affine");
          else addToArr(MF,Reg,ScalarArrs,AffineArrs,VectorArrs,"Vector");
          DenseSet<MachineInstr*> &insSet = AIToInsMap[Reg];
          dbgs()<<"    Add to InsList:\n";
          for (MachineInstr *MI : insSet) {
            InsList.insert(MI);
            dbgs()<<"    -> Add "<<*MI;
          }
	}
      }
    }
  }
  while (!InsList.empty()) {
    MachineInstr *Cur = InsList.front();
    MachineInstr &MI = *Cur;
    InsList.erase(InsList.begin());
    dbgs() <<"  " <<MI;
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    if (MI.isCopy()) {
    } else if (MI.getOpcode() == RISCV::ADD) {
      Register Reg = MI.getOperand(1).getReg();
      bool s0 = findReg(MI,ScalarArrs,AffineArrs,VectorArrs,0);
      bool a0 = findReg(MI,ScalarArrs,AffineArrs,VectorArrs,0,"Affine");
      if (s0) addToArr(MF,Reg,ScalarArrs,AffineArrs,VectorArrs);
      else if (a0) addToArr(MF,Reg,ScalarArrs,AffineArrs,VectorArrs,"Affine");
      else addToArr(MF,Reg,ScalarArrs,AffineArrs,VectorArrs,"Vector");
      /*Register insSet = AIToInsMap[Reg];
      dbgs()<<"    -> Add to InsList:\n";
      for (MachineInstr *MI : insSet) {
        InsList.insert(MI);
        dbgs()<<"    -> Add "<<*MI;
      }*/
    }
  }
}

void RISCVRegisterPressure::recordArrType(MachineFunction &MF, MachineBasicBlock *MBB, DenseSet<Register> &ScalarArrs, DenseSet<Register> &AffineArrs, DenseSet<Register> &VectorArrs, StringRef BType) {
  dbgs()<<"--------------------------------\nRecording Array type:\n";
  dbgs()<<"Return Cleanup: bb."<<MBB->getNumber()<<"."<<MBB->getName()<<"\n";
  for (MachineInstr &MI : *MBB) {
    dbgs() <<"  " <<MI;
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
    if (MI.isCopy()) {
      if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
        Register Reg = MI.getOperand(1).getReg();
        bool s1 = findArr(Reg,ScalarArrs,AffineArrs,VectorArrs);
        bool a1 = findArr(Reg,ScalarArrs,AffineArrs,VectorArrs,"Affine");
        Register SrcReg = MI.getOperand(0).getReg();
        if (SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) {
          dbgs() << "    -> Copy function parameter "<<MI.getOperand(1)<<"\n";
          if (s1) addToArr(MF,SrcReg,ScalarArrPars,AffineArrPars,VectorArrPars);
          else if (a1) addToArr(MF,SrcReg,ScalarArrPars,AffineArrPars,VectorArrPars,"Affine");
          else addToArr(MF,SrcReg,ScalarArrPars,AffineArrPars,VectorArrPars,"Vector");
	}
      }
    } else if (MI.getOpcode() == RISCV::PseudoCALL) {
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &MO = MI.getOperand(i);
	//dbgs()<<MO<<"\n";
      }
    }
  }
}

void RISCVRegisterPressure::detectRegType(MachineFunction &MF, DenseSet<Register> &ScalarRegs, DenseSet<Register> &AffineRegs, DenseSet<Register> &VectorRegs, StringRef BType) {
  dbgs() << "--------------------------------\nDetecting Register type:\n";
  for (auto &MBB : MF) {
    if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return Block: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
    dbgs() <<"  " <<MI;
    const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
    dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" instruction\n";
    if (MI.isCopy()) {
      if (MI.getOperand(1).isReg()) {
        bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
        bool v0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Vector");
	if (!a0 && !v0) {
          Register SrcReg = MI.getOperand(1).getReg();
          if (SrcReg >= RISCV::X10 && SrcReg <= RISCV::X17) {
            //dbgs() << "    -> Copy function parameter "<<MI.getOperand(1)<<"\n";
            addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
          } else if (SrcReg == RISCV::X0) {
            //dbgs() << "    -> Copy constant zero "<<MI.getOperand(1)<<"\n";
	    addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
          } else {
            if (findReg(MI,ScalarRegs,AffineRegs,VectorRegs)) addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
          }
	}
      }
    } else if (MI.getOpcode() == RISCV::LUI) {
      if (MI.getOperand(1).isImm()) {
        //dbgs() << "    -> LUI constant "<<MI.getOperand(1)<<"\n";
        addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
      }
    } else if (MI.getOpcode() == RISCV::ADDI || MI.getOpcode() == RISCV::SLLI || MI.getOpcode() == RISCV::SRLI || MI.getOpcode() == RISCV::SLLIW) {
      bool a0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Affine");
      if (!a0 /*|| BType != "Entry"*/) {
        dbgs() << "    -> "<<TII->getName(MI.getOpcode())<<" constant "<<MI.getOperand(2)<<"\n";
        if (findReg(MI,ScalarRegs,AffineRegs,VectorRegs,1,"Vector")) addToReg(MI,ScalarRegs,AffineRegs,VectorRegs,"Vector");
        else if(findReg(MI,ScalarRegs,AffineRegs,VectorRegs,1,"Affine")) addToReg(MI,ScalarRegs,AffineRegs,VectorRegs,"Affine");
        else addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
      }
    } else if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB || MI.getOpcode() == RISCV::AND ||
        MI.getOpcode() == RISCV::PseudoVADD_VV_M8 || MI.getOpcode() == RISCV::PseudoVADD_VX_M8 ||
        MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8 || MI.getOpcode() == RISCV::PseudoVSLIDEUP_VX_MF4 ||
	MI.getOpcode() == RISCV::PseudoVOR_VX_M8 || MI.getOpcode() == RISCV::PseudoVFADD_VV_M8_E32) {
      auto processADD = [&](int i, bool c=false) {
        bool s1 = findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i);
        bool a1 = s1 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Affine");
        //bool v1 = (s1 || a1) ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i, "Vector");
        bool s2 = findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1);
        bool a2 = s2 ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1, "Affine");
        //bool v2 = (s2 || a2) ? false : findReg(MI, ScalarRegs, AffineRegs, VectorRegs, i + 1, "Vector");
        Register X1 = (MI.getOperand(i).getReg() == RISCV::X0);
        Register X2 = (MI.getOperand(i + 1).getReg() == RISCV::X0);
        if ((!s1 && !a1 && !X1) || (!s2 && !a2 && !X2)) {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Vector");
        } else if ((!s1 && a1) && (!s2 && a2) && c) {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Vector");
        } else if ((!s1 && a1) || (!s2 && a2)) {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Affine");
        } else {
          addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
        }
      };
      if (MI.getOpcode() == RISCV::ADD || MI.getOpcode() == RISCV::SUB)
        processADD(1);
      else if (MI.getOpcode() == RISCV::AND)
        processADD(1, true);
      else if (MI.getOpcode() == RISCV::PseudoVADD_VV_M8 || MI.getOpcode() == RISCV::PseudoVADD_VX_M8 || MI.getOpcode() == RISCV::PseudoVSADDU_VX_M8 || MI.getOpcode() == RISCV::PseudoVSLIDEUP_VX_MF4 || MI.getOpcode() == RISCV::PseudoVOR_VX_M8 || MI.getOpcode() == RISCV::PseudoVFADD_VV_M8_E32)
        processADD(2);
    } else if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
      if (MI.getOperand(2).isImm()) {
        dbgs() << "    -> op2 constant "<<MI.getOperand(2)<<"\n";
        addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Scalar");
      }
    } else if (MI.getOpcode() == RISCV::LW || MI.getOpcode() == RISCV::PseudoVLE32_V_M8 || MI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK) {
      //dbgs() << "    -> op2 "<<MI.getOperand(2)<<"\n";
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Vector");
      if (MI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK) {
        bool s0 = findReg(MI,ScalarRegs,AffineRegs,VectorRegs,0,"Scalar");
	if (s0) {
          ScalarRegs.erase(MI.getOperand(0).getReg());
          dbgs()<<"    -> Erase "<<MI.getOperand(0)<<" from Scalar (size: "<<ScalarRegs.size()<<")\n";
	}
      }
    } else if (MI.getOpcode() == RISCV::SW || MI.getOpcode() == RISCV::PseudoVSE32_V_M8 || MI.getOpcode() == RISCV::PseudoVSE32_V_M8_MASK) {
      //dbgs() << "    -> op2 "<<MI.getOperand(2)<<"\n";
    } else if (MI.getOpcode() == RISCV::PseudoVID_V_M8 || MI.getOpcode() == RISCV::PseudoVMV_V_I_M8) {
      //dbgs() << "    -> op2 constant "<<MI.getOperand(2)<<"\n";
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
    } else if (MI.getOpcode() == RISCV::PseudoReadVLENB) {
      //dbgs() << "    -> op0 "<<MI.getOperand(0)<<"\n";
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs);
    } else if (MI.getOpcode() == RISCV::PseudoVMSLTU_VX_M8 || MI.getOpcode() == RISCV::PseudoVMSLT_VX_M8) {
      //dbgs() << "    -> op0 "<<MI.getOperand(0)<<"\n";
      addToReg(MI, ScalarRegs, AffineRegs, VectorRegs, "Scalar");
    }
  }
  dbgs()<<"-------------------------\n";
  }
}

void RISCVRegisterPressure::collectMIToI(MachineFunction &MF, DenseMap<MachineInstr*, DenseSet<Instruction*>> &MIToIMap) {
  dbgs() << "--------------------------------\nCollecting MI to instruction:\n";
  for (auto &MBB : MF) {
    //if (MBB.getName().ends_with(".body")) BType = "Body";
    dbgs() << "Return Block: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
      dbgs()<<"  "<<MI;
      if (auto DL = MI.getDebugLoc()) {
        for (BasicBlock &BB : *MSSA->getDomTree().getRoot()->getParent()) {
          for (Instruction &I : BB) {
            if (I.getDebugLoc() == DL) {
              dbgs() << "    -> Correspond to I: ";
              I.print(dbgs());dbgs()<<"\n";
              DenseSet<Instruction*> &insSet = MIToIMap[&MI];
	      insSet.insert(&I);
              //dbgs() << "    -> Add I to MI insSet (size: "<<insSet.size()<<")\n";
            }
          }
        }
      } else {
        dbgs() << "    -> No debug info for this instruction\n";
      }
      dbgs()<<"\n";
    }
    dbgs()<<"---------------------\n";
  }
  //MachineSSAContext MSSA(MF);
  //dbgs() << "=== Instruction-Level Uniformity Analysis ===\n";
  /*const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (auto &MBB : MF) {
    dbgs() << "Return Map: bb."<<MBB.getNumber()<<"." << MBB.getName() << "\n";
    for (MachineInstr &MI : MBB) {
      const TargetInstrInfo *TII = MI.getParent()->getParent()->getSubtarget().getInstrInfo();
      //dbgs() <<"  " <<MI;
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        if (MO.isReg()) {
          Register Reg = MO.getReg();*/
	  /*if (!RegToInsMap.count(Reg)) {
            RegToInsMap[keyReg] = DenseSet<MachineInstr*>();
	  }
	  //if (MRI.isSSA()) {
	    //dbgs()<<"    -> "<<MO<<" is divergent: "<<MUI->isDivergent(Reg)<<"\n";
	  //} else {
	  //  dbgs()<<"    -> "<<MO<<" is not SSA form!\n";
	  //}
	}
      }
    }
  }*/
  /*for (auto &MBB : MF) {
    if (const BasicBlock *BB = MBB.getBasicBlock()) {
      dbgs() << "Block: " << BB->getName() << "\n";
      for (const Instruction &I : *BB) {
        dbgs() << I << "\n";
        dbgs() << "    -> Is divergent: " << UI->isDivergent(&I) << "\n";
        if (const auto *BI = dyn_cast<BranchInst>(&I)) {
          if (BI->isConditional()) {
            dbgs() << "    -> Branch condition divergent: " 
                   << UI->isDivergent(BI->getCondition()) << "\n";
          } else {
            dbgs() << "    -> Unconditional branch\n";
          }
        }
        if (const auto *SI = dyn_cast<StoreInst>(&I)) {
          dbgs() << "    -> Store address divergent: " 
                 << UI->isDivergent(SI->getPointerOperand()) << "\n";
          dbgs() << "    -> Store value divergent: " 
                 << UI->isDivergent(SI->getValueOperand()) << "\n";
        }
        if (const auto *LI = dyn_cast<LoadInst>(&I)) {
          dbgs() << "    -> Load address divergent: " 
                 << UI->isDivergent(LI->getPointerOperand()) << "\n";
        }
        if (const auto *CI = dyn_cast<CallInst>(&I)) {
          dbgs() << "    -> Call instruction divergent: " 
                 << UI->isDivergent(CI) << "\n";
        }
        for (const Use &U : I.operands()) {
          if (UI->isDivergent(U.get())) {
            dbgs() << "    -> Operand divergent: " << *U.get() << "\n";
          }
        }
      }
    }
  }*/
  //dbgs() << "=== End Instruction-Level Analysis ===\n\n";
  dbgs() << "=== Value-Level Uniformity Analysis ===\n";
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned divergentRegs = 0;
  unsigned totalRegs = 0;
  for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    dbgs() << "  VReg " << printReg(Reg, TRI)<<"\n";
    /*if (MRI.reg_nodbg_empty(Reg))
      continue;
    totalRegs++;
    const Value *IRValue = nullptr;
    bool isDivergent = false;
    for (const MachineInstr &MI : MRI.reg_instructions(Reg)) {
      if (MI.isCall() || MI.mayLoad() || MI.mayStore()) {
        isDivergent = true;
        break;
      }
    }
    if (isDivergent) {
      divergentRegs++;*/
  }
  //dbgs() << "Summary: " << divergentRegs << "/" << totalRegs 
  //       << " virtual registers are divergent\n";
  dbgs() << "=== End Value-Level Analysis ===\n\n";
}

void RISCVRegisterPressure::DetectAndPrintDivergence(
    MachineFunction &MF,
    StringRef BType) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  dbgs() << "--------------------------------\nDetecting and print divergence:\n";
  for (auto &BB : *MSSA->getDomTree().getRoot()->getParent()) {
    dbgs() << "BB: " << BB.getName() << "\n";
    //if (!MSSA->getBlockAccesses(&BB)) continue;
    //for (const MemoryAccess &MA : *MSSA->getBlockAccesses(&BB)) {
    //dbgs()<<"-------------------------------------------------------------------------------------------\n";
    if (MemoryAccess *MA = MSSA->getMemoryAccess(&BB)) {
      if (MemoryPhi *MemPhi = dyn_cast<MemoryPhi>(MA)) {
	if (!MemPhi->getGamma()) dbgs() << "MemoryPhi: ";
	else dbgs() << "MemoryGamma: ";
        MemPhi->print(dbgs());dbgs()<<"\n";
        //dbgs() << "  Incoming values:\n";
        //for (unsigned i = 0, e = MemPhi->getNumIncomingValues(); i < e; ++i) {
        //  BasicBlock *IncomingBB = MemPhi->getIncomingBlock(i);
        //  MemoryAccess *IncomingValue = MemPhi->getIncomingValue(i);
        //  dbgs() << "    BB: " << IncomingBB->getName() << " -> ";
        //  if (IncomingValue) IncomingValue->print(dbgs());
        //  else dbgs() << "null";
        //  dbgs() << "\n";
        //}
      }
    }
    for (Instruction &I : BB) {
      if (MemoryAccess *MA = MSSA->getMemoryAccess(&I)) {
        MA->print(dbgs());dbgs()<<"\n";
	MemoryAccess *DefiningAccess = nullptr;
        if (auto *MemUse = dyn_cast<MemoryUse>(MA)) DefiningAccess = MemUse->getDefiningAccess();
        else if (auto *MemDef = dyn_cast<MemoryDef>(MA)) DefiningAccess = MemDef->getDefiningAccess();
        else if (auto *MemPhi = dyn_cast<MemoryPhi>(MA)) dbgs() << "MemoryPhi has multiple incoming values\n";
        if (DefiningAccess) dbgs()<<"  DefiningAccess: ";DefiningAccess->print(dbgs());dbgs()<<"\n";
	MemoryAccess *Clobber = MSSA->getWalker()->getClobberingMemoryAccess(MA);
        //dbgs()<<"  Clobbers: ";
        //if (Clobber) Clobber->print(dbgs());dbgs()<<"\n";
      }
      dbgs()<<""<<I<<"\n";
      if (auto DL = I.getDebugLoc()) {
        for (MachineBasicBlock &MBB : MF) {
          for (MachineInstr &MI : MBB) {
            if (MI.getDebugLoc() == DL) {
              dbgs() << "    -> Correspond to MI: ";
              MI.print(dbgs());
            }
          }
        }
      } else {
        dbgs() << "    -> No debug info for this instruction\n";
      }
      dbgs()<<"\n";
    }
    dbgs()<<"---------------------\n";
  }
}

void RISCVRegisterPressure::PBDA(MachineFunction &MF, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, DenseSet<Register> &visit, DenseMap<Register, DenseSet<MachineInstr*>> &DstToInsMap, DenseMap<MachineInstr*, DenseSet<Instruction*>> &MIToIMap) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  dbgs() << "--------------------------------\nPropagating divergence state:\n";
  for (auto &[u, insSet] : DstToInsMap) {
    if (visit.count(u)) continue;
    visit.insert(u);
    dbgs() << ""<< printReg(u, TRI)<<":\n";
    bool a0 = findRegPBD(u,DivergRegs,UniformRegs,"Diverg");
    if (a0) dbgs()<<"Diverg\n";
    else {
    for (MachineInstr *Cur : insSet) {
      MachineInstr &MI = *Cur;
      dbgs()<<"  -> "<<MI;
      if (MI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK || MI.getOpcode() == RISCV::PseudoVSE32_V_M8_MASK) {
        if (MIToIMap.count(&MI)) {
          DenseSet<Instruction*> &ISet = MIToIMap[&MI];
          for (Instruction *IPtr : ISet) {
            Instruction &I = *IPtr;
            dbgs() << "    -> Correspond to I:";
	    I.print(dbgs());dbgs()<<"\n";
	    if (MemoryAccess *MA = MSSA->getMemoryAccess(&I)) {
              dbgs()<<"    -> ";MA->print(dbgs());dbgs()<<"\n";
              MemoryAccess *DefiningAccess = nullptr;
              if (auto *MemUse = dyn_cast<MemoryUse>(MA)) DefiningAccess = MemUse->getDefiningAccess();
              else if (auto *MemDef = dyn_cast<MemoryDef>(MA)) DefiningAccess = MemDef->getDefiningAccess();
              if (DefiningAccess) dbgs()<<"    -> DefiningAccess: ";DefiningAccess->print(dbgs());dbgs()<<"\n";
              if (auto *MemPhi = dyn_cast<MemoryPhi>(DefiningAccess)) {
                bool s0 = findRegPBD(u,DivergRegs,UniformRegs);
                bool a0 = findRegPBD(u,DivergRegs,UniformRegs,"Diverg");
                if (!a0) {
                  addToRegPBD(MF,u,DivergRegs,UniformRegs,StringRef("  "),"Diverg");
                  if (s0) eraseRegPBD(MF,u,DivergRegs,UniformRegs);
                }
	      }
            }
	  }
	}
      }
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        if (MO.isReg()) {
          Register Reg = MO.getReg();
          BuildDRG(MF,Reg,DivergRegs,UniformRegs,visit,DstToInsMap,MIToIMap,StringRef("  "));
          bool s0 = findRegPBD(u,DivergRegs,UniformRegs);
          bool a0 = findRegPBD(u,DivergRegs,UniformRegs,"Diverg");
          bool a1 = findRegPBD(Reg,DivergRegs,UniformRegs,"Diverg");
          if (a1 && !a0) {
            addToRegPBD(MF,u,DivergRegs,UniformRegs,StringRef("  "),"Diverg");
            if (s0) eraseRegPBD(MF,u,DivergRegs,UniformRegs);
          }
        }
      }
    }
    }
    dbgs()<<"\n";
  }
  for (auto &[u, insSet] : DstToInsMap) {
    if (!DivergRegs.count(u)) {
      UniformRegs.insert(u);
    }
  }
}

void RISCVRegisterPressure::BuildDRG(MachineFunction &MF, Register &u, DenseSet<Register> &DivergRegs, DenseSet<Register> &UniformRegs, DenseSet<Register> &visit, DenseMap<Register, DenseSet<MachineInstr*>> &DstToInsMap, DenseMap<MachineInstr*, DenseSet<Instruction*>> &MIToIMap, StringRef indent) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  if (visit.count(u)) return;
  visit.insert(u);
  dbgs() << indent.str()<< printReg(u, TRI)<<":\n";
  bool a0 = findRegPBD(u,DivergRegs,UniformRegs,"Diverg");
  if (a0) dbgs()<<indent.str()<<"Diverg\n";
  else if (DstToInsMap.count(u)) {
    DenseSet<MachineInstr*> &insSet = DstToInsMap[u];
    for (MachineInstr *Cur : insSet) {
      MachineInstr &MI = *Cur;
      dbgs()<<indent.str()<<"  -> "<<MI;
      if (MI.getOpcode() == RISCV::PseudoVLE32_V_M8_MASK || MI.getOpcode() == RISCV::PseudoVSE32_V_M8_MASK) {
        if (MIToIMap.count(&MI)) {
          DenseSet<Instruction*> &ISet = MIToIMap[&MI];
          for (Instruction *IPtr : ISet) {
            Instruction &I = *IPtr;
            dbgs()<<indent.str()<<"  -> Correspond to I:";
	    I.print(dbgs());dbgs()<<"\n";
	    if (MemoryAccess *MA = MSSA->getMemoryAccess(&I)) {
              dbgs()<<indent.str()<<"  -> ";MA->print(dbgs());dbgs()<<"\n";
              MemoryAccess *DefiningAccess = nullptr;
              if (auto *MemUse = dyn_cast<MemoryUse>(MA)) DefiningAccess = MemUse->getDefiningAccess();
              else if (auto *MemDef = dyn_cast<MemoryDef>(MA)) DefiningAccess = MemDef->getDefiningAccess();
              if (DefiningAccess) dbgs()<<indent.str()<<"  -> DefiningAccess: ";DefiningAccess->print(dbgs());dbgs()<<"\n";
              if (auto *MemPhi = dyn_cast<MemoryPhi>(DefiningAccess)) {
                bool s0 = findRegPBD(u,DivergRegs,UniformRegs);
                bool a0 = findRegPBD(u,DivergRegs,UniformRegs,"Diverg");
                if (!a0) {
                  addToRegPBD(MF,u,DivergRegs,UniformRegs,StringRef("  "),"Diverg");
                  if (s0) eraseRegPBD(MF,u,DivergRegs,UniformRegs);
                }
	      }
            }
	  }
	}
      }
      for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        if (MO.isReg()) {
          Register Reg = MO.getReg();
          BuildDRG(MF,Reg,DivergRegs,UniformRegs,visit,DstToInsMap,MIToIMap,StringRef(indent.str()+"  "));
          bool s0 = findRegPBD(u,DivergRegs,UniformRegs);
          bool a0 = findRegPBD(u,DivergRegs,UniformRegs,"Diverg");
          bool a1 = findRegPBD(Reg,DivergRegs,UniformRegs,"Diverg");
          if (a1 && !a0) {
            addToRegPBD(MF,u,DivergRegs,UniformRegs,StringRef(indent.str()+"  "),"Diverg");
	    if (s0) eraseRegPBD(MF,u,DivergRegs,UniformRegs);
          }
        }
      }
    }
  }
  /*if (u.MemAccess) {
    if (visitBuild.count(u)) return;
    visitBuild.insert(u);
    //dbgs()<<"  -> BuildDRG\n";
    MemoryAccess *MA = u.MemAccess;
    //MemoryAccess *DefiningAccess = nullptr;
    //if (auto *MemUse = dyn_cast<MemoryUse>(u)) DefiningAccess = MemUse->getDefiningAccess();
    //else if (auto *MemDef = dyn_cast<MemoryDef>(u)) DefiningAccess = MemDef->getDefiningAccess();
    //else if (auto *MemPhi = dyn_cast<MemoryPhi>(u)) dbgs() << "MemoryPhi has multiple incoming values\n";
    //if (DefiningAccess) dbgs()<<"  DefiningAccess: ";DefiningAccess->print(dbgs());dbgs()<<"\n";
    dbgs()<<indent.str()<<"-> ";
    if (auto *MemPhi = dyn_cast<MemoryPhi>(MA)) {
      dbgs()<<"φ or γ functions\n";
      BuildGate(u,visitBuild, DRGMap, DivergenceStates, indent);
    } else if (auto *MemDef = dyn_cast<MemoryDef>(MA)) {
      dbgs()<<"χ annotation\n";
      BuildChi(u,visitBuild, DRGMap, DivergenceStates, indent);
    }
  } else if (u.I) {
    if (isa<GetElementPtrInst>(u.I)) {
      dbgs()<<"  -> Points-to location\n";
      BuildPointsToOrAliased(u, visitBuild, DRGMap, DivergenceStates, indent);
    }
  }*/
  /*} else if (isa<LoadInst>(DefInst)) {
    dbgs()<<"Indirect Load\n";
    //BuildMu(u, DefInst, visitBuild, DRGMap, DivergenceStates);
  } else if (auto *CI = dyn_cast<CallInst>(DefInst)) {
    dbgs()<<"Updated by function\n";
    //BuildCallee(u, DefInst, visitBuild, DRGMap, DivergenceStates);
  } else {
    dbgs()<<"Handle general assign\n";
    //handleGeneralAssignment(u, DefInst, visitBuild, DRGMap, DivergenceStates);
  }*/
}

void RISCVRegisterPressure::BuildPointsToOrAliased(DRGNode &u,DenseSet<DRGNode> &visitBuild, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates, StringRef indent) {
}

void RISCVRegisterPressure::BuildGate(DRGNode &u,DenseSet<DRGNode> &visitBuild, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates, StringRef indent) {
  MemoryAccess *MA = u.MemAccess;
  if (MemoryPhi *MemPhi = llvm::dyn_cast<MemoryPhi>(MA)) {
    unsigned ID = MemPhi->getID();
    dbgs()<<indent.str()<<"-> ";MA->print(dbgs());dbgs()<<"\n";
    dbgs()<<indent.str()<<"-> BuildGate("<<ID<<")\n";
    if (MemPhi->getGamma()) {
      BasicBlock *BB = MemPhi->getBlock();
      auto Preds = predecessors(BB);
      if (std::distance(Preds.begin(), Preds.end()) == 2) {
        BasicBlock *Pred1 = *Preds.begin();
        BasicBlock *Pred2 = *(++Preds.begin());
        /*auto PredPred = predecessors(Pred1);
        for (BasicBlock *BB : PredPred) {
          dbgs() << "  Pred1: "<<BB->getName() << "\n";
        }
        auto PredPred2 = predecessors(Pred2);
        for (BasicBlock *BB : PredPred2) {
          dbgs() << "  Pred2: " <<BB->getName() << "\n";
        }*/
	BasicBlock *CommonPred = nullptr;
        SetVector<BasicBlock*> Worklist;
        DenseSet<BasicBlock*> Pred1Ancestors;
        DenseSet<BasicBlock*> Pred2Ancestors;
        Worklist.insert(Pred1);
        while (!Worklist.empty()) {
          BasicBlock *Cur = Worklist.pop_back_val();
          Pred1Ancestors.insert(Cur);
          for (BasicBlock *Pred : predecessors(Cur)) {
            if (!Pred1Ancestors.count(Pred)) {
              Worklist.insert(Pred);
            }
          }
        }
        Worklist.clear();
        Worklist.insert(Pred2);
        while (!Worklist.empty()) {
          BasicBlock *Cur = Worklist.pop_back_val();
          Pred2Ancestors.insert(Cur);

          if (Pred1Ancestors.count(Cur)) {
            Instruction *TI = Cur->getTerminator();
            if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
              if (BI->isConditional()) {
                bool LeadsToPred1 = false;
                bool LeadsToPred2 = false;
                for (BasicBlock *Succ : BI->successors()) {
                  if (Succ == Pred1 || Pred1Ancestors.count(Succ)) {
                    LeadsToPred1 = true;
                  }
                  if (Succ == Pred2 || Pred2Ancestors.count(Succ)) {
                    LeadsToPred2 = true;
                  }
                }
                if (LeadsToPred1 && LeadsToPred2) {
                  CommonPred = Cur;
                  Value *Condition = BI->getCondition();
		  MemPhi->setCondition(Condition);
                  dbgs()<<indent.str()<<"  -> Found condition in " << CommonPred->getName()
                     << ": " << *Condition << "\n";
                  break;
                }
              }
            }
          }
          for (BasicBlock *Pred : predecessors(Cur)) {
            if (!Pred2Ancestors.count(Pred)) {
              Worklist.insert(Pred);
            }
          }
        }

        if (CommonPred) {
          //dbgs() <<"  -> "<<*MemPhi->getCondition()<<"\n";
	  Value *Condition = MemPhi->getCondition();
	  Instruction *CondI = cast<Instruction>(Condition);
          Value *Op = CondI->getOperand(0);
	  if (Argument *Arg = dyn_cast<Argument>(Op)) {
            dbgs()<<indent.str()<<"  -> Function argument: " << Arg->getName() << "\n";
	    //dbgs() << "Arg index: " << Arg->getArgNo() << "\n";
          }
        }
      }
    }
    for (unsigned i = 0, e = MemPhi->getNumIncomingValues(); i < e; ++i) {
      MemoryAccess *IncomingVal = MemPhi->getIncomingValue(i);
      //if (IncomingValue) IncomingValue->print(dbgs());
      DRGNode GatingNode = getDRGNode(IncomingVal);
      
      DenseSet<DRGNode> &insSet = DRGMap[u];
      insSet.insert(GatingNode);
      DivergenceStates[GatingNode] = NON_DIVERGENT;
      //BuildDRG(GatingNode,visitBuild, DRGMap, DivergenceStates, StringRef(indent.str()+"  "));
    }
  }
}

void RISCVRegisterPressure::BuildChi(DRGNode &u,DenseSet<DRGNode> &visitBuild, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates, StringRef indent) {
  MemoryAccess *MA = u.MemAccess;
  if (auto *MD = dyn_cast<MemoryDef>(MA)) {
    unsigned ID = MD->getID();
    dbgs()<<indent.str()<<"-> ";MA->print(dbgs());dbgs()<<"\n";
    dbgs()<<indent.str()<<"-> BuildChi("<<ID<<")\n";
    MemoryAccess *Def = MD->getDefiningAccess();
    DRGNode Node = getDRGNode(Def);

    DenseSet<DRGNode> &insSet = DRGMap[u];
    insSet.insert(Node);
    DivergenceStates[Node] = NON_DIVERGENT;
    //BuildDRG(Node, visitBuild, DRGMap, DivergenceStates, StringRef(indent.str()+"  "));

    Instruction *I = MD->getMemoryInst();
    if (I) {
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        dbgs()<<indent.str()<<"-> ";MA->print(dbgs());dbgs()<<"\n";
        dbgs()<<indent.str()<<"-> BuildChi("<<ID<<")\n";
        dbgs()<<indent.str()<<"-> Indirect Store: "<<*SI<<"\n";
        Value *destination = SI->getPointerOperand();
	dbgs()<<indent.str()<<"-> Destination address: ";destination->print(dbgs());dbgs() << "\n";
        dbgs()<<indent.str()<<"-> Destination name: %" << destination->getName() << "\n";
        /*DRGNode Node = getDRGNode(Def);

        DenseSet<DRGNode> &insSet = DRGMap[u];
        insSet.insert(Node);
        DivergenceStates[Node] = NON_DIVERGENT;
        BuildDRG(Node, visitBuild, DRGMap, DivergenceStates, StringRef(indent.str()+"  "));*/
      }
    } else {
      //dbgs()<<indent.str()<<"-> ";MA->print(dbgs());dbgs()<<"\n";
      //dbgs()<<indent.str()<<"-> BuildChi("<<ID<<")\n";
      //dbgs()<<indent.str()<<"-> No underlying instr\n";
    }
  }
}

bool RISCVRegisterPressure::DivergenceStatePropagation(DRGNode &u, DenseSet<DRGNode> &visitDRG, DenseMap<DRGNode, DenseSet<DRGNode>> &DRGMap, DenseMap<DRGNode, DivergenceState> &DivergenceStates) {
  /*bool state = (DRegs.count(u) ? true : false);
  if (visitDRG.count(u)) return state;
  if (!DRGMap.count(u)) return state;
  for(auto &v : DRGMap[u]) {
    visitDRG.insert(v);
    state |= DivergenceStatePropagation(v, visitDRG, DRGMap, DivergenceStates);
  }
  return state;*/
  return false;
}

void RISCVRegisterPressure::analyzeMachineLoop(MachineFunction &MF) {
  DenseSet<Register> ScalarRegs;
  DenseSet<Register> AffineRegs;
  DenseSet<Register> VectorRegs;
  DenseSet<Register> ScalarArrs;
  DenseSet<Register> AffineArrs;
  DenseSet<Register> VectorArrs;

  DenseSet<Register> DivergRegs;
  DenseSet<Register> UniformRegs;
  //MachineBasicBlock *Preheader = ML->getLoopPreheader();
  //MachineBasicBlock *Header = ML->getHeader();
  //MachineBasicBlock *Cleanup = ML->getExitBlock();
  int LB = 0, UB = 0, INC = 1;
  DenseSet<Register> VLs;
  DenseSet<Register> ARRs;
  DenseSet<Register> AIs;
  DenseSet<Register> Is;
  DenseMap<Register, DenseSet<MachineInstr*>> RegToInsMap;
  DenseMap<Register, DenseSet<MachineInstr*>> DstToInsMap;
  DenseMap<Register, DenseSet<MachineInstr*>> AIToInsMap;
  DenseMap<MachineInstr*, DenseSet<Instruction*>> MIToIMap;
  SetVector<MachineInstr*> WorkList;
  SetVector<MachineInstr*> InsList;
  DenseSet<Register> visit;
  DenseMap<DRGNode, DenseSet<DRGNode>> DRGMap;
  //collectIns(MF, ScalarRegs, AffineRegs, VectorRegs, RegToInsMap);
  //collectDstIns(MF, ScalarRegs, AffineRegs, VectorRegs, DstToInsMap);
  //collectMIToI(MF,MIToIMap);
  //findAffine(MF, ScalarRegs, AffineRegs, VectorRegs, VLs, ARRs, AIs);
  //findINC(MF,DivergRegs,UniformRegs,VLs,ARRs,AIs,Is);
  /*dbgs() << "--------------------------------\nCollecting AI instruction:\n";
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.getName() == "for.cond.cleanup") continue;
    StringRef BType = (MBB.getName().ends_with(".body") ? "Body" : "Entry");
    collectAIIns(&MBB,AIToInsMap,ARRs,AIs,BType);
  }
  dbgs() << "--------------------------------\nPasting Array type:\n";
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.getName() == "for.cond.cleanup") continue;
    StringRef BType = (MBB.getName().ends_with(".body") ? "Body" : "Entry");
    pasteArrType(MF,&MBB,ScalarRegs,AffineRegs,VectorRegs,ScalarArrs,AffineArrs,VectorArrs,ARRs,AIs,AIToInsMap,BType);
  }
  dbgs() << "--------------------------------\nFinding Scalar:\n";
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.getName() == "for.cond.cleanup") continue;
    StringRef BType = (MBB.getName().ends_with(".body") ? "Body" : "Entry");
    findScalar(&MBB, ScalarRegs, AffineRegs, VectorRegs, BType);
  }
  dbgs() << "--------------------------------\nFinding Vector:\n";
  findVector(MF,Cleanup,ScalarRegs,AffineRegs,VectorRegs,ScalarArrs,AffineArrs,VectorArrs,RegToInsMap,WorkList,AIs,"Body");
  dbgs()<<"--------------------------------\nDetecting Array type:\n";
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.getName() == "for.cond.cleanup") continue;
    StringRef BType = (MBB.getName().ends_with(".body") ? "Body" : "Entry");
    detectArrType(MF,&MBB,ScalarRegs,AffineRegs,VectorRegs,ScalarArrs,AffineArrs,VectorArrs,ARRs,AIs,AIToInsMap,InsList,BType);
  }
  recordArrType(MF,Cleanup,ScalarArrs,AffineArrs,VectorArrs, "Body");*/
  //DetectAndPrintDivergence(MF);
  //detectRegType(MF, ScalarRegs, AffineRegs, VectorRegs);
  //PBDA(MF,DivergRegs,UniformRegs,visit,DstToInsMap,MIToIMap);
  //analyzePrint(MF, ScalarRegs, AffineRegs, VectorRegs, DivergRegs, UniformRegs);
  //analyzePrintDiv(MF, ScalarRegs, AffineRegs, VectorRegs, DivergRegs, UniformRegs);
}

bool RISCVRegisterPressure::runOnMachineFunction(MachineFunction &MF) {
  if (!enableRegisterPressureOpt())
    return false;
  dbgs() << "RP:Entering RegisterPressure for " << MF.getName() << "\n";
  //MUI = &getAnalysis<MachineUniformityAnalysisPass>().getUniformityInfo();
  //MUI = getAnalysis<UniformityBeforePhiWrapper>().getUniformityInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MSSA = &getAnalysis<MemorySSAWrapperPass>().getMSSA();

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  /*unsigned numPSets = TRI->getNumRegPressureSets();
  dbgs() << "numPSets: "<<numPSets<<"\n";
  for (const TargetRegisterClass *RC : TRI->regclasses()) {
    const int *PSets = TRI->getRegClassPressureSets(RC);
    std::string RCName = TRI->getRegClassName(RC);
    for (unsigned i = 0; PSets[i] != -1; ++i) {
      if (PSets[i] == 14)
        dbgs()<<RCName<<" maps to "<<PSets[i]<<"\n";
    }
  }*/

  bool change = false;
  if (MLI->empty()) {
    return false;
  }
  for (auto  ML : *MLI) {		// for all loop in function
    auto LoopID = getLoopID(ML);  
    auto IsVectorized = getOptionalBoolLoopAttribute(LoopID, "llvm.loop.isvectorized");

    // dbgs() << "getNumRegPressureSets" << TRI->getNumRegPressureSets() << '\n';
    // for (unsigned i = 0, e = TRI->getNumRegPressureSets(); i < e; ++i) {
    //   dbgs() << "i = " << i  << ":" << TRI->getRegPressureSetName(i)  << '\n';
    // }
    // static const char *PressureNameTable[] = {
    // "GPRC_and_PGPR", 0
    // "GPRX0",
    // "SP",
    // "VCSR",
    // "FPR32C",
    // "GPRC",
    // "SR07",
    // "VMV0",
    // "PGPR",
    // "GPRC_with_SR07",
    // "GPRTC",
    // "PGPR_with_GPRC",
    // "VRM8NoV0",
    // "FPR16",
    // "VM",  14
    // "GPR", 15
    // };
    

    RegisterPressureHotSpot P;
    //analyzeMachineLoop(MF, ML);
    bool changed = true;
    
    while (0 && changed && hasHotSpotInLoop(MF, ML, P)) {
      dbgs()<<"Return from Hot Spot\n";
      std::vector<Register> LiveVRegs;
      SlotIndex BeginIndex = LIS->getInstructionIndex(*P.BeginPos), EndIndex = LIS->getInstructionIndex(*P.EndPos);
      dbgs() << "RP: FindHotSpot " << "\n";
      dbgs() << "RP: P.BeginPos " << *P.BeginPos << "SlotIndex: " << BeginIndex << "\n";
      dbgs() << "RP: P.EndPos " << *P.EndPos << "SlotIndex: " << EndIndex << "\n";

      auto &MRI = MF.getRegInfo();
      //LIS->print(dbgs(), MF.getFunction().getParent());
      dbgs() << "RP: getNumVirtRegs " << MRI.getNumVirtRegs() << "\n";
      for (unsigned i = 0; i < MRI.getNumVirtRegs(); ++i) {
        Register Reg = Register::index2VirtReg(i);
        auto *RC = MRI.getRegClass(Reg);
        auto *Pset = TRI->getRegClassPressureSets(RC);
        dbgs() << "Register: " << printReg(Reg, TRI, 0, &MRI) << " RegClass: " << TRI->getRegClassName(RC) << " Pset: " << *Pset << " PSName: " << TRI->getRegPressureSetName(*Pset) << "\n";
        if (*Pset == 14) {
          LiveInterval &LI = LIS->getInterval(Reg);
          std::string RCName = TRI->getRegClassName(RC);
          for (const LiveRange::Segment &Seg : LI) {
            dbgs() << "    [" << Seg.start << "-" << Seg.end << ")\n";
          }
          if (LI.overlaps(BeginIndex, EndIndex)) {
            LiveVRegs.push_back(Reg);
            dbgs() << "!Push to LiveVRegs: " << LI << "\n";
          }
        }
      }
      // for (auto Reg: LiveVRegs) {
      //   LLVM_DEBUG(dbgs() << "Register: " << printReg(Reg, TRI, 0, &MRI) << " RegClass: " << TRI->getRegClassName(MRI.getRegClass(Reg)) << "\n");
      // }
      // for (unsigned i = 0; i < TRI->getNumRegClasses(); i++) {
      //   LLVM_DEBUG(dbgs() << i << " " << TRI->getRegClassName(TRI->getRegClass(i)) << "\n");
      // }
      auto MII = P.BeginPos.getInstrIterator();
      auto MII_end = P.EndPos.getInstrIterator();
      // pair: active, score
      std::vector<std::pair<int, int>> scores;
      for (auto Reg: LiveVRegs) {
        dbgs() << "RP: " << printReg(Reg, TRI, 0, &MRI) << "\n";
        std::pair<int, int> score;
        score.first = 1;
        score.second = 0;
        scores.push_back(score);
      }
      for (; MII != MII_end; MII++) {
        for (size_t i = 0; i < LiveVRegs.size(); i++) {
          auto RW = (*MII).readsWritesVirtualRegister(LiveVRegs[i]);
          if (RW.first || RW.second) {
            scores[i].first = 0;
          } else if (scores[i].first) {
            scores[i].second += 1;
          }
        }
      }

      // get highest score Reg
      int maxID, maxScore = -1;
      for (size_t i = 0; i < LiveVRegs.size(); i++) {
        if (scores[i].second > maxScore) {
          maxID = i;
          maxScore = scores[i].second;
        }
        dbgs() << printReg(LiveVRegs[i], TRI, 0, &MRI) << " scores: " << scores[i].second << "\n";
      }

      changed = false;

      changed |= trySinkLoad(MF, P, LiveVRegs, LiveVRegs[maxID]);
      change |= changed;
      break;
    }
    
  }
  analyzeMachineLoop(MF);
  dbgs() << "Exist RegisterPressure for " << MF.getName() << "\n";
  dbgs()<<"\n\n";
  return change;
}

FunctionPass *llvm::createRISCVRegisterPressurePass() {
  return new RISCVRegisterPressure();
}


