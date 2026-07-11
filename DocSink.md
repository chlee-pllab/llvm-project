# Group Sink Optimization

`ProcessInSameBlock` has a targeted group-sinking path for selected masked
32-bit vector loads.  It recognizes selected loads and moves the load together with the setup needed to execute it immediately before its vector arithmetic consumer.

The group is only moved within the same machine basic block.

## Intended pattern

A masked `PseudoVLE32_V_M8_MASK` load is selected when its memory operand has an offset of at least half a block:

```text
memory offset >= (BlockSize / 2) * 4
```

The full expected idiom is:

```text
%47 = PseudoVMV_V_I_M8 undef %47, 0                 # VMV zero setup
$v0 = COPY %21                                      # COPY mask to $v0
%47 = PseudoVLE32_V_M8_MASK %47, base+{256|384}, $v0, %vl
%60  = PseudoVFADD_VV_M8_E32 ..., %31, %47, ..., %vl
```

`PseudoVFMAX_VV_M8_E32` is also accepted as the consumer.

## What `ProcessInSameBlock` actually checks

For a masked load, the pass collects these instructions in this order:

1. The `PseudoVLE32_V_M8_MASK` load.
2. The nearest preceding definition of the first physical register used by the
   load.  For this MIR that register is `$v0`, and the definition must be a `COPY`.
3. A different definition of the load's tied destination whose opcode name
   contains `PseudoVMV_V_I_M8`; this is the zero/undisturbed-vector setup.

# Single-Instruction Sink Optimization

`ProcessInSameBlock` contains a narrow single-instruction sink optimization.  It moves an eligible zero-initializing vector move immediately before the masked vector load that reuses its tied destination.

## Intended pattern

It starts at an `PseudoVMV_V_I` instruction.  Before sinking, unrelated instructions may separate the initialization from its load:

```text
%55 = PseudoVMV_V_I_M8 undef %55, 0
... unrelated instructions ...
%55 = PseudoVLE32_V_M8_MASK %55, %54, $v0, %vl, 5, 1
```

The pass sinks the `PseudoVMV_V_I` directly before the load:

```text
... unrelated instructions ...
%55 = PseudoVMV_V_I_M8 undef %55, 0
%55 = PseudoVLE32_V_M8_MASK %55, %54, $v0, %vl, 5, 1
```

## What `ProcessInSameBlock` actually checks

- It starts at the `PseudoVMV_V_I`.
- It looks specifically for a following `PseudoVLE32_V_M8_MASK` with the same tied destination.

# Expand-Pseudos COPY-to-VMV Optimization

This phase runs before the sink diagnostics.  `LowerCopy` removes a redundant virtual vector `COPY` by recreating its VMV producer directly in the COPY destination.

## Intended pattern

The pass looks for a VMV that defines the COPY source:

```text
src:VRM8 = PseudoVMV_V_I_M8 undef src, 0, %vl, ...
dst:VRM8 = COPY src
```

The COPY is erased and a new VMV is inserted at the former COPY location:

```text
dst = PseudoVMV_V_I_M8 undef dst, 0
```

A vector-from-scalar `PseudoVMV_V_X_M8` move is also accepted.

## Observed instances and connection to group sinking

The example records seven immediate-form rewrites of
`%dst:vrm8nov0 = COPY %55:vrm8nov0`: destinations `%27`, `%31`, `%35`, `%39`,
`%43`, `%47`, and `%51`.  These replacements create VMV definitions at the
per-load destinations.  The later group-sink path can then find the
`PseudoVMV_V_I_M8` setup for `%31`, `%35`, `%47`, and `%51`, bundle it with the
`$v0` mask COPY and masked VLE32, and sink that group to its VFADD consumer.

## Observed instances and connection to single-instruction sinking

The `LowerCopy` phase also supplies one of the VMVs later handled by the single-instruction sink path.  It first lowers:

```text
%27 = COPY %55  ->  %27 = PseudoVMV_V_I_M8 undef %27, 0, %9, 5, 0
```

`ProcessInSameBlock` subsequently recognizes `%27` as an eligible VMV and finds its later tied-destination masked load.  It then moves the materialized VMV immediately before this load. The flow is therefore:

```text
original %55 VMV -> COPY to a destination -> LowerCopy creates VMV -> single sink places before VLE32
```

This shows that `LowerCopy` creates independent zero/undisturbed setups for copied vector destinations.
