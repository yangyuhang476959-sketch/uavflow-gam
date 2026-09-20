#!/usr/bin/env bash

# Single source of truth for the compact ablation order. Scheduler arrays use
# the zero-based position as their cell index.
COMPACT_RUN_IDS=(
  B0 S1COS P1 L1 D1 D2 D1LOG W3 W5 W10 H0 HB F3 F7 F10 M1
  C1_PL C2_D2HB C3_W3HB C4_F3W3 C5_F10HB
)

