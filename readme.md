# SpecDVFS: Acceptance-Rate-Adaptive GPU Frequency Scaling

This repository contains the codebase for the **SpecDVFS** research project. We explore the energy inefficiency of Speculative Decoding (SD) and propose a lightweight runtime controller to dynamically scale GPU frequencies (DVFS) based on SD phase transitions and acceptance rates.

## Project Goal
Standard LLM inference optimizations are designed for two-phase inference (prefill/decode). Speculative Decoding introduces a three-phase structure (Draft, Verify, Rollback). Our research demonstrates that running a fixed GPU clock speed throughout these phases wastes energy. **SpecDVFS** recovers energy efficiency by scaling the GPU clock in real-time according to the arithmetic intensity of the current phase.

## Repository Structure
- `controller/`: NVML-based GPU frequency control logic.
- `data/`: Dataset download and sampling scripts.
- `evaluation/`: Scripts to calculate Energy Saving Factors ($\gamma_e^{GPU}$) and EDP.
- `experiments/`: Replication scripts and experiment runners.
- `profiling/`: Instrumentation for phase-aware timing and GPU monitoring.

## Getting Started

### 1. Environment Setup
```bash
conda create -n specdvfs python=3.10 -y
conda activate specdvfs
pip install -r requirements.txt
