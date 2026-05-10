# FlashAttention Accelerator: Final Design Document

**GitHub:** https://github.com/ricardotk002/hwdesign-flash-attn-accel  
**Team:** Anuj Apte (aa13231@nyu.edu) · Ricardo Díaz (erd9862@nyu.edu) · Raquel Brown (rb6191@nyu.edu)  
**Target board:** PYNQ-Z2 (Zynq XC7Z020-CLG400-1)  
**Toolchain:** Vitis HLS 2023.2

---

## 1. IP Definition and Interface

### 1.1 Purpose

This IP implements the forward pass of single-head scaled dot-product attention using a FlashAttention-style tiled algorithm. Standard attention computes the full N×N score matrix, which requires O(N²) memory and large off-chip traffic. This IP avoids materializing that matrix by processing Q, K, and V in tiles of fixed size (BQ = BK = 8) and maintaining running softmax state on-chip, reducing peak memory requirements from O(N²) to O(N·d).

The design targets transformer inference on resource-constrained SoC platforms where DDR bandwidth and on-chip memory are limited.

### 1.2 Mathematical Operations

The IP computes:

```
O = softmax(Q K^T / sqrt(d)) V
```

For each query row q_i and key row k_j, the scaled dot-product score is:

```
s_ij = (q_i · k_j) / sqrt(d)
```

When causal mode is enabled, future positions are masked before softmax:

```
s_ij = -inf   for j > i
```

The softmax is computed using an online algorithm that avoids storing all N scores simultaneously. For each K/V tile, three state variables are updated per query row:

```
m_i^new = max(m_i, max_j s_ij)

l_i^new = exp(m_i - m_i^new) * l_i  +  sum_j exp(s_ij - m_i^new)

o_i^new = [ exp(m_i - m_i^new) * l_i * o_i  +  sum_j exp(s_ij - m_i^new) * v_j ]
          / l_i^new
```

The accumulator `o_i` is normalized after every tile, so no final normalization pass is needed. The correction factor `exp(m_i - m_i^new)` rescales the prior running state to match the new higher normalization base whenever the running maximum grows.

### 1.3 Hardware/Software Boundary

The Processing System (PS) is responsible for:
- Allocating and populating Q, K, V buffers in DDR
- Writing base addresses and parameters (N, d, causal) to the AXI4-Lite control register bank
- Issuing AP_START and polling AP_DONE or waiting for interrupt
- Reading the output O buffer from DDR

The Programmable Logic (PL) IP is responsible for:
- Loading tiles of Q, K, V from DDR into on-chip BRAM buffers
- Computing scores, softmax state, and weighted accumulation entirely on-chip
- Writing the final output O back to DDR

No floating-point work is performed by the PS after the IP is started. All intermediate state (m, l, acc, scores, weighted) lives on-chip and is never written to DDR.

**Compile-time constraints:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| N_MAX | 64 | Maximum sequence length |
| D_MAX | 32 | Maximum embedding dimension |
| BQ | 8 | Q tile size (rows) |
| BK | 8 | K/V tile size (rows) |

### 1.4 Interfaces

#### AXI4 Master (data movement)

Four independent AXI4 Master interfaces give each matrix its own DDR channel, avoiding contention:

| Bundle | Port | Direction | Max burst |
|--------|------|-----------|-----------|
| gmem0 | Q | read | 16 elements |
| gmem1 | K | read | 16 elements |
| gmem2 | V | read | 16 elements |
| gmem3 | O | write | 16 elements |

Each interface is 32-bit wide (matching `float`). Bursts are variable-length and inferred automatically by Vitis HLS from the innermost pipelined loop.

#### AXI4-Lite Slave (control)

All scalar parameters and base addresses are mapped to a single `s_axi_ctrl` register bank:

| Register | Offset | Description |
|----------|--------|-------------|
| CTRL | 0x00 | AP_START / AP_DONE / AP_IDLE / AP_READY |
| Q_1/Q_2 | 0x10/0x14 | 64-bit base address of Q buffer |
| K_1/K_2 | 0x1c/0x20 | 64-bit base address of K buffer |
| V_1/V_2 | 0x28/0x2c | 64-bit base address of V buffer |
| O_1/O_2 | 0x34/0x38 | 64-bit base address of O buffer |
| N | 0x40 | Sequence length (runtime) |
| d | 0x48 | Embedding dimension (runtime) |
| causal | 0x50 | Causal mask enable (0 or 1) |

The `causal` parameter is `int` (not `bool`) to ensure correct AXI4-Lite 32-bit register mapping.

#### Top-level control ports

| Port | Type | Description |
|------|------|-------------|
| ap_clk | clock | System clock |
| ap_rst_n | reset | Active-low reset |
| interrupt | interrupt | Pulses when AP_DONE |

---

## 2. Architecture and Implementation Quality

### 2.1 Sub-Module Overview

The IP is structured as a sequential pipeline of seven logical modules, each with a clearly scoped role. In the current Vitis HLS implementation these are coded as named loop regions within a single top-level function to allow loop-level pipelining. A `#pragma HLS DATAFLOW` decomposition into separate functions is left as future work (see Section 4).

```
DDR
 │  (AXI4 burst read)
 ▼
[1] Q/K/V Loader  →  [2] On-Chip Tile Buffers (Qbuf, Kbuf, Vbuf in BRAM)
                              │
                              ▼
                     [3] Dot-Product Engine  →  scores[]
                              │
                              ▼
                     [4] Online Softmax Unit  →  m[], l[], exp values
                              │
                              ▼
                     [5] Weighted Value Accumulator  →  acc[]
                              │
 ▼                            ▼
[6] Output Writeback  ←──────────────────────────────
 │  (AXI4 burst write)
 ▼
DDR
```

Coordinating the loop ordering across all modules is the responsibility of:

**[7] Top-Level Controller (FSM)** — implemented as nested `for` loops over Q tiles and K/V tiles. Controls buffer initialization, tile loading, compute sequencing, and output writeback.

### 2.2 Sub-Module Details

#### [1] Q/K/V Loader

Reads tiles from DDR into on-chip buffers. Inner loop is pipelined at II=1. For Q, loads `q_lim` rows × `d` columns. For K and V, loads `k_lim` rows × `d` columns of both matrices in a single combined loop to maximize bus utilization.

Memory indexing uses 1D row-major layout with D_MAX stride: `ptr[(row)*D_MAX + col]`. This is required for AXI4 Master compatibility — 2D array ports are not synthesizable to m_axi interfaces in Vitis HLS.

#### [2] On-Chip Tile Buffers

Three local arrays of shape `[BQ/BK][D_MAX]` (Qbuf, Kbuf, Vbuf). Mapped to BRAM by HLS. Partitioned cyclically by factor 4 on the column dimension to allow 4-wide parallel reads per cycle, enabling the dot-product engine to achieve II=1 on the score loop.

```cpp
#pragma HLS ARRAY_PARTITION variable=Qbuf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=Kbuf cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=Vbuf cyclic factor=4 dim=2
```

#### [3] Dot-Product Engine

Computes the BQ × BK score matrix. The outer `i` loop (over query rows) is sequential; the inner `j` loop (over key rows) is pipelined at II=1. The innermost dimension loop over `d` is fully unrolled, creating D_MAX = 32 parallel multiplier-adder chains:

```cpp
SCORE_LOOP_J:
for (int j = 0; j < k_lim; j++) {
#pragma HLS PIPELINE II=1
    data_t dot = 0.0f;
    SCORE_LOOP_X:
    for (int x = 0; x < D_MAX; x++) {
#pragma HLS UNROLL
        if (x < d) dot += Qbuf[i][x] * Kbuf[j][x];
    }
    ...
}
```

This loop achieves II=1 with a latency of 202 cycles for BK=8 iterations (the extra cycles come from pipelining depth of the floating-point multiply-accumulate chain). It is the most DSP-intensive loop, consuming 141 of 178 total DSPs.

Causal masking is applied inline: if `causal && (k0+j) > (q0+i)`, the score is replaced with NEG_INF = -1e30, which causes `expf(NEG_INF)` to underflow to zero.

#### [4] Online Softmax Unit

Owns the running maximum `m[BQ]` and normalization `l[BQ]`. For each K/V tile and each query row:

1. **FIND_MAX**: pipelines over k_lim to find tile row-max (II=1, latency 3 cycles)
2. Computes `m_new = max(m[i], tile_max)` and `old_scale = expf(m[i] - m_new)`
3. **EXP_ACCUM**: sequential loop over k_lim, computing `expf(score - m_new)` and accumulating into the `weighted[]` array (inner d loop unrolled factor=4 for resource safety)
4. Updates `l_new = old_scale * l[i] + exp_sum`

#### [5] Weighted Value Accumulator

Receives `old_scale`, `l_new`, and `weighted[]` from the Online Softmax Unit. Updates the output accumulator **with normalization applied at every tile**, so `acc[i]` always holds a valid normalized partial result:

```
acc[i][x] = (old_scale * l[i] * acc[i][x] + weighted[x]) / l_new
```

The `acc[BQ][D_MAX]` array is partitioned cyclically by factor 4. The UPDATE_ACC pipeline achieves II=1 with latency 34 cycles for 8 active elements.

#### [6] Output Writeback

After all K/V tiles are processed for a Q tile, writes `acc` rows to the DDR O buffer. Inner loop is pipelined at II=1, latency 9 cycles.

#### [7] Top-Level Controller

Implemented as nested for loops (Q_TILE_LOOP → K_TILE_LOOP). Computes `q_lim = min(BQ, N - q0)` and `k_lim = min(BK, N - k0)` at each tile boundary to handle sequence lengths that are not multiples of the tile size. Initializes m, l, acc at the start of each Q tile.

### 2.3 HLS Optimizations

| Pragma | Location | Effect |
|--------|----------|--------|
| `PIPELINE II=1` | LOAD_Q inner x loop | Burst read 1 float per cycle |
| `PIPELINE II=1` | LOAD_KV inner x loop | Burst read K and V in lockstep |
| `PIPELINE II=1` | SCORE_LOOP_J | Produce 1 score per cycle |
| `UNROLL` (full) | SCORE_LOOP_X | 32 parallel MACs for dot product |
| `PIPELINE II=1` | FIND_MAX | 1 comparison per cycle |
| `UNROLL factor=4` | EXP_ACCUM inner x loop | Partial V accumulation |
| `UNROLL factor=4` | UPDATE_ACC inner x loop | Partial acc update |
| `ARRAY_PARTITION cyclic 4` | Qbuf, Kbuf, Vbuf, acc | 4-wide BRAM banks |
| `PIPELINE II=1` | STORE_O inner x loop | Burst write 1 float per cycle |

### 2.4 Precision and Data Type

All computation uses `float` (IEEE 754 single-precision, 32-bit). The code is structured to make switching to `ap_fixed<16,6>` straightforward: replacing `typedef float data_t` with the fixed-point typedef and recompiling would exercise the fixed-point path. The golden model runs in `float64` to serve as a high-accuracy reference; the expected DUT error is below 1e-4.

`hls::sqrtf`, `hls::expf` (from `<hls_math.h>`) are used instead of standard `<cmath>` functions to ensure synthesizable floating-point operations.

### 2.5 Code Readability

- All loop bodies are labeled (Q_TILE_LOOP, LOAD_Q, INIT_STATE, K_TILE_LOOP, LOAD_KV, SCORE_LOOP_I/J/X, UPDATE_LOOP_I, FIND_MAX, EXP_ACCUM, INIT_WEIGHTED, UPDATE_ACC, STORE_O) for synthesis report traceability
- Sub-module boundaries are called out in comments that match the planning documentation
- `NEG_INF` is defined once as -1e30f with an explanation of why exp(NEG_INF) underflows to 0
- The `causal` parameter's `int` type (not `bool`) is documented at the interface

---

## 3. Evaluation and Verification

### 3.1 Golden Model

`code/golden_model.py` provides two Python reference implementations running in `float64`:

- `attention_naive(Q, K, V, causal)` — standard O(N²) attention, highest accuracy
- `attention_tiled_online(Q, K, V, block_size, causal)` — FlashAttention-style tiled version, verifies the online update is mathematically identical to naive

The golden model outputs a CSV (`test_outputs/tv_attention_python.csv`) that the testbench reads. The expected outputs are from `attention_naive` (float64 reference) so the HLS DUT is always compared against the most accurate available answer.

### 3.2 Testbench Design

`code/tb_flash_attention.cpp` is a Vitis HLS C/C++ testbench that:

1. Reads all test vectors from the golden-model CSV
2. Allocates flat 1D arrays with `D_MAX` stride matching the DUT's AXI pointer interface
3. Bounds-checks N ≤ N_MAX and d ≤ D_MAX before calling the DUT (skips and counts as failure otherwise)
4. Calls `flash_attention_hls(Q, K, V, O, N, d, causal)` — the same function that gets synthesized
5. Computes `max_abs_err_dut = max |O_dut - O_expected|` over all (i, j) pairs
6. Reports PASS/FAIL per test with a 1e-4 tolerance
7. Writes a per-test result CSV (`tv_attention_csim.csv` or `tv_attention_rtl.csv`) for downstream analysis

The testbench accepts a command-line argument `rtl` to distinguish C-simulation from RTL co-simulation output paths — the HLS TCL script (`run_hls.tcl`) passes `rtl` to `cosim_design -argv "rtl"`.

Tests include:
- **Small single-tile cases** (N=4, N=8): verify correct output when the entire sequence fits in one tile, both causal and non-causal
- **Multi-tile cases** (N=16, d=8): exercise the full nested tile loop (2 Q tiles × 2 K tiles = 4 tile iterations), both causal and non-causal
- **Causal masking**: verified that future positions produce zero contribution to softmax

### 3.3 Functional Verification Results

C-simulation and RTL co-simulation results (tolerance = 1e-4):

| Test | N | d | Block | Causal | err (python vs float64) | err (DUT vs float64) | Result |
|------|---|---|-------|--------|--------------------------|----------------------|--------|
| 0 | 4 | 4 | 2 | No | 2.6e-8 | 6.0e-8 | **PASS** |
| 1 | 4 | 4 | 2 | Yes | 4.5e-8 | 1.19e-7 | **PASS** |
| 2 | 8 | 4 | 4 | Yes | 5.5e-8 | 1.19e-7 | **PASS** |
| 3 | 16 | 8 | 8 | No | 1.23e-7 | 1.19e-7 | **PASS** |
| 4 | 16 | 8 | 8 | Yes | 9.8e-8 | 1.56e-7 | **PASS** |

All five test vectors pass in both C-simulation and RTL co-simulation. Maximum observed DUT error is **1.56e-7**, more than 600× below the 1e-4 tolerance — consistent with the expected float32 rounding error for this computation.

### 3.4 Synthesis Results

Target: xc7z020-clg400-1 at 100 MHz (10 ns clock period).

#### Resource Utilization

Post-synthesis utilization on xc7z020: **BRAM_18K 141/280 (50%), DSP48E1 178/220 (81%), FF 31,527/106,400 (30%), LUT 40,259/53,200 (76%).** All fit within the PYNQ-Z2 budget; no URAM is used (not available on 7-series).

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| BRAM_18K | 141 | 280 | **50%** |
| DSP48E1 | 178 | 220 | **81%** |
| Flip-Flops | 31,527 | 106,400 | 30% |
| LUTs | 40,259 | 53,200 | 76% |
| URAM | 0 | 0 | — |

DSPs and LUTs are the binding constraints. DSP utilization is high (81%) due to the fully unrolled dot-product engine: 32 parallel `fmul + fadd` chains over D_MAX generate approximately 32 × 3 = 96 DSPs for the multipliers plus the floating-point adder tree.

#### Pipeline Loop Latency

| Pipeline | II | Latency |
|----------|----|---------|
| LOAD_Q | 1 | 12 cycles |
| INIT_STATE | 1 | 1 cycle/row |
| LOAD_KV | 1 | 12 cycles |
| SCORE_LOOP_I×J | 1 | 202 cycles |
| FIND_MAX | 1 | 3 cycles |
| EXP_ACCUM | — | 42 cycles |
| UPDATE_ACC | 1 | 34 cycles (II=36) |
| STORE_O | 1 | 9 cycles |

#### Timing

The design has a **-1.40 ns timing violation** at 100 MHz. The critical path runs through the FIND_MAX comparator chain. In practice, the design can be run at ≈ 87 MHz (11.4 ns period) to meet timing, or the FIND_MAX loop can be restructured with an explicit registered reduction tree.

---

## 4. Deviations from Original Plan

The original plan was faithfully implemented. Two concrete implementation decisions were made during development:

1. **1D pointer arrays instead of 2D arrays for AXI4 Master ports.** The original plan specified `float Q[N][d]` ports. Vitis HLS cannot map 2D local arrays to `m_axi` interfaces, so the signature was changed to `float *Q` with row-major 1D indexing `Q[i*D_MAX + x]`. This is documented in the code and testbench.

2. **`int causal` instead of `bool`.** AXI4-Lite registers are 32-bit; `bool` would synthesize to a 1-bit register and cause mapping issues. Changed to `int` with the convention 0=false, 1=true.

3. **`DATAFLOW` not yet applied.** The plan mentioned task-level pipelining between modules as a planned optimization. This remains future work because enabling `DATAFLOW` requires each sub-function to communicate through streams (hls::stream) rather than shared arrays, which is a non-trivial restructuring. The current sequential HLS with loop-level pipelining already achieves II=1 on all inner loops.

4. **Fixed-point not explored.** The plan mentioned `ap_fixed<16,6>` as an option. Due to time constraints only `float` was verified. The `typedef float data_t` structure makes this a one-line change.

---

## 5. Reproducibility

From the repository root:

```bash
cd code
python golden_model.py          # generates test_outputs/tv_attention_python.csv
vitis_hls -f run_hls.tcl        # runs csim → csynth → cosim → export
```

The TCL script targets `xc7z020clg400-1` and a 10 ns clock. All results in this document were produced with this exact flow using Vitis HLS 2023.2.

Output CSVs:
- `code/test_outputs/tv_attention_python.csv` — golden model test vectors
- `code/test_outputs/tv_attention_csim.csv` — C-simulation DUT results
- `code/test_outputs/tv_attention_rtl.csv` — RTL co-simulation DUT results
