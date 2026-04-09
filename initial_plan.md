## GitHub Repository

https://github.com/ricardotk002/hwdesign-flash-attn-accel

## Team Members

- Anuj Apte (aa13231@nyu.edu)
- Ricardo Díaz (erd9862@nyu.edu)
- Raquel Brown (rb6191@nyu.edu)

## IP Architecture

### What the IP does

This IP implements a simplified FlashAttention-style accelerator for the forward pass of single-head transformer attention. Its goal is to compute attention efficiently without storing the full $N \times N$ score matrix in external memory.

The problem it solves is that standard attention has high memory traffic because it forms all pairwise scores between queries and keys. This becomes expensive as sequence length grows. The IP applies to **machine learning inference**, especially transformer-based models on FPGA or SoC platforms.

Its core functionality is:

- receive $(Q)$, $(K)$, and $(V)$ matrices
- compute scaled dot-product attention
- apply softmax
- produce the output matrix $(O)$

Instead of materializing all attention scores at once, it processes the computation in tiles and keeps intermediate values on chip.

---

### How it interacts with the PS and peripherals

The Processing System (PS) configures and controls the IP through control registers, typically over AXI-Lite. The PS is responsible for:

- providing the base addresses of the $(Q)$, $(K)$, $(V)$, and output buffers
- setting parameters such as sequence length $(N)$, embedding dimension $(d)$, and optional causal mode
- starting the accelerator and checking completion

The IP reads:

- query matrix $(Q \in \mathbb{R}^{N \times d})$
- key matrix $(K \in \mathbb{R}^{N \times d})$
- value matrix $(V \in \mathbb{R}^{N \times d})$

The IP writes:

- output matrix $(O \in \mathbb{R}^{N \times d})$

These transfers would typically use AXI master interfaces to DDR memory. Optional peripherals include DMA engines for efficient data movement between PS memory and PL logic.

---

### Mathematical operations involved

The IP computes standard scaled dot-product attention:

$$
O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

For each query row $q_i$, the score with key row $k_j$ is:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}
$$

When causal mode is enabled, future positions are masked before softmax:

$$
s_{ij} = -\infty \quad \text{for } j > i
$$

This ensures each query attends only to positions up to and including its own. The masking is applied by the Dot-Product Engine before scores are passed downstream.

The softmax probability is:

$$
p_{ij} = \frac{e^{s_{ij}}}{\sum_{k=1}^{N} e^{s_{ik}}}
$$

The final output row is:

$$
o_i = \sum_{j=1}^{N} p_{ij} v_j
$$

To avoid storing the full score matrix, the IP uses tiled processing and an online softmax update. For each tile, it maintains:

- a running maximum $m_i$
- a running normalization term $l_i$
- a running output accumulator $o_i$

Updates per tile:

$$
m_i^{new} = \max(m_i,\ \max_j s_{ij})
$$

$$
l_i^{new} = e^{m_i - m_i^{new}} l_i + \sum_j e^{s_{ij} - m_i^{new}}
$$

$$
o_i^{new} =
\frac{
e^{m_i - m_i^{new}} l_i \cdot o_i +
\sum_j e^{s_{ij} - m_i^{new}} v_j
}{
l_i^{new}
}
$$

---

### High-level algorithm

```
for each Q tile:
    load Q tile
    initialize m, l, and output accumulator

    for each K/V tile:
        load K tile and V tile
        compute scores = Q_tile x K_tile^T / sqrt(d)
        update running max (m)
        update normalization (l)
        update output accumulator with V

    write output tile
```

## Major Sub-Modules

The proposed accelerator is decomposed into a set of well-defined hardware sub-modules that mirror the structure of the tiled attention algorithm:

### 1. Q/K/V Loader Module

Responsible for reading input matrices (Q, K, V) from off-chip memory (e.g., DDR via AXI). It loads tiles of size $B_Q \times d$ and $B_K \times d$ into on-chip buffers (BRAM).

### 2. On-Chip Buffering System

Local storage for Q, K, and V tiles. These buffers enable data reuse across multiple compute cycles, significantly reducing external memory bandwidth requirements.

### 3. Dot-Product Engine

Computes partial attention scores:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}
$$

This is implemented as a pipelined MAC array with optional loop unrolling for parallelism.

When causal mode is enabled, scores at future positions ($k_0 + j > q_0 + i$) are set to $-\infty$ immediately after computation and before being passed to the Online Softmax Unit. This causes their exponential to underflow to zero, effectively excluding them from the softmax.

### 4. Online Softmax Unit

Owns all online softmax state. For each K/V tile it:

- finds the tile row-wise maximum and updates the running max $m_i^{new} = \max(m_i,\ \max_j s_{ij})$
- computes the correction factor $e^{m_i - m_i^{new}}$ to rescale prior state
- computes per-element exponentials $e^{s_{ij} - m_i^{new}}$
- updates the running normalization $l_i^{new} = e^{m_i - m_i^{new}} l_i + \sum_j e^{s_{ij} - m_i^{new}}$
- emits the correction factor, scaled exponentials, and updated $l_i$ to the accumulator

Owns: $m_i$, $l_i$

### 5. Weighted Value Accumulator

Receives the correction factor $e^{m_i - m_i^{new}}$, scaled exponentials $e^{s_{ij} - m_i^{new}}$, and updated $l_i^{new}$ from the Online Softmax Unit. Multiplies the scaled exponentials with the V tile and updates the running output accumulator. Because normalization is applied incrementally, the accumulator always holds a valid normalized partial result:

$$
o_i \leftarrow \frac{e^{m_i - m_i^{new}} \cdot l_i \cdot o_i + \sum_j e^{s_{ij} - m_i^{new}} \cdot v_j}{l_i^{new}}
$$

After the last K/V tile, $o_i$ is the final normalized output row — no separate normalization step is needed.

### 6. Output Writeback Module

Writes the final output tile \(O\) back to off-chip memory.

### 7. Top-Level Controller (FSM / Dataflow Scheduler)

Coordinates:

- tile iteration over Q and K/V
- buffer loading and reuse
- synchronization between compute stages

---

## Sub-Module Communication

The design follows a streaming + tiled dataflow architecture, with clear separation between memory movement and computation.

### Interfaces

#### AXI4 Master Interface

- Used by loader and writeback modules for global memory access
- Supports burst transfers for Q/K/V tiles

#### AXI4-Lite Interface

- Used for control signals (e.g., start, done, N, d, causal flag)

#### AXI-Stream / FIFO Channels (Internal)

- Connect compute stages in a pipelined fashion
- Can enable concurrent execution of load, compute, and store once
  `#pragma HLS DATAFLOW` is applied at the top level; in the current
  sequential HLS implementation this is a planned optimization

---

### Dataflow Pattern

1. Q tile is loaded into local buffer
2. For each K/V tile:
   - K and V tiles are loaded into on-chip buffers by the Loader
   - Dot-product engine generates scores
   - Online Softmax Unit updates running max, normalization, and emits scaled exponentials
   - Weighted Value Accumulator updates output tile
3. Final normalized output is written back

This follows a producer–consumer pipeline:

- Loader → Buffer → Compute → Accumulator → Writeback

---

### Control Signals

- `start`, `done` (top-level execution control)
- tile loop counters ($q_{idx}$, $k_{idx}$)
- valid/ready handshakes between modules
- synchronization signals for updating $m_i$, $l_i$, and accumulators

---

## Why Modularization Matters

The decomposition is intentional and critical for both correctness and performance.

### 1. Incremental Development

Each module can be designed and verified independently:

- Start with dot-product engine (basic correctness)
- Add online softmax unit (numerical stability)
- Integrate weighted value accumulator
- Finally integrate memory and control

This reduces debugging complexity significantly.

---

### 2. Independent Verification

Each sub-module can be tested against the Python golden model:

- Dot-product → compare partial scores
- Online Softmax Unit → compare running max, normalization, and weights
- Full pipeline → compare final output

This enables unit testing before full system integration, which is essential in hardware design.

---

### 3. Performance Optimization

Different modules expose different optimization levers:

- Dot-product → loop unrolling, DSP utilization
- Buffers → BRAM partitioning
- Dataflow → pipeline depth and initiation interval

Because modules are decoupled, optimizations can be applied locally without breaking the entire system.

---

### 4. Scalability and Design Space Exploration

Modular design allows easy experimentation with:

- tile sizes ($B_Q$), ($B_K$)
- precision (float vs fixed-point)
- parallelism (number of MAC units)

Without modularization, these changes would require rewriting large portions of the design.

---

### 5. Alignment with HLS Dataflow Model

High-Level Synthesis tools (Vitis HLS) exploit task-level parallelism using `DATAFLOW` pragmas. A modular architecture maps naturally to:

- separate functions per module
- streaming FIFOs between them
- concurrent execution in hardware

This results in higher throughput and better resource utilization.

---

## Summary

The IP is structured as a pipeline of specialized compute and memory modules connected through streaming interfaces and controlled by a top-level scheduler. This modular decomposition:

- mirrors the tiled attention algorithm
- enables efficient memory reuse
- supports parallel execution
- allows independent testing and optimization

Overall, it ensures the design is both implementable within a course timeline and representative of real accelerator architectures used in modern ML systems.
