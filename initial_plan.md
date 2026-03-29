# IP Architecture

## Major Sub-Modules

The proposed accelerator is decomposed into a set of well-defined hardware sub-modules that mirror the structure of the tiled attention algorithm:

### 1. Q/K/V Loader Module
Responsible for reading input matrices (Q, K, V) from off-chip memory (e.g., DDR via AXI). It loads tiles of size \(B_Q \times d\) and \(B_K \times d\) into on-chip buffers (BRAM/URAM).

### 2. On-Chip Buffering System
Local storage for Q, K, and V tiles. These buffers enable data reuse across multiple compute cycles, significantly reducing external memory bandwidth requirements.

### 3. Dot-Product Engine
Computes partial attention scores:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}
$$

This is implemented as a pipelined MAC array with optional loop unrolling for parallelism.

### 4. Score Processing Unit (Max + Scaling)
Computes row-wise maximum values and performs score scaling. This module is critical for numerical stability and implements the online softmax update logic.

### 5. Exponential / Softmax Unit
Computes exponentials (via LUT or approximation) and accumulates normalization terms. It maintains:
- running max $m_i$
- normalization factor $l_i$

### 6. Weighted Value Accumulator
Multiplies softmax weights with V tiles and accumulates partial outputs:

$$
z_i = \sum_j (p_{ij} \cdot v_j)
$$

### 7. Output Writeback Module
Writes the final output tile \(O\) back to off-chip memory.

### 8. Top-Level Controller (FSM / Dataflow Scheduler)
Coordinates:
- tile iteration over Q and K/V
- buffer loading and reuse
- synchronization between compute stages

---

## Sub-Module Communication

The design follows a **streaming + tiled dataflow architecture**, with clear separation between memory movement and computation.

### Interfaces

#### AXI4 Master Interface
- Used by loader and writeback modules for global memory access
- Supports burst transfers for Q/K/V tiles

#### AXI4-Lite Interface
- Used for control signals (e.g., start, done, N, d, causal flag)

#### AXI-Stream / FIFO Channels (Internal)
- Connect compute stages in a pipelined fashion
- Enable concurrent execution of:
  - load
  - compute
  - store

---

### Dataflow Pattern

1. Q tile is loaded into local buffer  
2. For each K/V tile:
   - K and V tiles are streamed into compute units  
   - Dot-product engine generates scores  
   - Score processing unit updates max and normalization  
   - Accumulator updates output tile  
3. Final normalized output is written back  

This follows a **producer–consumer pipeline**:
- Loader → Compute → Accumulator → Writeback


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

- Start with **dot-product engine** (basic correctness)
- Add **softmax unit** (numerical stability)
- Integrate **accumulator**
- Finally integrate memory and control

This reduces debugging complexity significantly.

---

### 2. Independent Verification
Each sub-module can be tested against the Python golden model:

- Dot-product → compare partial scores  
- Softmax → compare normalized weights  
- Full pipeline → compare final output  

This enables **unit testing before full system integration**, which is essential in hardware design.

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

High-Level Synthesis (HLS) tools (e.g., Vitis HLS) exploit task-level parallelism using `DATAFLOW` pragmas. A modular architecture maps naturally to:

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

Overall, it ensures the design is both **implementable within a course timeline** and **representative of real accelerator architectures used in modern ML systems**.
