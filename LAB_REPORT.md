# Benchmarking Parallel Computing Paradigms: $4000 \times 4000$ Matrix Multiplication

This project presents an empirical benchmarking study comparing **Sequential Processing**, **Shared-Memory Multi-Threading (OpenMP)**, and **Distributed-Memory Message Passing (MPI)** using a high-density $4000 \times 4000$ matrix multiplication workload.

---

## Technical Specifications & Environment

### Experiment Parameters
* **Matrix Setup:** Square dense matrices ($A, B, C$) of size $4000 \times 4000$, populated with initial values of `1.0`.
* **Workload Volume:** $O(N^3) \approx 6.4 \times 10^{10}$ floating-point operations.
* **Correctness Criteria:** Every element in output matrix $C$ evaluates to $4000.00$:

$$\sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$

### System Environments
* **Sequential & OpenMP:** Run on Ubuntu via WSL2.
* **MPI Cluster:** Executed across a 4-node Ubuntu virtualized network (`master`, `worker1`, `worker2`, `worker3`).

---

## Implementation Architectures

### 1. Serial (Baseline)
Executes standard triply nested loops (`i`, `j`, `k`) on a single logical core. It serves as the control baseline with zero inter-process overhead or thread synchronization delays.

### 2. Multi-Threaded (OpenMP)
Employs shared-memory parallelism using compiler directives (`#pragma omp parallel for private(j, k)`). Outer loop iterations are split across 8 concurrent CPU worker threads accessing a unified address space without requiring manual data transfers.

### 3. Distributed-Memory (MPI)
Utilizes process-level distribution across 4 virtualized nodes:
* **Rank 0 (Master):** Initializes and holds complete copies of $A$, $B$, and $C$.
* **Scatter & Broadcast Phase:** `MPI_Scatter` partitions matrix $A$ into 1000-row chunks per node, while `MPI_Bcast` transmits the complete $4000 \times 4000$ matrix $B$ across all ranks.
* **Computation & Aggregation:** Nodes calculate their assigned sub-matrix before `MPI_Gather` collects partial outputs back into Rank 0.

---

## Feature Matrix

| Design Aspect | Serial CPU | OpenMP (Shared-Memory) | MPI (Distributed-Memory) |
| :--- | :--- | :--- | :--- |
| **Model** | Single-threaded | Multi-threaded (Fork-Join) | Multi-process Message Passing |
| **Hardware** | 1 CPU Core | 1 Host / 8 Threads | 4 VM Cluster Nodes |
| **Memory Access** | Local RAM | Shared RAM Address Space | Disjoint / Isolated Memory |
| **Work Partitioning** | None (Serial) | Dynamic/Static Loop Splitting | Direct Row Slicing (1000 rows/rank) |
| **Communication** | None | Implicit Shared Memory | Explicit Network IPC (`Bcast`, `Scatter`, `Gather`) |
| **Overhead Types** | None | Thread Sync & Bus Contention | Network Latency & Inter-VM Sockets |

---

## Experimental Benchmark Results

### Evaluation Formulas

$$\text{Speedup } (S) = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$$

$$\text{Parallel Efficiency } (E) = \left( \frac{S}{P} \right) \times 100\%$$

*(where $P$ is the count of compute units)*

### Performance Breakdown

| Approach | Compute Resources | Wall-Clock Time (s) | Measured Speedup | Efficiency | Verification Check (`C[0][0]`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sequential** | 1 Core | **266.48 s** | **1.00×** | 100.0% | `4000.00` |
| **OpenMP** | 8 Threads | **41.02 s** | **6.50×** | 81.2% | `4000.00` |
| **MPI** | 4 Ranks / 4 VMs | **226.17 s** | **1.18×** | 29.5% | `4000.00` |

### Speedup Calculations
* **OpenMP Speedup:** $\frac{266.477234\,\text{s}}{41.021555\,\text{s}} \approx \mathbf{6.50\times}$
* **MPI Speedup:** $\frac{266.477234\,\text{s}}{226.167575\,\text{s}} \approx \mathbf{1.18\times}$

---

## Key Performance Insights

1. **Shared-Memory Efficiency:** OpenMP delivered the highest speedup (**6.50×**), dropping execution time from 266.48s to 41.02s. Directly referencing shared memory completely bypasses network buffer copies.
2. **Distributed Overhead:** MPI achieved a modest **1.18×** speedup. Sending the full ~128 MB matrix $B$ via `MPI_Bcast` and scattering/gathering matrix slices introduced communication latency that offset parallel processing gains over virtual networks.
3. **Core vs. Network Scaling:** In-memory thread scaling (OpenMP) is significantly faster than distributed socket communication across virtual machines (MPI) for dense matrix operations of this size.
4. **Validation:** All three strategies yielded `C[0][0] = 4000.00`, confirming complete numerical accuracy across sequential, threaded, and distributed executions.
