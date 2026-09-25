#!/usr/bin/env python3
"""
PGC Benchmark Results Parser & Speedup Calculator
Parses matrix multiplication execution metrics across Sequential, OpenMP, MPI, and CUDA models.
"""

def parse_and_report():
    baseline_time = 348.023990

    results = [
        {"name": "Sequential CPU", "time": 348.023990, "threads_nodes": "1 CPU Core", "verified": "4000.00"},
        {"name": "OpenMP Shared Memory", "time": 132.457362, "threads_nodes": "8 CPU Threads", "verified": "4000.00"},
        {"name": "MPI Distributed Memory", "time": 92.979510, "threads_nodes": "4 Nodes / VMs", "verified": "4000.00"},
        {"name": "CUDA GPU (Kernel Only)", "time": 0.146443, "threads_nodes": "16M CUDA Threads", "verified": "4000.00"},
        {"name": "CUDA GPU (Total Phase)", "time": 0.165004, "threads_nodes": "16M CUDA Threads", "verified": "4000.00"},
    ]

    # Matrix multiplication FLOPs = 2 * N^3 = 2 * (4000)^3 = 128 GFLOPs
    total_flops = 2.0 * (4000 ** 3)

    print("=" * 85)
    print("      PARALLEL & GRID COMPUTING (PGC) - 4000x4000 MATRIX MULTIPLICATION BENCHMARK     ")
    print("=" * 85)
    print(f"{'Model / Framework':<26} {'Resources':<18} {'Execution Time':<16} {'Speedup (x)':<14} {'GFLOPS':<10}")
    print("-" * 85)

    for r in results:
        speedup = baseline_time / r["time"]
        gflops = (total_flops / (r["time"] * 1e9))
        time_str = f"{r['time']:.6f} s" if r["time"] < 1 else f"{r['time']:.2f} s"
        print(f"{r['name']:<26} {r['threads_nodes']:<18} {time_str:<16} {speedup:<14.2f} {gflops:<10.2f}")

    print("=" * 85)

if __name__ == "__main__":
    parse_and_report()