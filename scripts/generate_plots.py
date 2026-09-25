import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('images', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dpi = 300

models = ['Sequential\n(1 CPU Core)', 'OpenMP\n(8 CPU Threads)', 'MPI Cluster\n(4 Processes / VMs)', 'CUDA GPU\n(RTX 4500 Ada)']
execution_times = [348.023990, 132.457362, 92.979510, 0.165004]
speedup_factors = [1.00, 2.63, 3.74, 2109.18]

colors = ['#d9534f', '#f0ad4e', '#0275d8', '#5cb85c']

# -------------------------------------------------------------------------
# Plot 1: Execution Time Comparison (Log Scale & Linear)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=fig_dpi)
bars = ax.bar(models, execution_times, color=colors, width=0.5, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Execution Time (Seconds) - Log Scale', fontsize=12, fontweight='bold')
ax.set_title('Matrix Multiplication (4000x4000) Execution Time Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_yscale('log')
ax.set_ylim(0.01, 1000)

for bar in bars:
    height = bar.get_height()
    if height < 1.0:
        label = f'{height:.6f} s'
    else:
        label = f'{height:.2f} s'
    ax.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('images/execution_time_comparison.png')
plt.close()

# -------------------------------------------------------------------------
# Plot 2: Speedup Factor Comparison
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=fig_dpi)
bars = ax.bar(models, speedup_factors, color=colors, width=0.5, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Speedup Factor (x Baseline)', fontsize=12, fontweight='bold')
ax.set_title('Parallel & GPU Speedup Comparison over Sequential Baseline (348.02s)', fontsize=14, fontweight='bold', pad=15)
ax.set_yscale('log')
ax.set_ylim(0.5, 5000)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,.2f}x',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('images/speedup_comparison.png')
plt.close()

# -------------------------------------------------------------------------
# Plot 3: Comprehensive Multi-Panel Dashboard
# -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=fig_dpi)
fig.suptitle('Parallel & Grid Computing (PGC) Performance Dashboard: 4000x4000 Matrix Multiplication', 
             fontsize=16, fontweight='bold', y=0.98)

# Subplot 1: CPU Models Execution Time
cpu_models = ['Sequential', 'OpenMP (8 Threads)', 'MPI (4 Nodes)']
cpu_times = [348.023990, 132.457362, 92.979510]
axs[0, 0].bar(cpu_models, cpu_times, color=['#d9534f', '#f0ad4e', '#0275d8'], width=0.45, edgecolor='black')
axs[0, 0].set_title('CPU Execution Time (Seconds)', fontsize=12, fontweight='bold')
axs[0, 0].set_ylabel('Time (s)', fontsize=10)
axs[0, 0].set_ylim(0, 400)
for bar in axs[0, 0].patches:
    axs[0, 0].annotate(f'{bar.get_height():.2f} s', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

# Subplot 2: GPU Acceleration Impact
gpu_labels = ['Sequential CPU', 'CUDA Kernel', 'CUDA Total Phase']
gpu_times = [348.023990, 0.146443, 0.165004]
axs[0, 1].bar(gpu_labels, gpu_times, color=['#d9534f', '#5cb85c', '#2e7d32'], width=0.45, edgecolor='black')
axs[0, 1].set_title('CPU vs CUDA GPU Execution Time (Log Scale)', fontsize=12, fontweight='bold')
axs[0, 1].set_ylabel('Time (s)', fontsize=10)
axs[0, 1].set_yscale('log')
axs[0, 1].set_ylim(0.01, 1000)
for bar in axs[0, 1].patches:
    h = bar.get_height()
    lbl = f'{h:.4f} s' if h < 1 else f'{h:.2f} s'
    axs[0, 1].annotate(lbl, (bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

# Subplot 3: Speedup Factors
axs[1, 0].bar(models, speedup_factors, color=colors, width=0.45, edgecolor='black')
axs[1, 0].set_title('Speedup Factor vs Sequential Baseline', fontsize=12, fontweight='bold')
axs[1, 0].set_ylabel('Speedup (x)', fontsize=10)
axs[1, 0].set_yscale('log')
axs[1, 0].set_ylim(0.5, 5000)
for bar in axs[1, 0].patches:
    axs[1, 0].annotate(f'{bar.get_height():,.2f}x', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

# Subplot 4: Computational Throughput (GFLOPS)
# 4000x4000 matrix multiplication = 2 * N^3 = 2 * (4000)^3 = 128,000,000,000 FLOPs = 128 GFLOPs
gflops = [128.0 / t for t in execution_times]
axs[1, 1].bar(models, gflops, color=colors, width=0.45, edgecolor='black')
axs[1, 1].set_title('Computational Throughput (GFLOPS)', fontsize=12, fontweight='bold')
axs[1, 1].set_ylabel('GFLOPS (Higher is Better)', fontsize=10)
axs[1, 1].set_yscale('log')
axs[1, 1].set_ylim(0.1, 2000)
for bar in axs[1, 1].patches:
    axs[1, 1].annotate(f'{bar.get_height():,.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('images/overall_performance_dashboard.png')
plt.close()

print("All PGC benchmark plots generated successfully in images/ directory.")