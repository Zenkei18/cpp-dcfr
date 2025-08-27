#!/usr/bin/env python3
"""
Create a performance comparison chart from benchmark results.
"""

import matplotlib.pyplot as plt
import numpy as np

# Performance data from benchmarks
components = ['State Encoding', 'CFR Traversal', 'Device Transfers', 'Overall']
original_times = [0.03, 6.53, 0.01, 1.0]  # Normalized for overall
optimized_times = [0.02, 0.02, 0.00, 0.127]  # 1/7.85 for overall
speedups = [1.28, 386.78, 2.5, 7.85]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Execution Time Comparison
x = np.arange(len(components))
width = 0.35

bars1 = ax1.bar(x - width/2, original_times, width, label='Original', color='#ff6b6b', alpha=0.8)
bars2 = ax1.bar(x + width/2, optimized_times, width, label='Optimized', color='#4ecdc4', alpha=0.8)

ax1.set_xlabel('Components')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Time: Original vs Optimized')
ax1.set_xticks(x)
ax1.set_xticklabels(components, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
def add_value_labels(ax, bars, values, format_str='{:.2f}'):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                format_str.format(value), ha='center', va='bottom', fontsize=9)

add_value_labels(ax1, bars1, original_times)
add_value_labels(ax1, bars2, optimized_times)

# Plot 2: Speedup Factors
colors = ['#51cf66' if s >= 1.5 else '#ffd43b' if s >= 1.1 else '#ff8787' for s in speedups]
bars3 = ax2.bar(components, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

ax2.set_xlabel('Components')
ax2.set_ylabel('Speedup Factor (x)')
ax2.set_title('Performance Speedup by Component')
ax2.set_xticklabels(components, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Add horizontal line at 1.5x target
ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Target (1.5x)')
ax2.legend()

# Add value labels on speedup bars
for bar, value in zip(bars3, speedups):
    height = bar.get_height()
    if value > 100:
        label = f'{value:.0f}x'
    else:
        label = f'{value:.1f}x'
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(speedups) * 0.01,
             label, ha='center', va='bottom', fontsize=10, fontweight='bold')

# Adjust layout and save
plt.tight_layout()
plt.savefig('bench/results/performance_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('bench/results/performance_chart.pdf', bbox_inches='tight')

print("Performance charts saved:")
print("- bench/results/performance_chart.png")
print("- bench/results/performance_chart.pdf")

# Display key metrics
print("\n" + "="*60)
print("DEEPCFR PERFORMANCE OPTIMIZATION RESULTS")
print("="*60)
print(f"Overall Speedup: {speedups[-1]:.2f}x")
print(f"Target Achievement: {'✅ EXCEEDED' if speedups[-1] >= 1.5 else '❌ NOT MET'}")
print("\nComponent Breakdown:")
for comp, speedup in zip(components[:-1], speedups[:-1]):
    status = "✅ PASS" if speedup >= 1.5 else "⚠️ PARTIAL" if speedup >= 1.1 else "❌ FAIL"
    print(f"  {comp:18s}: {speedup:6.1f}x  {status}")

print(f"\nBenchmark Target (≥1.5x): {'✅ ACHIEVED' if speedups[-1] >= 1.5 else '❌ NOT MET'}")
print("="*60)