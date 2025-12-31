#!/usr/bin/env python3
"""
Benchmark Analysis Script

Analyzes benchmark CSV results and generates visualizations.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def load_benchmark(path: str) -> pd.DataFrame:
    """Load benchmark CSV file"""
    df = pd.read_csv(path)
    return df

def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive statistics"""
    total = df['total_us']
    
    stats = {
        'frames': len(df),
        'mean_us': total.mean(),
        'std_us': total.std(),
        'min_us': total.min(),
        'max_us': total.max(),
        'p50_us': total.quantile(0.50),
        'p90_us': total.quantile(0.90),
        'p95_us': total.quantile(0.95),
        'p99_us': total.quantile(0.99),
        'fps_mean': 1e6 / total.mean(),
        'fps_p50': 1e6 / total.quantile(0.50),
        'fps_p99': 1e6 / total.quantile(0.99),
        'mean_preprocess_us': df['preprocess_us'].mean(),
        'mean_inference_us': df['inference_us'].mean(),
        'mean_postprocess_us': df['postprocess_us'].mean(),
        'frames_over_50ms': (total > 50000).sum(),
    }
    
    return stats

def print_report(stats: dict):
    """Print formatted report"""
    print("\n" + "="*60)
    print("BENCHMARK ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nFrames Analyzed: {stats['frames']}")
    print(f"Frames > 50ms:   {stats['frames_over_50ms']}")
    
    print("\nLATENCY (microseconds)")
    print(f"  Mean:     {stats['mean_us']:>10.1f}")
    print(f"  Std Dev:  {stats['std_us']:>10.1f}")
    print(f"  Min:      {stats['min_us']:>10.1f}")
    print(f"  Max:      {stats['max_us']:>10.1f}")
    print(f"  P50:      {stats['p50_us']:>10.1f}")
    print(f"  P90:      {stats['p90_us']:>10.1f}")
    print(f"  P95:      {stats['p95_us']:>10.1f}")
    print(f"  P99:      {stats['p99_us']:>10.1f}")
    
    print("\nCOMPONENT BREAKDOWN (mean)")
    print(f"  Preprocess:   {stats['mean_preprocess_us']:>10.1f} us")
    print(f"  Inference:    {stats['mean_inference_us']:>10.1f} us")
    print(f"  Postprocess:  {stats['mean_postprocess_us']:>10.1f} us")
    
    print("\nTHROUGHPUT (FPS)")
    print(f"  Mean:   {stats['fps_mean']:>10.2f}")
    print(f"  P50:    {stats['fps_p50']:>10.2f}")
    print(f"  P99:    {stats['fps_p99']:>10.2f}")
    
    print("\nVALIDATION")
    fps_pass = stats['fps_p99'] >= 20.0
    lat_pass = stats['p99_us'] <= 50000
    jitter_pass = stats['std_us'] <= 5000
    
    print(f"  ≥20 FPS (P99):    {'PASS ✓' if fps_pass else 'FAIL ✗'}")
    print(f"  ≤50ms latency:    {'PASS ✓' if lat_pass else 'FAIL ✗'}")
    print(f"  ≤5ms jitter:      {'PASS ✓' if jitter_pass else 'FAIL ✗'}")
    
    all_pass = fps_pass and lat_pass and jitter_pass
    print("\n" + "="*60)
    if all_pass:
        print("★ SYSTEM VALIDATED ★")
    else:
        print("✗ SYSTEM NOT VALIDATED ✗")
    print("="*60 + "\n")

def compare_benchmarks(path1: str, path2: str):
    """Compare two benchmark runs"""
    df1 = load_benchmark(path1)
    df2 = load_benchmark(path2)
    
    stats1 = calculate_statistics(df1)
    stats2 = calculate_statistics(df2)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    print(f"\nBaseline: {path1}")
    print(f"Current:  {path2}")
    
    def delta_pct(old, new, lower_is_better=True):
        pct = ((new - old) / old) * 100
        improved = (pct < 0) if lower_is_better else (pct > 0)
        color = "\033[32m" if improved else ("\033[31m" if abs(pct) > 5 else "")
        return f"{color}{pct:+.1f}%\033[0m"
    
    print("\nLatency Changes:")
    print(f"  Mean:  {stats1['mean_us']:.0f} → {stats2['mean_us']:.0f} us ({delta_pct(stats1['mean_us'], stats2['mean_us'])})")
    print(f"  P99:   {stats1['p99_us']:.0f} → {stats2['p99_us']:.0f} us ({delta_pct(stats1['p99_us'], stats2['p99_us'])})")
    print(f"  Jitter: {stats1['std_us']:.0f} → {stats2['std_us']:.0f} us ({delta_pct(stats1['std_us'], stats2['std_us'])})")
    
    print("\nThroughput Changes:")
    print(f"  Mean FPS: {stats1['fps_mean']:.1f} → {stats2['fps_mean']:.1f} ({delta_pct(stats1['fps_mean'], stats2['fps_mean'], False)})")
    print(f"  P99 FPS:  {stats1['fps_p99']:.1f} → {stats2['fps_p99']:.1f} ({delta_pct(stats1['fps_p99'], stats2['fps_p99'], False)})")
    print()

def plot_histogram(df: pd.DataFrame, output_path: str = None):
    """Plot latency histogram (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total latency histogram
        ax = axes[0, 0]
        ax.hist(df['total_us'] / 1000, bins=50, alpha=0.7, color='steelblue')
        ax.axvline(x=50, color='red', linestyle='--', label='50ms budget')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Total Latency Distribution')
        ax.legend()
        
        # Latency over time
        ax = axes[0, 1]
        ax.plot(df['frame_index'], df['total_us'] / 1000, alpha=0.5, linewidth=0.5)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Over Time')
        
        # Component breakdown
        ax = axes[1, 0]
        components = ['preprocess_us', 'inference_us', 'postprocess_us']
        means = [df[c].mean() / 1000 for c in components]
        ax.bar(['Preprocess', 'Inference', 'Postprocess'], means, color=['green', 'blue', 'orange'])
        ax.set_ylabel('Time (ms)')
        ax.set_title('Component Breakdown (Mean)')
        
        # CDF
        ax = axes[1, 1]
        sorted_latency = np.sort(df['total_us'] / 1000)
        cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
        ax.plot(sorted_latency, cdf)
        ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='P99')
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50ms')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('CDF')
        ax.set_title('Latency CDF')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib not available, skipping plot")

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("csv", help="Path to benchmark CSV file")
    parser.add_argument("--compare", help="Compare with another benchmark CSV")
    parser.add_argument("--plot", nargs='?', const='benchmark_analysis.png', 
                       help="Generate plots (optional: output path)")
    
    args = parser.parse_args()
    
    if not Path(args.csv).exists():
        print(f"Error: {args.csv} not found")
        return 1
    
    df = load_benchmark(args.csv)
    stats = calculate_statistics(df)
    print_report(stats)
    
    if args.compare:
        if Path(args.compare).exists():
            compare_benchmarks(args.compare, args.csv)
        else:
            print(f"Warning: {args.compare} not found, skipping comparison")
    
    if args.plot:
        plot_histogram(df, args.plot)
    
    return 0 if stats['fps_p99'] >= 20.0 else 1

if __name__ == "__main__":
    sys.exit(main())
