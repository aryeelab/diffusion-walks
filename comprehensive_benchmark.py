#!/usr/bin/env python3
"""
Comprehensive benchmark of all diffusion simulation implementations
"""

import numpy as np
from timeit import timeit
from functools import partial
import matplotlib.pyplot as plt

from two_regime_walk_example import (
    confinement_walks_binary_morphology,
    confinement_walks_binary_morphology_optimized,
    vectorized_confinement,
    vectorized_confinement_optimized,
    choose_tx_start_locations
)

def create_synthetic_mask(size=(200, 200)):
    """Create a synthetic binary mask for testing"""
    mask = np.zeros(size, dtype=bool)
    # Create some circular regions
    y, x = np.ogrid[:size[0], :size[1]]
    centers = [(50, 50), (150, 150), (50, 150), (150, 50)]
    for cy, cx in centers:
        mask_circle = (x - cx)**2 + (y - cy)**2 < 30**2
        mask |= mask_circle
    return mask

def run_comprehensive_benchmark():
    """Run comprehensive benchmark with the same parameters as main()"""
    print("=" * 80)
    print("COMPREHENSIVE DIFFUSION SIMULATION PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Use the same parameters as the original main() function
    D = 0.07
    glandA = 1.5 
    stromaA = 0.8
    transition_prob = 0.5
    dt = 18
    
    # Test different problem sizes
    test_cases = [
        {'n': 10, 't': 200, 'name': 'Small', 'iterations': 5},
        {'n': 25, 't': 500, 'name': 'Medium', 'iterations': 3},
        {'n': 50, 't': 1000, 'name': 'Large (Original)', 'iterations': 2},
        {'n': 100, 't': 1000, 'name': 'Extra Large', 'iterations': 1}
    ]
    
    all_results = {}
    
    for case in test_cases:
        print(f"\n{case['name']} Test (n={case['n']}, t={case['t']}):")
        print("-" * 60)
        
        # Create test data
        mask = create_synthetic_mask()
        starts = choose_tx_start_locations(mask, case['n'])
        
        arguments = {
            'mask': mask,
            'start_points': starts,
            'alphas': [glandA, stromaA],
            'Ds': [5*D, 5*D],
            'T': case['t'],
            'trans': transition_prob,
            'deltaT': dt,
            'L': 500
        }
        
        implementations = {
            'Original Slow': partial(confinement_walks_binary_morphology, **arguments),
            'Optimized Slow': partial(confinement_walks_binary_morphology_optimized, **arguments),
            'Original Fast': partial(vectorized_confinement, **arguments),
            'Optimized Fast': partial(vectorized_confinement_optimized, **arguments)
        }
        
        case_results = {}
        
        for name, func in implementations.items():
            try:
                print(f"  {name:15}: ", end="", flush=True)
                time_taken = timeit(func, number=case['iterations']) / case['iterations']
                case_results[name] = time_taken
                print(f"{time_taken:.4f}s")
            except Exception as e:
                print(f"ERROR - {e}")
                case_results[name] = None
        
        all_results[case['name']] = case_results
        
        # Calculate and display speedups for this case
        print(f"\n  Speedup Analysis for {case['name']}:")
        if case_results['Original Slow'] and case_results['Optimized Slow']:
            speedup = case_results['Original Slow'] / case_results['Optimized Slow']
            print(f"    Slow optimization speedup: {speedup:.2f}x")
        
        if case_results['Original Fast'] and case_results['Optimized Fast']:
            speedup = case_results['Original Fast'] / case_results['Optimized Fast']
            print(f"    Fast optimization speedup: {speedup:.2f}x")
        
        if case_results['Original Slow'] and case_results['Original Fast']:
            speedup = case_results['Original Slow'] / case_results['Original Fast']
            print(f"    Original vectorization speedup: {speedup:.2f}x")
        
        if case_results['Optimized Slow'] and case_results['Optimized Fast']:
            speedup = case_results['Optimized Slow'] / case_results['Optimized Fast']
            print(f"    Optimized vectorization speedup: {speedup:.2f}x")
        
        if case_results['Original Slow'] and case_results['Optimized Fast']:
            speedup = case_results['Original Slow'] / case_results['Optimized Fast']
            print(f"    Overall best improvement: {speedup:.2f}x")
    
    # Summary report
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print("\nTiming Results (seconds):")
    print(f"{'Test Case':<20} {'Orig Slow':<12} {'Opt Slow':<12} {'Orig Fast':<12} {'Opt Fast':<12}")
    print("-" * 80)
    
    for case_name, results in all_results.items():
        orig_slow = f"{results['Original Slow']:.4f}" if results['Original Slow'] else "ERROR"
        opt_slow = f"{results['Optimized Slow']:.4f}" if results['Optimized Slow'] else "ERROR"
        orig_fast = f"{results['Original Fast']:.4f}" if results['Original Fast'] else "ERROR"
        opt_fast = f"{results['Optimized Fast']:.4f}" if results['Optimized Fast'] else "ERROR"
        
        print(f"{case_name:<20} {orig_slow:<12} {opt_slow:<12} {orig_fast:<12} {opt_fast:<12}")
    
    print("\nSpeedup Factors:")
    print(f"{'Test Case':<20} {'Slow Opt':<12} {'Fast Opt':<12} {'Orig Vec':<12} {'Best':<12}")
    print("-" * 80)
    
    for case_name, results in all_results.items():
        slow_opt = f"{results['Original Slow']/results['Optimized Slow']:.2f}x" if results['Original Slow'] and results['Optimized Slow'] else "N/A"
        fast_opt = f"{results['Original Fast']/results['Optimized Fast']:.2f}x" if results['Original Fast'] and results['Optimized Fast'] else "N/A"
        orig_vec = f"{results['Original Slow']/results['Original Fast']:.2f}x" if results['Original Slow'] and results['Original Fast'] else "N/A"
        best = f"{results['Original Slow']/results['Optimized Fast']:.2f}x" if results['Original Slow'] and results['Optimized Fast'] else "N/A"
        
        print(f"{case_name:<20} {slow_opt:<12} {fast_opt:<12} {orig_vec:<12} {best:<12}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    print("\nKey Optimizations Applied:")
    print("1. SLOW IMPLEMENTATION OPTIMIZATIONS:")
    print("   - Single mask labeling instead of N repetitions")
    print("   - Pre-computed displacements for all particles")
    print("   - Vectorized boundary checking and compartment lookup")
    print("   - Reduced function call overhead")
    print("   - Optimized memory allocation (float32 vs float64)")
    
    print("\n2. FAST IMPLEMENTATION OPTIMIZATIONS:")
    print("   - More efficient displacement generation")
    print("   - Optimized memory access patterns")
    print("   - Streamlined reflection logic with limited attempts")
    print("   - In-place boundary operations")
    print("   - Better cache efficiency")
    
    print("\n3. WHY VECTORIZED IS FASTER:")
    print("   - Eliminates O(N) redundant operations")
    print("   - Leverages NumPy's optimized C implementations")
    print("   - Reduces Python loop overhead")
    print("   - Better memory locality and cache efficiency")
    print("   - Batch processing of all particles simultaneously")

def main():
    run_comprehensive_benchmark()

if __name__ == "__main__":
    main()
