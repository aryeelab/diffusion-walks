#!/usr/bin/env python3
"""
Test script for the optimized diffusion simulation implementations
"""

import numpy as np
from timeit import timeit
from functools import partial
import sys

# Import our functions
from two_regime_walk_example import (
    confinement_walks_binary_morphology,
    confinement_walks_binary_morphology_optimized,
    vectorized_confinement,
    vectorized_confinement_optimized,
    choose_tx_start_locations
)

def create_test_mask(size=(100, 100)):
    """Create a synthetic binary mask for testing"""
    mask = np.zeros(size, dtype=bool)
    # Create some circular regions
    y, x = np.ogrid[:size[0], :size[1]]
    centers = [(25, 25), (75, 75), (25, 75), (75, 25)]
    for cy, cx in centers:
        mask_circle = (x - cx)**2 + (y - cy)**2 < 15**2
        mask |= mask_circle
    return mask

def test_basic_functionality():
    """Test that all implementations work without errors"""
    print("Testing basic functionality...")
    
    # Create test data
    mask = create_test_mask()
    starts = choose_tx_start_locations(mask, 5)
    
    arguments = {
        'mask': mask,
        'start_points': starts,
        'alphas': [1.5, 0.8],
        'Ds': [0.35, 0.35],
        'T': 50,
        'trans': 0.5,
        'deltaT': 18,
        'L': 200
    }
    
    implementations = {
        'Original Slow': confinement_walks_binary_morphology,
        'Optimized Slow': confinement_walks_binary_morphology_optimized,
        'Original Fast': vectorized_confinement,
        'Optimized Fast': vectorized_confinement_optimized
    }
    
    results = {}
    for name, func in implementations.items():
        try:
            print(f"  Testing {name}...", end=" ")
            result = func(**arguments)
            if isinstance(result, tuple):
                data, labels = result
                print(f"✓ Shape: {data.shape}")
                results[name] = data
            else:
                print(f"✓ Shape: {result.shape}")
                results[name] = result
        except Exception as e:
            print(f"✗ Error: {e}")
            results[name] = None
    
    return results

def benchmark_implementations():
    """Benchmark all implementations with different problem sizes"""
    print("\nBenchmarking implementations...")
    
    test_cases = [
        {'n': 10, 't': 100, 'name': 'Small'},
        {'n': 25, 't': 200, 'name': 'Medium'},
        {'n': 50, 't': 500, 'name': 'Large'}
    ]
    
    for case in test_cases:
        print(f"\n{case['name']} test (n={case['n']}, t={case['t']}):")
        print("-" * 50)
        
        # Create test data
        mask = create_test_mask()
        starts = choose_tx_start_locations(mask, case['n'])
        
        arguments = {
            'mask': mask,
            'start_points': starts,
            'alphas': [1.5, 0.8],
            'Ds': [0.35, 0.35],
            'T': case['t'],
            'trans': 0.5,
            'deltaT': 18,
            'L': 200
        }
        
        implementations = {
            'Original Slow': partial(confinement_walks_binary_morphology, **arguments),
            'Optimized Slow': partial(confinement_walks_binary_morphology_optimized, **arguments),
            'Original Fast': partial(vectorized_confinement, **arguments),
            'Optimized Fast': partial(vectorized_confinement_optimized, **arguments)
        }
        
        times = {}
        iterations = 3 if case['n'] <= 25 else 1
        
        for name, func in implementations.items():
            try:
                time_taken = timeit(func, number=iterations) / iterations
                times[name] = time_taken
                print(f"  {name:15}: {time_taken:.4f}s")
            except Exception as e:
                print(f"  {name:15}: ERROR - {e}")
                times[name] = None
        
        # Calculate speedups
        print("\n  Speedup analysis:")
        if times['Original Slow'] and times['Optimized Slow']:
            speedup = times['Original Slow'] / times['Optimized Slow']
            print(f"    Slow optimization: {speedup:.2f}x faster")
        
        if times['Original Fast'] and times['Optimized Fast']:
            speedup = times['Original Fast'] / times['Optimized Fast']
            print(f"    Fast optimization: {speedup:.2f}x faster")
        
        if times['Original Slow'] and times['Original Fast']:
            speedup = times['Original Slow'] / times['Original Fast']
            print(f"    Original vectorization: {speedup:.2f}x faster")
        
        if times['Optimized Slow'] and times['Optimized Fast']:
            speedup = times['Optimized Slow'] / times['Optimized Fast']
            print(f"    Optimized vectorization: {speedup:.2f}x faster")
        
        if times['Original Slow'] and times['Optimized Fast']:
            speedup = times['Original Slow'] / times['Optimized Fast']
            print(f"    Overall improvement: {speedup:.2f}x faster")

def main():
    print("=" * 60)
    print("DIFFUSION SIMULATION OPTIMIZATION TEST")
    print("=" * 60)
    
    # Test basic functionality
    results = test_basic_functionality()
    
    # Run benchmarks
    benchmark_implementations()
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main()
