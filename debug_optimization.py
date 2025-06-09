#!/usr/bin/env python3
"""
Debug script to identify the broadcasting error
"""

import numpy as np
import traceback
from two_regime_walk_example import (
    confinement_walks_binary_morphology_optimized,
    choose_tx_start_locations
)

def debug_optimized_function():
    """Debug the optimized function step by step"""
    print("Debugging optimized function...")
    
    # Create simple test data
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:30, 20:30] = True
    starts = choose_tx_start_locations(mask, 3)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Starts shape: {starts.shape}")
    print(f"Starts: {starts}")
    
    arguments = {
        'mask': mask,
        'start_points': starts,
        'alphas': [1.5, 0.8],
        'Ds': [0.35, 0.35],
        'T': 10,  # Very small for debugging
        'trans': 0.5,
        'deltaT': 18,
        'L': 100
    }
    
    try:
        result = confinement_walks_binary_morphology_optimized(**arguments)
        print(f"Success! Result shape: {result[0].shape}")
    except Exception as e:
        print(f"Error: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_optimized_function()
