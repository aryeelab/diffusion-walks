# Diffusion Walks Project

This project implements two-regime random walk simulations for studying particle diffusion in environments with different diffusion states (e.g., glands vs stroma). The project includes both original implementations and highly optimized versions for maximum performance.

## Environment Setup

### Using Conda (Recommended)

1. **Activate the environment:**
   ```bash
   conda activate diffusion-walks
   ```

2. **If you need to recreate the environment:**
   ```bash
   conda env create -f environment.yml
   ```

### Using pip

Alternatively, you can install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Dependencies

- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scikit-image**: Image processing (morphology, filters, measurements)
- **tifffile**: Reading TIFF image files
- **ipython**: Interactive development
- **stochastic**: Fractional Gaussian noise generation
- **scipy**: Scientific computing (dependency for scikit-image)

## Available Implementations

The project provides four different implementations with varying performance characteristics:

### 1. Original Implementations
- **`confinement_walks_binary_morphology`**: Sequential processing, returns positions and labels
- **`vectorized_confinement`**: Vectorized processing, returns positions only

### 2. Optimized Implementations ⚡
- **`confinement_walks_binary_morphology_optimized`**: Optimized sequential processing with 1.3-2.0x speedup
- **`vectorized_confinement_optimized`**: Optimized vectorized processing with 1.0-1.1x speedup

### Performance Comparison
| Implementation | Small (n=10) | Medium (n=25) | Large (n=50) | Extra Large (n=100) |
|----------------|---------------|---------------|--------------|---------------------|
| Original Slow | 0.015s | 0.081s | 0.296s | 0.590s |
| **Optimized Slow** | **0.012s** | **0.048s** | **0.170s** | **0.295s** |
| Original Fast | 0.008s | 0.022s | 0.051s | 0.070s |
| **Optimized Fast** | **0.007s** | **0.020s** | **0.045s** | **0.067s** |

**Overall speedup: Up to 8.78x faster** (Original Slow → Optimized Fast)

## Quick Start

### Basic Usage

Run the main example with all implementations:
```bash
python two_regime_walk_example.py
```

### Using Optimized Functions

```python
import numpy as np
from two_regime_walk_example import (
    confinement_walks_binary_morphology_optimized,
    vectorized_confinement_optimized,
    choose_tx_start_locations
)

# Create or load your binary mask
mask = np.zeros((200, 200), dtype=bool)
mask[50:150, 50:150] = True  # Simple square region

# Choose starting positions
starts = choose_tx_start_locations(mask, n_particles=50)

# Set simulation parameters
params = {
    'mask': mask,
    'start_points': starts,
    'alphas': [1.5, 0.8],        # Anomalous exponents [gland, stroma]
    'Ds': [0.35, 0.35],          # Diffusion coefficients
    'T': 1000,                   # Time points
    'trans': 0.5,                # Transition probability
    'deltaT': 18,                # Time step (seconds)
    'L': 500                     # Boundary size
}

# Use optimized slow implementation (returns positions + labels)
data, labels = confinement_walks_binary_morphology_optimized(**params)
print(f"Trajectory shape: {data.shape}")      # (T, N, 2)
print(f"Labels shape: {labels.shape}")        # (T, N, 3)

# Use optimized fast implementation (positions only, fastest)
data_fast = vectorized_confinement_optimized(**params)
print(f"Fast trajectory shape: {data_fast.shape}")  # (T, N, 2)
```

## Performance Benchmarking

### Run Comprehensive Benchmarks

```bash
# Full performance analysis with multiple problem sizes
python comprehensive_benchmark.py

# Basic functionality and timing tests
python test_optimizations.py
```

### Custom Timing Example

```python
from timeit import timeit
from functools import partial

# Create partial functions for timing
slow_func = partial(confinement_walks_binary_morphology_optimized, **params)
fast_func = partial(vectorized_confinement_optimized, **params)

# Time the functions
iterations = 3
slow_time = timeit(slow_func, number=iterations) / iterations
fast_time = timeit(fast_func, number=iterations) / iterations

print(f"Optimized slow: {slow_time:.4f}s")
print(f"Optimized fast: {fast_time:.4f}s")
print(f"Speedup: {slow_time/fast_time:.2f}x")
```

### Expected Performance Results

For the standard test case (n=50 particles, t=1000 time points):
- **Original slow**: ~0.30s
- **Optimized slow**: ~0.17s (1.74x faster)
- **Original fast**: ~0.05s
- **Optimized fast**: ~0.045s (1.12x faster)
- **Best overall**: 6.56x improvement

## Key Optimization Features

### Optimized Slow Implementation
- ✅ **Single mask labeling** (eliminates O(N) redundancy)
- ✅ **Pre-computed displacements** for all particles
- ✅ **Vectorized boundary checking** and compartment lookup
- ✅ **Memory optimization** (float32 vs float64)
- ✅ **Reduced function call overhead**

### Optimized Fast Implementation
- ✅ **Efficient displacement generation** with better memory layout
- ✅ **In-place operations** for boundary handling
- ✅ **Streamlined reflection logic** (limited attempts)
- ✅ **Optimized memory access patterns**
- ✅ **Better cache efficiency**

## Function Parameters

All implementations accept the same parameters:

```python
def simulation_function(
    mask,                    # Binary mask defining regions
    start_points,           # Starting positions (N x 2 array)
    T=200,                  # Number of time points
    Ds=[1.0, 0.1],         # Diffusion coefficients [regime0, regime1]
    alphas=[1.0, 1.0],     # Anomalous exponents [regime0, regime1]
    S=1,                   # Scaling factor
    trans=0.1,             # Boundary transmittance (0-1)
    deltaT=1,              # Time step
    L=None,                # Boundary size (auto if None)
    **kwargs
):
```

### Return Values
- **Slow implementations**: `(positions, labels)` tuple
  - `positions`: Shape (T, N, 2) - particle trajectories
  - `labels`: Shape (T, N, 3) - [alpha, D, state] for each particle/time
- **Fast implementations**: `positions` only
  - `positions`: Shape (T, N, 2) - particle trajectories

## Choosing the Right Implementation

| Use Case | Recommended Implementation | Reason |
|----------|---------------------------|---------|
| **Production/Large datasets** | `vectorized_confinement_optimized` | Fastest performance |
| **Research/Analysis** | `confinement_walks_binary_morphology_optimized` | Includes detailed labels |
| **Debugging** | `confinement_walks_binary_morphology` | Most readable code |
| **Legacy compatibility** | `vectorized_confinement` | Original fast implementation |

## Advanced Usage

### Memory Considerations
```python
# For large simulations, monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / 1024 / 1024  # MB

# Run simulation
data = vectorized_confinement_optimized(**params)

memory_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory used: {memory_after - memory_before:.1f} MB")
```

### Batch Processing
```python
# Process multiple masks efficiently
masks = [mask1, mask2, mask3]  # List of different tissue regions
results = []

for i, mask in enumerate(masks):
    print(f"Processing mask {i+1}/{len(masks)}")
    starts = choose_tx_start_locations(mask, 50)
    params['mask'] = mask
    params['start_points'] = starts

    data = vectorized_confinement_optimized(**params)
    results.append(data)

print(f"Processed {len(results)} masks")
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**:
   ```python
   # Reduce precision or problem size
   params['data_type'] = np.float32  # Use less memory
   # Or reduce T or number of particles
   ```

2. **Slow performance**:
   ```python
   # Ensure you're using optimized versions
   from two_regime_walk_example import vectorized_confinement_optimized
   # Not the original slow implementation
   ```

3. **Shape mismatches**:
   ```python
   # Ensure start_points is 2D array
   starts = np.array(starts).reshape(-1, 2)
   ```

## Files in This Project

- `two_regime_walk_example.py` - Main implementation with all functions
- `comprehensive_benchmark.py` - Performance benchmarking script
- `test_optimizations.py` - Basic functionality tests
- `OPTIMIZATION_REPORT.md` - Detailed optimization analysis
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment specification

## Original Features

- **Two-regime diffusion**: Different diffusion coefficients and anomalous exponents for different regions
- **Boundary conditions**: Particles can transition between regions with configurable transmittance
- **Vectorized implementation**: Fast generation of multiple trajectories
- **Image-based masks**: Use real tissue images to define diffusion regions
- **Fractional Brownian motion**: Supports anomalous diffusion with configurable Hurst exponents


## Optimization prompts

1. "I would like to optimize this code. In particular I currently have two implementations that both perform the same simulation: confinement_walks_binary_morphology and vectorized_confinement. vectorized_confinement is more efficient. I would like you to examine and optimize first one function and then the other to see what the best performance achievable is. I would like timing reports for the different implementations you develop and to compare them to the original functions."

2. "Update the README with details on the new optimized functions and show how to run and time them."
