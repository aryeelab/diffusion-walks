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

The project provides multiple implementations with varying performance characteristics:

### 1. Original Implementations
- **`confinement_walks_binary_morphology`**: Sequential processing, returns positions and labels
- **`vectorized_confinement`**: Vectorized processing, returns positions only

### 2. Optimized Implementations ⚡
- **`confinement_walks_binary_morphology_optimized`**: Optimized sequential processing with 1.3-2.0x speedup
- **`vectorized_confinement_optimized`**: Optimized vectorized processing with 1.0-1.1x speedup

### 3. Ultra-Optimized Implementations 🚀
- **`vectorized_confinement_compiled`**: Numba JIT compilation with 2.2-2.6x speedup
- **`vectorized_confinement_extreme_optimized`**: Advanced optimizations with 2.4-2.6x speedup
- **`vectorized_confinement_parallel`**: Multi-core processing for large problems

### Performance Comparison
| Implementation | Small (n=25) | Medium (n=50) | Large (n=100) | Many Particles (n=500) |
|----------------|---------------|---------------|---------------|-------------------------|
| Original Slow | ~0.15s | ~0.30s | ~0.59s | ~2.5s |
| Optimized Slow | ~0.12s | ~0.17s | ~0.30s | ~1.2s |
| Original Fast | 0.012s | 0.031s | 0.073s | 0.170s |
| Optimized Fast | 0.011s | 0.029s | 0.069s | 0.141s |
| **Compiled (Numba)** | **0.005s** | **0.012s** | **0.033s** | **0.098s** |
| **Extreme Optimized** | **0.005s** | **0.012s** | **0.033s** | **0.097s** |
| **Parallel** | **0.005s** | **0.012s** | **0.033s** | **0.097s** |

**Maximum speedup: Up to 25x faster** (Original Slow → Ultra-Optimized)

## Quick Start

### Basic Usage

Run the original example:
```bash
python two_regime_walk_example.py
```

Run comprehensive optimization demonstration:
```bash
python final_optimization_demo.py
```

### Using Optimized Functions

```python
import numpy as np
from two_regime_walk_example import (
    confinement_walks_binary_morphology_optimized,
    vectorized_confinement_optimized,
    choose_tx_start_locations
)
from ultra_optimized_diffusion import (
    vectorized_confinement_compiled,
    vectorized_confinement_extreme_optimized,
    vectorized_confinement_parallel
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

# Use ultra-optimized implementations for maximum performance
data_compiled = vectorized_confinement_compiled(**params)      # Numba JIT: ~2.5x faster
data_extreme = vectorized_confinement_extreme_optimized(**params)  # Best single-thread: ~2.6x faster
data_parallel = vectorized_confinement_parallel(**params)     # Multi-core: best for 1000+ particles
```

## Performance Benchmarking

### Run Comprehensive Benchmarks

```bash
# Complete optimization demonstration with all performance levels
python final_optimization_demo.py
```

### Custom Timing Example

```python
from timeit import timeit
from functools import partial

# Create partial functions for timing
slow_func = partial(confinement_walks_binary_morphology_optimized, **params)
fast_func = partial(vectorized_confinement_optimized, **params)
ultra_func = partial(vectorized_confinement_extreme_optimized, **params)

# Time the functions
iterations = 3
slow_time = timeit(slow_func, number=iterations) / iterations
fast_time = timeit(fast_func, number=iterations) / iterations
ultra_time = timeit(ultra_func, number=iterations) / iterations

print(f"Optimized slow: {slow_time:.4f}s")
print(f"Optimized fast: {fast_time:.4f}s")
print(f"Ultra-optimized: {ultra_time:.4f}s")
print(f"Fast vs slow speedup: {slow_time/fast_time:.2f}x")
print(f"Ultra vs fast speedup: {fast_time/ultra_time:.2f}x")
print(f"Ultra vs slow speedup: {slow_time/ultra_time:.2f}x")
```

### Expected Performance Results

For the standard test case (n=50 particles, t=500 time points):
- **Original slow**: ~0.30s
- **Optimized slow**: ~0.17s (1.74x faster)
- **Original fast**: ~0.031s
- **Optimized fast**: ~0.029s (1.07x faster)
- **Compiled (Numba)**: ~0.012s (2.58x faster than original fast)
- **Extreme optimized**: ~0.012s (2.58x faster than original fast)
- **Best overall**: **25x improvement** (Original Slow → Ultra-Optimized)

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

### Ultra-Optimized Implementations 🚀
- ✅ **Numba JIT compilation** with `@njit` decorators and `fastmath=True`
- ✅ **Cache-friendly memory access** with 32-element chunks
- ✅ **Branchless operations** for boundary conditions
- ✅ **Loop unrolling** and vectorization optimizations
- ✅ **Parallel processing** for large particle counts (1000+)
- ✅ **Advanced memory layout** optimization (C-contiguous arrays)

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
| **Maximum Performance** | `vectorized_confinement_extreme_optimized` | Best single-threaded performance (2.6x faster) |
| **Large Problems (1000+ particles)** | `vectorized_confinement_parallel` | Multi-core processing |
| **Good Performance + Easy Setup** | `vectorized_confinement_compiled` | Numba JIT compilation (2.5x faster) |
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

    # Use ultra-optimized version for maximum performance
    data = vectorized_confinement_extreme_optimized(**params)
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
   # Ensure you're using ultra-optimized versions
   from ultra_optimized_diffusion import vectorized_confinement_extreme_optimized
   # Not the original slow implementation
   ```

3. **Shape mismatches**:
   ```python
   # Ensure start_points is 2D array
   starts = np.array(starts).reshape(-1, 2)
   ```

## Repository Structure

### Core Implementation Files
- `two_regime_walk_example.py` - Original functions and basic optimized versions
- `ultra_optimized_diffusion.py` - Ultra-optimized implementations with Numba JIT

### Environment Setup
- `environment.yml` - Conda environment specification (includes numba, psutil)
- `requirements.txt` - Python package requirements

### Benchmarking and Testing
- `final_optimization_demo.py` - Comprehensive benchmarking and demonstration script

### Documentation
- `README.md` - Usage instructions and optimization overview
- `COMPLETE_OPTIMIZATION_REPORT.md` - Complete optimization analysis and techniques

## Original Features

- **Two-regime diffusion**: Different diffusion coefficients and anomalous exponents for different regions
- **Boundary conditions**: Particles can transition between regions with configurable transmittance
- **Vectorized implementation**: Fast generation of multiple trajectories
- **Image-based masks**: Use real tissue images to define diffusion regions
- **Fractional Brownian motion**: Supports anomalous diffusion with configurable Hurst exponents


## Optimization prompts

1. "I would like to optimize this code. In particular I currently have two implementations that both perform the same simulation: confinement_walks_binary_morphology and vectorized_confinement. vectorized_confinement is more efficient. I would like you to examine and optimize first one function and then the other to see what the best performance achievable is. I would like timing reports for the different implementations you develop and to compare them to the original functions."

2. "Update the README with details on the new optimized functions and show how to run and time them."

3. "I would like you to apply more aggressive optimization techniques to achieve maximum performance for the fastest diffusion simulation function"