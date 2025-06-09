# Diffusion Walks Project

This project implements two-regime random walk simulations for studying particle diffusion in environments with different diffusion states (e.g., glands vs stroma).

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

## Usage

Run the main example:
```bash
python two_regime_walk_example.py
```

This will:
1. Read and process a TIFF image to create a binary mask
2. Generate random walk trajectories with different diffusion parameters for different regions
3. Compare performance between slow and fast implementation methods
4. Display the results

## Key Features

- **Two-regime diffusion**: Different diffusion coefficients and anomalous exponents for different regions
- **Boundary conditions**: Particles can transition between regions with configurable transmittance
- **Vectorized implementation**: Fast generation of multiple trajectories
- **Image-based masks**: Use real tissue images to define diffusion regions


## Optimization prompt

"I would like to optimize this code. In particular I currently have two implementations that both perform the same simulation: confinement_walks_binary_morphology and vectorized_confinement. vectorized_confinement is more efficient. I would like you to examine and optimize first one function and then the other to see what the best performance achievable is. I would like timing reports for the different implementations you develop and to compare them to the original functions."
