# Environment Setup Guide

## Clone Conda Environment

To clone the `ta_env` conda environment and name it `sf2026`, run:

```bash
conda create --name sf2026 --clone ta_env
```

## Alternative Methods

### Method 1: Export and Create (More control)
```bash
# Export the environment specification
conda env export -n ta_env > ta_env.yml

# Create new environment from the exported file
conda env create -n sf2026 -f ta_env.yml
```

### Method 2: Using conda-pack (Preserves exact versions)
```bash
# Activate source environment
conda activate ta_env

# Install conda-pack if not already installed
conda install conda-pack

# Pack the environment
conda pack -n ta_env -o ta_env.tar.gz

# Unpack to new location
mkdir -p sf2026_env
cd sf2026_env
tar -xzf ../ta_env.tar.gz

# Activate the packed environment
source bin/activate
```

### After Creating the Environment

1. **Activate the new environment:**
   ```bash
   conda activate sf2026
   ```

2. **Verify the installation:**
   ```bash
   conda list
   ```

3. **Deactivate when done:**
   ```bash
   conda deactivate
   ```

## Troubleshooting

- If the clone command fails, check that `ta_env` exists:
  ```bash
  conda env list
  ```

- If you get permission errors, try:
  ```bash
  conda create --name sf2026 --clone ta_env --offline
  ```

