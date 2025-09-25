# scSVAE Cell Cycle Analysis

This repository contains code and notebooks for applying **scSVAE** (single-cell spherical variational autoencoder) to single-cell RNA-seq data, focusing on **cell cycle gene expression**.

## Overview

The workflow demonstrates how to:

- Load and preprocess single-cell RNA-seq data with [Scanpy](https://scanpy.readthedocs.io/).
- Filter genes and normalize counts.
- Restrict analysis to **cell cycle genes**.
- Train a **scSVAE model** on the dataset.
- Explore latent representations with PCA and UMAP.
- Visualize clustering, cell cycle phases, and gene expression patterns.

## Workflow Summary

1. **Setup**
   - Import dependencies: `scanpy`, `scSVAE`, `torch`, `pandas`, `matplotlib`.
   - Load cell cycle gene list (`cc_mouse.csv`).
   - Read single-cell RNA-seq data (`small_rna.h5ad`).

2. **Preprocessing**
   - Filter genes with low counts.
   - Normalize total counts and log-transform.
   - Select highly variable genes.
   - Restrict analysis to known **cell cycle genes**.

3. **Model Training**
   - Configure the AnnData object for scSVAE.
   - Train a spherical VAE with `n_latent = 2`.
   - Track training history and reconstruction error.

4. **Visualization**
   - Latent space embedding.
   - PCA and UMAP plots colored by donor, cell cycle, Leiden clusters.
   - Latent space distribution histograms.
   - Polar transformation (`r, θ`) of latent coordinates.
   - Custom gene heatmaps ordered by θ.

## Example Results

- **UMAP embedding** shows separation of cells by cell cycle phase and donor.
- **Latent space histograms** illustrate the distribution of learned representations.
- **Heatmaps** highlight expression dynamics of cell cycle genes across the latent trajectory.

## Requirements

- Python 3.7+
- [Scanpy](https://scanpy.readthedocs.io/)
- [scvi-tools](https://scvi-tools.org/)
- [PyTorch](https://pytorch.org/)
- [scSVAE](https://github.com/) (custom module, imported via `sys.path`)
- Numpy, Pandas, Matplotlib

Install dependencies in a conda environment:

```bash
conda create -n scvi-env python=3.7
conda activate scvi-env
pip install scanpy torch matplotlib pandas numpy scvi-tools
```


## Credits
- Main Author: 
    - Debojyoti Das ([@BioDebojyoti](https://github.com/BioDebojyoti))   
- Collaborator(s):
    - Johan Henriksson ([@mahogny](https://github.com/mahogny))