# Sparse-view CT in 2D

Two notebooks, outputs cleared.

- `gaussian_splatting_2d_astra.ipynb`: 2D Gaussian Splatting reconstruction with a differentiable ASTRA parallel-beam
  projector (custom autograd function), cloning, splitting and pruning of Gaussians, and FBP, SART and SIRT baselines.
- `evidential_unet_start.ipynb`: U-Net and evidential U-Net (Normal-Inverse-Gamma head) trained on Shepp-Logan pairs
  and used to place the initial Gaussians, with the trust signals that fall back to FBP.

They need `numpy matplotlib scikit-image torch scipy astra-toolbox pydicom` and a DICOM slice, which is not included.
