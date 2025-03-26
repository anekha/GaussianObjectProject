
# GaussianObject Implementation for 3D Object Reconstruction

This repository contains my implementation of GaussianObject for high-quality 3D object reconstruction from multiple images, based on the paper "GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting" presented at SIGGRAPH Asia 2024.

---

## Overview

I’ve implemented the GaussianObject framework, which reconstructs 3D objects using Gaussian splatting. The original paper achieves high-quality 3D reconstruction from four images, even without the need for COLMAP.

This implementation is now extended to handle more than four images (tested with up to 50 images), providing more flexibility for real-world applications. The project uses a Gaussian repair model based on diffusion models to improve 3D object representations by refining missing or corrupted object details.

---

## Key Features

- No COLMAP Required: This implementation does not rely on COLMAP for 3D object reconstruction.
- Flexible Input: Works with 4 to 50 input images, giving flexibility for different types of datasets.
- Gaussian Splatting: Uses Gaussian splatting for 3D object representation and rendering.
- Gaussian Repair: Incorporates a diffusion model-based repair system to refine the 3D model.
- Modal Integration: Deployed using Modal, leveraging GPU infrastructure for efficient computation.

---

## Setup

### Requirements

- Python 3.8 or higher
- The dependencies are listed in `requirements.txt`, install them via:

```bash
pip install -r requirements.txt
```

### Modal Deployment (Recommended)

This implementation is set up for Modal deployment, which allows you to leverage GPU infrastructure to handle the resource-heavy computations involved in 3D object reconstruction.

1. Create a Modal app: You can deploy and run the app remotely using Modal’s GPU infrastructure.
2. Run the code using the following Modal command:

```bash
modal deploy -m <your-entry-point-file>
```

---

## Testing

I have tested this implementation with custom datasets, using a variety of image sets ranging from 4 to 50 images. The results demonstrate that the framework is capable of generating high-quality 3D object reconstructions even with a varied number of input images.

---

## How It Works

1. **Initial Gaussian Representation**: 
   - Uses a visual hull generated from camera parameters and masked images.
   - Optimizes the initial 3D Gaussian representation using a custom loss function.

2. **Gaussian Repair Model**:
   - Trains a repair model using corrupted Gaussian renderings and reference images.
   - The model refines missing or corrupted object details for improved reconstruction.

3. **Optimization**: 
   - Once the repair model is trained, it is used to rectify any views that need adjustments.

---

## Future Work

I plan to continue working on improving the model and integrating it with more advanced techniques. Contributions are welcome!

---

## License

This project is open-source under the MIT License. Feel free to fork or contribute!
