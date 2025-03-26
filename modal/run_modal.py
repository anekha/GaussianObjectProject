import modal
import os
import subprocess

# Reference your volume and secrets
project_volume = modal.Volume.from_name("gaussianobjectvolume")
huggingface_secret = modal.Secret.from_env("HUGGINGFACE_TOKEN")  # This fetches the token from the environment

# Define the Modal app with secrets
app = modal.App("gaussian_object", secrets=[huggingface_secret])

# Define the Modal image with environment setup and submodule installation
def gauss_image():
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",  # CUDA base image
            add_python="3.11"  # Adds Python 3.11
        )

        .copy_local_dir(
            "/Users/anekha/Documents/GitHub/GaussianObjectProject/setup",
            "/root/setup"  # Copy the local setup directory to this path in the image
        )

        .run_commands(
            [
                "chmod +x /root/setup/setup_environment.sh",  # Make the script executable
                "/bin/bash /root/setup/setup_environment.sh",  # Run the script

                # Verify CUDA toolkit installation
                "nvcc --version",  # Check CUDA version
                "which nvcc",  # Print the path to the nvcc compiler
            ]
        )

        .pip_install_private_repos(
            "github.com/ashawkey/diff-gaussian-rasterization@main",
            "github.com/rmurai0610/diff-gaussian-rasterization-w-pose@main",
            "github.com/camenduru/simple-knn@main",
            "github.com/facebookresearch/pytorch3d@main",
            "github.com/cccntu/minLoRA@main",
            "github.com/openai/CLIP@main",
            "github.com/facebookresearch/segment-anything@main",
            git_user="anekha",  # Your GitHub username
            secrets=[modal.Secret.from_name("github-secret")]
        )
    )
    return image

# Function to check imports
@app.function(volumes={"/my_vol": project_volume}, gpu="A100", timeout=10000, image=gauss_image())
def main_function():
    # Attempt to import each submodule and print results
    submodules = [
        "diff_gaussian_rasterization",
        "diff_gaussian_rasterization_w_pose",
        "simple_knn",
        "pytorch3d",
        "minLoRA",
        "clip",
        "segment_anything"
    ]

    for module in submodules:
        try:
            imported_module = __import__(module)
            print(f"{module} is imported successfully.")
        except ImportError as e:
            print(f"Failed to import {module}: {e}")

    return 10  # dummy return value
