import modal
import os
import subprocess
import sys

# Reference your volume
project_volume = modal.Volume.from_name("gaussianobjectvolume")

# Define the Modal app with secrets
huggingface_secret = modal.Secret.from_name("my-huggingface-secret")  # Define the secret for Hugging Face

# Image
def gaussian_image():

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
            add_python="3.11",  # Adds Python 3.11
        )

        .copy_local_dir(
        "/Users/anekha/Documents/GitHub/GaussianObjectProject/setup",
        "/root/setup"  # Copy the local setup directory to this path in the image
    )

        .copy_local_dir(
        "/Users/anekha/Documents/GitHub/GaussianObjectProject/GaussianObject/submodules",
        "/root/GaussianObject/submodules"  # Copy the local submodules directory to this path in the image
    )

        .run_commands(
            [
                "chmod +x /root/setup/setup_environment.sh",  # Make the script executable
                "/bin/bash /root/setup/setup_environment.sh",  # Run the script

                # Verify CUDA toolkit installation
                "echo 'Checking CUDA toolkit installation...'",
                "nvcc --version",  # Check CUDA version
                "echo 'CUDA path:'",
                "which nvcc",  # Print the path to the nvcc compiler

                # Verify Python 3.11 installation
                "echo 'Checking Python version...'",
                "python3.11 --version || echo 'Python 3.11 is not installed'",

                # Verify GCC and G++ versions
                "echo 'Checking GCC and G++ versions...'",
                "gcc --version",  # Check GCC version
                "g++ --version",  # Check G++ version

                # Check PyTorch availability and version
                "echo 'Checking PyTorch installation...'",
                "python3.11 -c 'import torch; print(\"PyTorch version:\", torch.__version__)' || echo 'PyTorch is not installed or unavailable in Python 3.11'",

                #Setup the submodules
                "chmod +x /root/setup/setup_submodules.sh",
                "/bin/bash /root/setup/setup_submodules.sh",

                #Build CroCo module
                # "python /root/GaussianObject/submodules/croco/models/curope/setup.py build_ext --inplace"
            ]
        )
        # Set PYTHONPATH after building submodules
        #.env({"PYTHONPATH": "/__modal/volumes/vo-9RdT2YQ2K9J3auEo2kXAyu/GaussianObjectProject/GaussianObject/submodules"})
    )
    return image
    
# Create the Modal app with secrets
app = modal.App("gaussian_object", secrets=[huggingface_secret])

# Function to execute your main logic
@app.function(volumes={"/my_vol": project_volume},
              gpu="A100",
              timeout=10000,
              secrets=[modal.Secret.from_name("my-huggingface-secret")],
              image=gaussian_image())

def main_function():
    try:
        # Define the submodules paths in the volume
        submodule_paths = [
            "/my_vol/GaussianObjectProject/GaussianObject/submodules/diff-gaussian-rasterization",
            "/my_vol/GaussianObjectProject/GaussianObject/submodules/diff-gaussian-rasterization-w-pose/build/lib.linux-x86_64-cpython-311",
            "/my_vol/GaussianObjectProject/GaussianObject/submodules/simple-knn",
            "/my_vol/GaussianObjectProject/GaussianObject/submodules/pytorch3d",
            "/my_vol/GaussianObjectProject/GaussianObject/submodules/minLoRA",
            "/my_vol/GaussianObjectProject/GaussianObject/submodules/CLIP"
        ]

        # Update the PYTHONPATH to include the submodule paths in the volume
        import sys
        sys.path.extend(submodule_paths)

        # You can also copy files from the volume to the submodules directory in the container if needed
        import shutil
        submodules_dir = "/root/GaussianObject/submodules"
        for path in submodule_paths:
            dest = os.path.join(submodules_dir, os.path.basename(path))
            if os.path.exists(path):
                shutil.copytree(path, dest, dirs_exist_ok=True)

        # Now you should be able to import the modules from the volume
        import diff_gaussian_rasterization
        import diff_gaussian_rasterization_w_pose
        import simple_knn
        import pytorch3d
        import minlora
        import clip

        print("Modules imported successfully!")

    except Exception as e:
        print("Error:", str(e))
        return f"Error: {str(e)}"


    # Add each submodule's path based on the mounted volume at /my_vol
    # sys.path.extend([
    #     "/my_vol/GaussianObjectProject/GaussianObject/submodules/diff-gaussian-rasterization",
    #     "/my_vol/GaussianObjectProject/GaussianObject/submodules/diff-gaussian-rasterization-w-pose/build/lib.linux-x86_64-cpython-311",
    #     "/my_vol/GaussianObjectProject/GaussianObject/submodules/simple-knn",
    #     "/my_vol/GaussianObjectProject/GaussianObject/submodules/pytorch3d",
    #     "/my_vol/GaussianObjectProject/GaussianObject/submodules/minLoRA",
    #     "/my_vol/GaussianObjectProject/GaussianObject/submodules/CLIP"
    # ])

    #
    # packages = {
    #     "camtools": "camtools",
    #     "einops": "einops",
    #     "huggingface-hub": "huggingface_hub",
    #     "ipykernel": "ipykernel",
    #     "lpips": "lpips",
    #     "ninja": "ninja",
    #     "numpy": "numpy",
    #     "omegaconf": "omegaconf",
    #     "open3d": "open3d",
    #     "opencv-python-headless": "cv2",
    #     "plyfile": "plyfile",
    #     "pytorch-lightning": "pytorch_lightning",
    #     "PyYAML": "yaml",
    #     "roma": "roma",
    #     "scipy": "scipy",
    #     "tensorboard": "tensorboard",
    #     "tensorboard-data-server": "tensorboard_data_server",
    #     "torchmetrics": "torchmetrics",
    #     "tqdm": "tqdm",
    #     "transformers": "transformers",
    #     "trimesh": "trimesh",
    # }
    # for pkg_name, import_name in packages.items():
    #     try:
    #         pkg = __import__(import_name)
    #         version = getattr(pkg, "__version__", "No version attribute")
    #         print(f"{pkg_name} version:", version)
    #     except ImportError as e:
    #         print(f"Failed to import {pkg_name}: {e}")
    #
    # packages = ["roma", "plyfile", "lpips"]
    # for pkg in packages:
    #     result = subprocess.run(["pip", "show", pkg], capture_output=True, text=True)
    #     if result.returncode == 0:
    #         print(f"{pkg} is installed:\n{result.stdout}")
    #     else:
    #         print(f"{pkg} is not installed.")
    #
    # import torch, torchvision, torchaudio;
    # print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}, torchaudio: {torchaudio.__version__}')
    #
    # submodules_to_check = [
    #     "diff_gaussian_rasterization",
    #     "diff_gaussian_rasterization_w_pose",
    #     "simple_knn",
    #     "pytorch3d",
    #     "minlora",
    #     "clip",
    #     "segment_anything",
    # ]
    #
    # for submodule in submodules_to_check:
    #     try:
    #         module = __import__(submodule)
    #         print(f"{submodule} is installed and importable.")
    #     except ImportError as e:
    #         print(f"{submodule} is missing or could not be imported: {e}")
    #pre processing
    try:
        # Define the full path to the data within the mounted volume
        data_path = "/my_vol/GaussianObjectProject/GaussianObject/data/jewelry/ring_1"

        # Run the CLI command with the updated path
        result = subprocess.run(
            ["python", "/my_vol/GaussianObjectProject/GaussianObject/pred_poses.py", "-s", data_path, "--sparse_num", "12"],
            check=True,  # Raise an error if the command fails
            capture_output=True,  # Capture the output
            text=True  # Decode output to string
        )

        # Print the output for visibility and debugging
        print("Command output:", result.stdout)
        return result.stdout  # Return the output for further use if needed

    except subprocess.CalledProcessError as e:
        print("Error executing command:", e.stderr)
        return f"Command failed with error: {e.stderr}"

    return 10 #dummy

      
    
@app.local_entrypoint()
def main():
    main_function.remote()


### versions needed
# pip install \
#     torchmetrics==1.5.2 \ ok
#     trimesh==4.5.2 \ ok
#     plyfile==1.1 \ ok
#     scipy==1.14.1 \ ok
#     ninja==1.11.1.1 \ ok
#     camtools==0.1.5 \ ok
#     einops==0.8.0 \ ok
#     lpips==0.1.4 \ ok
#     tensorboard==2.18.0 \ ok
#     tqdm==4.67.0 \ ok
#     transformers==4.46.2 \
#     omegaconf==2.3.0 \ ok
#     open-clip-torch==2.29.0 \
#     open3d==0.18.0 \ ok
#     opencv-python-headless==4.10.0.84 \
#     "numpy<2" \ 1.26.4
#     Pillow==10.2.0 \
#     pytorch-lightning==2.4.0 \
#     PyYAML==6.0.2 \ ok
#     ipykernel==6.29.5 \ ok
#     roma==1.5.1 \ ok
#     huggingface-hub==0.26.2 ok
#PyTorch: 2.1.0+cu118, torchvision: 0.16.0+cu118, torchaudio: 2.1.0+cu118
