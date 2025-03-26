#!/bin/bash

# -------------------------------
# 7. Set the Correct Submodules Path
# -------------------------------
# Set the absolute path to the submodules directory
SUBMODULES_PATH="/root/GaussianObject/submodules"

# Verify that the submodules directory exists
if [ ! -d "$SUBMODULES_PATH" ]; then
    echo "Error: Submodules directory not found at $SUBMODULES_PATH. Please check the path."
    exit 1
fi

echo "Submodules directory found at $SUBMODULES_PATH."

# -------------------------------
# 8. Install Submodules
# -------------------------------
echo "Installing submodules..."

# Install each submodule if the directory exists
for submodule in diff-gaussian-rasterization diff-gaussian-rasterization-w-pose simple-knn pytorch3d minLoRA CLIP segment-anything; do
    if [ -d "$SUBMODULES_PATH/$submodule" ]; then
        echo "Installing $submodule..."
        pip install -v "$SUBMODULES_PATH/$submodule"
    else
        echo "Warning: Submodule $submodule directory not found at $SUBMODULES_PATH, skipping installation."
    fi
done

# -------------------------------
# 9. Build Submodules in Editable Mode
# -------------------------------
echo "Building submodules in editable mode..."

for submodule in diff-gaussian-rasterization diff-gaussian-rasterization-w-pose simple-knn pytorch3d minLoRA CLIP segment-anything; do
    if [ -d "$SUBMODULES_PATH/$submodule" ]; then
        echo "Building $submodule in editable mode..."
        pip install -e "$SUBMODULES_PATH/$submodule"
    else
        echo "Warning: Submodule $submodule directory not found at $SUBMODULES_PATH, skipping editable mode build."
    fi
done
