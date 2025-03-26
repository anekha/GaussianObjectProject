import os
import json
import numpy as np
import trimesh
from PIL import Image
from geometry import depth_to_points, create_triangles
from scipy.spatial.transform import Rotation as R


def load_poses(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    poses = []
    for frame in data:
        print(f"Frame data: {frame}")  # Debug print
        if 'img_name' not in frame:
            raise ValueError(f"Frame is missing 'img_name': {frame}")
        poses.append({
            'position': np.array(frame['position']),
            'rotation': np.array(frame['rotation']),
            'img_name': frame['img_name']  # Ensure 'img_name' is included
        })

    return poses



def validate_pose(pose):
    """
    Validate and normalize the pose data.
    Args:
        pose (dict): Pose containing 'position' and 'rotation'.
    Returns:
        Tuple of position and rotation matrix.
    """
    position = np.array(pose['position'])
    rotation = np.array(pose['rotation'])

    # Ensure the rotation matrix is orthonormal
    if not np.allclose(np.dot(rotation, rotation.T), np.eye(3), atol=1e-6):
        raise ValueError("Rotation matrix is not orthonormal")

    # Ensure the determinant of the rotation matrix is 1
    if not np.isclose(np.linalg.det(rotation), 1.0):
        raise ValueError("Rotation matrix determinant is not 1")

    return position, rotation


def transform_points(points, position, rotation_matrix):
    """
    Transform 3D points using position and rotation matrix.
    Args:
        points (np.ndarray): 3D points (N, 3).
        position (np.ndarray): Translation vector (3,).
        rotation_matrix (np.ndarray): Rotation matrix (3, 3).
    Returns:
        np.ndarray: Transformed 3D points (N, 3).
    """
    # Apply rotation and translation
    transformed_points = rotation_matrix @ points.T + position.reshape(3, 1)
    return transformed_points.T  # Transpose back to (N, 3)


def generate_3d_mesh(image_folder, depths, poses):
    all_verts = []
    all_faces = []
    all_colors = []
    vert_offset = 0

    for idx, (depth, pose) in enumerate(zip(depths, poses)):
        print(f"Processing Image {idx}: depth.shape = {depth.shape}")

        image_path = os.path.join(image_folder, pose['img_name'])
        image = np.asarray(Image.open(image_path))

        points = depth_to_points(depth[None])
        points = transform_points(points[0], pose['position'], pose['rotation'])

        triangles = create_triangles(depth.shape[0], depth.shape[1])
        colors = image.reshape(-1, 3)

        assert np.max(triangles + vert_offset) < vert_offset + points.size // 3, (
            f"Face indices exceed vertex count at Image {idx}."
        )

        all_verts.append(points.reshape(-1, 3))
        all_faces.append(triangles + vert_offset)
        all_colors.append(colors)

        print(f"Image {idx}: Max face index = {np.max(triangles + vert_offset)}")
        vert_offset += points.shape[0] * points.shape[1]

    all_verts = np.vstack(all_verts)
    all_faces = np.vstack(all_faces)
    all_colors = np.vstack(all_colors)

    print(f"Total vertices: {len(all_verts)}, Total faces: {len(all_faces)}")

    return trimesh.Trimesh(vertices=all_verts, faces=all_faces, vertex_colors=all_colors)




# Main execution
if __name__ == "__main__":
    from PIL import Image

    image_folder = "/Users/anekha/Documents/GitHub/GaussianObjectProject/GaussianObject/data/jewelry/ring_one_40frames/images"  # Replace with actual path
    depth_file = "/Users/anekha/Documents/GitHub/GaussianObjectProject/GaussianObject/data/jewelry/ring_one_40frames/dust3r_depth_40.npy"  # Replace with actual path
    json_file = "/Users/anekha/Documents/GitHub/GaussianObjectProject/GaussianObject/data/jewelry/ring_one_40frames/dust3r_40.json"  # Replace with actual path

    # Load images
    image_filenames = [f"{str(i).zfill(3)}.png" for i in range(1, 41)]  # '001.png' to '040.png'
    images = [Image.open(f"{image_folder}/{filename}") for filename in image_filenames]

    # Load depths
    depths = np.load(depth_file)  # Shape: (40, H, W)

    # Load poses
    poses = load_poses(json_file)

    # Generate the 3D mesh
    mesh = generate_3d_mesh(image_folder, depths, poses)

    # Save the mesh as GLB
    output_file = "/Users/anekha/Documents/GitHub/GaussianObjectProject/GaussianObject/data/jewelry/ring_one_40frames/mesh.glb"  # Replace with actual path
    mesh.export(output_file)
    print(f"Mesh saved to {output_file}")