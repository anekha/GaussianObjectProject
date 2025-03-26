import numpy as np

def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def depth_to_points(depth, R=None, t=None):

    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]


def create_triangles(h, w, mask=None):
    """
    Creates mesh triangle indices from a given pixel grid size.
    Args:
        h: (int) Height of the grid.
        w: (int) Width of the grid.
        mask: Optional mask to filter valid triangles.
    Returns:
        triangles: 2D numpy array of shape (2 * (h-1) * (w-1), 3).
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x       # Top-left
    tr = y * w + x + 1   # Top-right
    bl = (y + 1) * w + x  # Bottom-left
    br = (y + 1) * w + x + 1  # Bottom-right

    # Create two triangles for each grid cell
    triangles = np.vstack([
        np.stack([tl, bl, tr], axis=-1),  # First triangle
        np.stack([br, tr, bl], axis=-1)   # Second triangle
    ]).reshape(-1, 3)

    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(axis=1)]

    return triangles
