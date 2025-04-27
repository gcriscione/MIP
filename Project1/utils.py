import os
import numpy as np
import pydicom
from typing import Tuple
from scipy.ndimage import rotate

def load_ct_slices(ct_dir: str) -> Tuple[np.ndarray, dict]:
    """Load CT slices with error handling."""
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT directory '{ct_dir}' does not exist.")

    # Gather and read all DICOM files
    filepaths = [os.path.join(ct_dir, f)
                 for f in os.listdir(ct_dir)
                 if f.lower().endswith('.dcm')]
    if not filepaths:
        raise FileNotFoundError(f"No DICOM files found in '{ct_dir}'.")

    datasets = []
    for fp in filepaths:
        try:
            ds = pydicom.dcmread(fp)
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: could not read '{fp}': {e}")
    if not datasets:
        raise ValueError("No valid DICOM slices loaded.")

    # Sort by z-position
    datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]

    # Stack pixel arrays and convert to float32 HU
    volume = np.stack([ds.pixel_array for ds in datasets], axis=0).astype(np.float32)
    slope = float(getattr(datasets[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(datasets[0], 'RescaleIntercept', 0.0))
    volume = volume * slope + intercept

    # Compute spacing
    row_spacing, col_spacing = map(float, datasets[0].PixelSpacing)
    try:
        slice_thickness = abs(positions[1] - positions[0])
    except Exception:
        slice_thickness = float(getattr(datasets[0], 'SliceThickness', 1.0))

    metadata = {
        'spacing': (row_spacing, col_spacing, slice_thickness),
        'positions': positions
    }
    return volume, metadata

def load_segmentation(seg_path: str, ct_positions: list) -> Tuple[np.ndarray, list]:
    """Load segmentation with error handling."""
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Segmentation file '{seg_path}' not found.")

    seg_ds = pydicom.dcmread(seg_path)
    n_frames = int(getattr(seg_ds, 'NumberOfFrames', 1))
    frames = seg_ds.PerFrameFunctionalGroupsSequence
    if len(frames) != n_frames:
        raise ValueError("Mismatch between NumberOfFrames and sequence length.")

    # Extract per-frame z positions
    seg_z = []
    for f in frames:
        pos = f.PlanePositionSequence[0].ImagePositionPatient
        seg_z.append(float(pos[2]))

    # Initialize empty mask volume
    num_slices = len(ct_positions)
    rows, cols = seg_ds.Rows, seg_ds.Columns
    mask3d = np.zeros((num_slices, rows, cols), dtype=bool)

    # Raw segmentation data: shape (n_frames, rows, cols)
    data = seg_ds.pixel_array.reshape(n_frames, rows, cols)

    # Map each 2D frame to closest CT slice index
    ct_positions_arr = np.array(ct_positions)
    for k, z in enumerate(seg_z):
        idx = int(np.argmin(np.abs(ct_positions_arr - z)))
        mask3d[idx] = data[k].astype(bool)

    return mask3d

def create_mip(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Create a Maximum Intensity Projection (MIP) along a given axis.
    """
    return np.max(volume, axis=axis)

def overlay_mask(ct_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay a binary mask onto a CT slice (mask in green).
    """
    norm = (ct_image - ct_image.min()) / (ct_image.max() - ct_image.min())
    rgb = (np.stack([norm]*3, axis=-1) * 255).astype(np.uint8)
    rgb[mask] = [0, 255, 0]
    return rgb

def rotate_volume(volume: np.ndarray, angle: float, axis: str = 'y') -> np.ndarray:
    """
    Rotate a 3D volume around a given axis.
    """
    if axis == 'y':
        rotated = rotate(volume, angle=angle, axes=(2, 0), reshape=False, order=1)
    elif axis == 'x':
        rotated = rotate(volume, angle=angle, axes=(1, 2), reshape=False, order=1)
    elif axis == 'z':
        rotated = rotate(volume, angle=angle, axes=(1, 0), reshape=False, order=1)
    else:
        raise ValueError(f"Invalid rotation axis '{axis}'. Use 'x', 'y' or 'z'.")
    
    return rotated