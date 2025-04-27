import os
import numpy as np
import pydicom
from typing import Tuple
from scipy.ndimage import rotate
import time
import psutil


def load_ct_slices(ct_dir: str) -> Tuple[np.ndarray, dict, dict]:
    """Load CT slices with single-acquisition check, header parsing, and timing."""
    # Initialize process for memory measurement
    process = psutil.Process(os.getpid())

    t0 = time.perf_counter()
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT directory '{ct_dir}' does not exist.")

    # Gather DICOM files
    filepaths = [os.path.join(ct_dir, f)
                 for f in os.listdir(ct_dir)
                 if f.lower().endswith('.dcm')]
    if not filepaths:
        raise FileNotFoundError(f"No DICOM files found in '{ct_dir}'.")

    # Read datasets
    datasets = []
    for fp in filepaths:
        try:
            ds = pydicom.dcmread(fp)
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: could not read '{fp}': {e}")
    if not datasets:
        raise ValueError("No valid DICOM slices loaded.")
    t1 = time.perf_counter()

    # Verify single acquisition
    acqs = []
    for ds in datasets:
        num = getattr(ds, 'AcquisitionNumber', None)
        if num is None:
            acqs.append(-1)
        else:
            acqs.append(int(num))
    unique_acqs = set(acqs)
    if len(unique_acqs) > 1:
        # Keep only the most common acquisition
        common = max(unique_acqs, key=lambda x: acqs.count(x))
        print(f"Warning: multiple AcquisitionNumbers found {unique_acqs}, filtering to {common}.")
        datasets = [ds for ds, a in zip(datasets, acqs) if a == common]
    raw = getattr(datasets[0], 'AcquisitionNumber', None)
    if raw is None:
        acquisition_number = -1
    else:
        acquisition_number = int(raw)

    # Sort by InstanceNumber for cross-check; fallback to z-position
    try:
        datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    except Exception:
        datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]
    t2 = time.perf_counter()

    # Stack and convert to HU
    arrs = [ds.pixel_array for ds in datasets]
    volume = np.stack(arrs, axis=0).astype(np.float32)
    slope = float(getattr(datasets[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(datasets[0], 'RescaleIntercept', 0.0))
    volume = volume * slope + intercept
    t3 = time.perf_counter()

    # Compute spacing
    row_spacing, col_spacing = map(float, datasets[0].PixelSpacing)
    try:
        slice_thickness = abs(positions[1] - positions[0])
    except Exception:
        slice_thickness = float(getattr(datasets[0], 'SliceThickness', 1.0))
    t4 = time.perf_counter()

    metadata = {
        'spacing': (row_spacing, col_spacing, slice_thickness),
        'positions': positions,
        'acquisition_number': acquisition_number
    }
    timings = {
        'read_dicom_time': t1 - t0,
        'filter_sort_time': t2 - t1,
        'hu_conversion_time': t3 - t2,
        'spacing_calc_time': t4 - t3,
        'memory_usage': process.memory_info().rss
    }
    return volume, metadata, timings


def load_segmentation(seg_path: str, ct_positions: list) -> np.ndarray:
    """Load segmentation labels for multi-segment ROI as a single volume."""
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Segmentation file '{seg_path}' not found.")

    seg_ds = pydicom.dcmread(seg_path)
    n_frames = int(getattr(seg_ds, 'NumberOfFrames', 1))
    frames = seg_ds.PerFrameFunctionalGroupsSequence

    # Prepare output label volume: 0 background, 1-4 segments
    num_slices = len(ct_positions)
    rows, cols = seg_ds.Rows, seg_ds.Columns
    label_vol = np.zeros((num_slices, rows, cols), dtype=np.uint8)

    # Extract pixel data
    data = seg_ds.pixel_array.reshape(n_frames, rows, cols)

    # Map each frame to slice and segment
    ct_pos_arr = np.array(ct_positions)
    for k, f in enumerate(frames):
        pos = f.PlanePositionSequence[0].ImagePositionPatient
        seg_z = float(pos[2])
        idx = int(np.argmin(np.abs(ct_pos_arr - seg_z)))

        # Get segment number
        sid = f.SegmentIdentificationSequence[0]
        seg_num = int(sid.ReferencedSegmentNumber)

        # Assign label
        mask2d = data[k].astype(bool)
        label_vol[idx][mask2d] = seg_num

    return label_vol


def create_mip(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.max(volume, axis=axis)


def overlay_mask(ct_image: np.ndarray, mask_labels: np.ndarray) -> np.ndarray:
    """Overlay multi-label mask with distinct colors."""
    # Normalize CT
    norm = (ct_image - ct_image.min()) / (ct_image.max() - ct_image.min())
    rgb = (np.stack([norm]*3, axis=-1) * 255).astype(np.uint8)

    # Color map for labels
    cmap = {
        1: [255, 0, 0],    # red
        2: [0, 255, 0],    # green
        3: [0, 0, 255],    # blue
        4: [255, 255, 0]   # yellow
    }
    for label, color in cmap.items():
        rgb[mask_labels == label] = color
    return rgb


def rotate_volume(volume: np.ndarray, angle: float, axis: str = 'y') -> np.ndarray:
    if axis == 'y':
        return rotate(volume, angle=angle, axes=(2, 0), reshape=False, order=1)
    elif axis == 'x':
        return rotate(volume, angle=angle, axes=(1, 2), reshape=False, order=1)
    elif axis == 'z':
        return rotate(volume, angle=angle, axes=(1, 0), reshape=False, order=1)
    else:
        raise ValueError(f"Invalid rotation axis '{axis}'. Use 'x', 'y' or 'z'.")