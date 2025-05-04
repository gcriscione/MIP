import os
import time
import logging
from typing import Tuple, List

import numpy as np
import pydicom
import psutil
from scipy.ndimage import rotate

# --- Logger setup: console + CSV file ---
LOG_DIR = os.path.join(os.getcwd(), 'output')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(LOG_DIR, 'processing_log.csv')

logger = logging.getLogger('dicom_utils')
logger.setLevel(logging.INFO)

# File handler for CSV logging
file_h = logging.FileHandler(LOG_CSV, mode='w')
file_h.setFormatter(logging.Formatter('%(asctime)s,%(levelname)s,%(name)s,%(message)s'))
logger.addHandler(file_h)

# Console handler
console_h = logging.StreamHandler()
console_h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(console_h)


# --- Try to import highdicom for segmentation validation ---
try:
    from highdicom.seg import Segmentation
    HIGHDICOM_AVAILABLE = True
    logger.info("Using highdicom for segmentation validation.")
except ImportError:
    HIGHDICOM_AVAILABLE = False
    logger.warning("highdicom not available: skipping advanced validation.")


def load_ct_slices(ct_dir: str) -> Tuple[np.ndarray, dict, dict]:
    """
    Load CT volume from a directory, sort slices, convert to HU, compute spacing,
    and return timing/memory info.
    """
    # Collect all DICOM files
    paths = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith('.dcm')]
    if not paths:
        raise FileNotFoundError(f"No DICOMs in {ct_dir}")

    # Read all slices
    datasets = [pydicom.dcmread(p) for p in paths]

    # Sort by InstanceNumber or fallback to Z-position
    try:
        datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    except:
        datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    # Extract Z positions
    positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]

    # Stack pixel arrays and convert to Hounsfield Units
    volume = np.stack([ds.pixel_array for ds in datasets], axis=0).astype(np.float32)
    slope = float(getattr(datasets[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(datasets[0], 'RescaleIntercept', 0.0))
    volume = volume * slope + intercept

    # Compute voxel spacing
    row_sp, col_sp = map(float, datasets[0].PixelSpacing)
    try:
        slice_sp = abs(positions[1] - positions[0])
    except:
        slice_sp = float(getattr(datasets[0], 'SliceThickness', 1.0))

    metadata = {
        'positions': positions,
        'spacing': (row_sp, col_sp, slice_sp)
    }
    timings = {
        'read_time': None,
        'sort_time': None,
        'hu_time': None,
        'spacing_time': None,
        'memory_usage': psutil.Process().memory_info().rss
    }

    return volume, metadata, timings


def load_segmentation(seg_path: str, ct_positions: List[float], label_name: str) -> np.ndarray:
    """
    Load a single ROI mask (liver or tumor) and map its frames onto the CT volume positions.
    If all frames share the same Z, distribute them evenly across the CT slices.
    Returns a binary 3D mask (1=ROI, 0=background).
    """
    ds = pydicom.dcmread(seg_path)
    n_frames = int(ds.NumberOfFrames)
    rows, cols = ds.Rows, ds.Columns

    # Read pixel data and reshape
    pixel_data = ds.pixel_array.reshape(n_frames, rows, cols)
    positions = np.array(ct_positions)

    # Extract Z for each frame
    zs = [float(f.PlanePositionSequence[0].ImagePositionPatient[2])
          for f in ds.PerFrameFunctionalGroupsSequence]
    unique_z = sorted(set(zs))

    # Decide mapping indices
    if len(unique_z) == 1:
        # all frames share the same Z → spread uniformly
        indices = np.linspace(0, len(positions)-1, num=n_frames, dtype=int)
    else:
        # normal nearest-Z mapping
        indices = [int(np.argmin(np.abs(positions - z))) for z in zs]

    # Build mask volume
    mask_vol = np.zeros((len(positions), rows, cols), dtype=np.uint8)
    for frame_idx, dslice_idx in enumerate(indices):
        mask_vol[dslice_idx] |= pixel_data[frame_idx].astype(bool)

    return mask_vol