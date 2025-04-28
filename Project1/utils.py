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
    Load CT slices from a directory, enforce single-acquisition, sort,
    convert to HU, compute spacing, and collect timing & memory info.
    """
    proc = psutil.Process(os.getpid())
    t0 = time.perf_counter()

    # Gather DICOM files
    if not os.path.isdir(ct_dir):
        logger.error(f"CT directory does not exist: {ct_dir}")
        raise FileNotFoundError(ct_dir)
    paths = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.lower().endswith('.dcm')]
    if not paths:
        logger.error(f"No DICOMs found in: {ct_dir}")
        raise FileNotFoundError(ct_dir)

    # Read datasets
    datasets = []
    for p in paths:
        try:
            datasets.append(pydicom.dcmread(p))
        except Exception as e:
            logger.warning(f"Failed to read {p}: {e!r}")
    if not datasets:
        logger.error("No valid DICOM slices loaded.")
        raise ValueError("Empty dataset list")
    t1 = time.perf_counter()

    # --- Single-acquisition check ---
    acqs = []
    for ds in datasets:
        num = getattr(ds, 'AcquisitionNumber', None)
        if isinstance(num, (int, float, str)) and num is not None:
            try:
                acqs.append(int(num))
            except Exception:
                acqs.append(-1)
                logger.warning(f"Non-integer AcquisitionNumber ({num}) in {getattr(ds, 'filename', '')}")
        else:
            acqs.append(-1)
    uniq = set(acqs)
    if uniq == {-1}:
        logger.warning("No AcquisitionNumber found in any slice.")
        acq_num = -1
    elif len(uniq) > 1:
        common = max(uniq, key=lambda x: acqs.count(x))
        logger.warning(f"Multiple AcquisitionNumbers {uniq}, filtering to {common}.")
        datasets = [ds for ds, a in zip(datasets, acqs) if a == common]
        acq_num = common
    else:
        acq_num = uniq.pop()
        logger.info(f"Using AcquisitionNumber: {acq_num}")
    # --- End acquisition check ---

    # Sort slices by InstanceNumber, fallback to z-position
    try:
        datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    except Exception:
        datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]
    t2 = time.perf_counter()

    # Stack pixel arrays and convert to Hounsfield units
    volume = np.stack([ds.pixel_array for ds in datasets], axis=0).astype(np.float32)
    slope = float(getattr(datasets[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(datasets[0], 'RescaleIntercept', 0.0))
    volume = volume * slope + intercept
    t3 = time.perf_counter()

    # Compute voxel spacing
    row_sp, col_sp = map(float, datasets[0].PixelSpacing)
    try:
        slice_sp = abs(positions[1] - positions[0])
    except Exception:
        slice_sp = float(getattr(datasets[0], 'SliceThickness', 1.0))
    t4 = time.perf_counter()

    metadata = {
        'acquisition_number': acq_num,
        'positions': positions,
        'spacing': (row_sp, col_sp, slice_sp)
    }
    timings = {
        'read_time': t1 - t0,
        'sort_time': t2 - t1,
        'hu_time': t3 - t2,
        'spacing_time': t4 - t3,
        'memory_usage': proc.memory_info().rss
    }

    logger.info(f"Loaded {volume.shape[0]} CT slices; spacing={metadata['spacing']}")
    return volume, metadata, timings


def load_segmentation(seg_path: str, ct_positions: List[float]) -> np.ndarray:
    """
    Load multi-label segmentation (4 ROIs) into a single volume,
    validate with highdicom if available, and log any missing segments.
    """
    if not os.path.isfile(seg_path):
        logger.error(f"Segmentation file not found: {seg_path}")
        raise FileNotFoundError(seg_path)

    # Highdicom validation
    if HIGHDICOM_AVAILABLE:
        try:
            ds = pydicom.dcmread(seg_path)
            seg = Segmentation.from_dataset(ds)
            logger.info(f"highdicom validated {len(seg.SegmentSequence)} segments.")
        except Exception as e:
            logger.error(f"highdicom validation failed: {e!r}")
            raise

    # Fallback parsing with pydicom
    ds = pydicom.dcmread(seg_path)
    n_frames = int(getattr(ds, 'NumberOfFrames', 1))
    frames = ds.PerFrameFunctionalGroupsSequence
    rows, cols = ds.Rows, ds.Columns

    label_vol = np.zeros((len(ct_positions), rows, cols), dtype=np.uint8)
    pixels = ds.pixel_array.reshape(n_frames, rows, cols)
    pos_arr = np.array(ct_positions)

    found = set()
    for i, f in enumerate(frames):
        z = float(f.PlanePositionSequence[0].ImagePositionPatient[2])
        idx = int(np.argmin(np.abs(pos_arr - z)))
        seg_num = int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        found.add(seg_num)
        mask = pixels[i].astype(bool)
        label_vol[idx][mask] = seg_num

    expected = set(range(1, 5))
    missing = expected - found
    if missing:
        logger.warning(f"Found segments {found}, missing: {missing}")
    else:
        logger.info("All 4 segments (1–4) mapped successfully.")

    # after the missing‐check warning/info
    for seg_id in sorted(found):
        count = int((label_vol == seg_id).sum())
        logger.info(f"Segment {seg_id} voxel count: {count}")

    return label_vol


def create_mip(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the maximum-intensity projection along the given axis."""
    return volume.max(axis=axis)


def overlay_mask(ct_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    """
    Overlay multi-label mask onto a CT slice.
    Labels 1–4 will be mapped to distinct colors.
    """
    norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
    rgb = (np.stack([norm]*3, axis=-1) * 255).astype(np.uint8)
    cmap = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 255, 0]}
    for lbl, col in cmap.items():
        rgb[mask_slice == lbl] = col
    return rgb


def rotate_volume(volume: np.ndarray, angle: float, axis: str = 'y') -> np.ndarray:
    """
    Rotate the 3D volume by the specified angle around the given axis.
    Axis can be 'x', 'y', or 'z'.
    """
    axes_map = {'y': (2, 0), 'x': (1, 2), 'z': (1, 0)}
    if axis not in axes_map:
        logger.error(f"Invalid rotation axis: {axis}")
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    return rotate(volume, angle=angle, axes=axes_map[axis], reshape=False, order=1)
