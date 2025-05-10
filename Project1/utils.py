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


def apply_modality_lut(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """
    Applica RescaleSlope e RescaleIntercept per ottenere valori in Hounsfield Units.
    """
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return pixel_array.astype(np.float32) * slope + intercept


def load_ct_slices(ct_dir: str) -> Tuple[np.ndarray, dict, dict]:
    """
    Carica tutte le slice DICOM in una cartella, verifica AcquisitionNumber e spacing,
    filtra slice atipiche, ordina in dual-mode, converte in HU, e ritorna volume + metadata.
    """
    # 1) Trova tutti i file .dcm
    paths = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.lower().endswith('.dcm')]
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {ct_dir}")

    # 2) Leggi tutti i dataset
    datasets = [pydicom.dcmread(p) for p in paths]

    # 3) Filtra slice con shape diversa dalla prima
    ref_shape = datasets[0].pixel_array.shape
    good, bad = [], 0
    for ds in datasets:
        if ds.pixel_array.shape == ref_shape:
            good.append(ds)
        else:
            bad += 1
    if bad:
        logger.warning(f"Scartate {bad} slice con shape diversa da {ref_shape}")
    datasets = good
    if not datasets:
        raise RuntimeError("Nessuna slice valida dopo il filtraggio per shape")

    # 4) Controlla AcquisitionNumber uniforme
    acq_nums = [getattr(ds, 'AcquisitionNumber', None) for ds in datasets]
    if len(set(acq_nums)) != 1:
        raise ValueError(f"Inconsistent AcquisitionNumber found: {set(acq_nums)}")
    acq = acq_nums[0]
    logger.info(f"Single acquisition #{acq} confirmed for all slices")

    # 5) Controlla SpacingBetweenSlices o calcola da Z-positions
    sbss = [getattr(ds, 'SpacingBetweenSlices', None) for ds in datasets]
    if all(sbss):
        if len(set(sbss)) != 1:
            raise ValueError(f"Inconsistent SpacingBetweenSlices: {set(sbss)}")
        slice_sp = float(sbss[0])
        logger.info(f"Using DICOM SpacingBetweenSlices = {slice_sp}")
    else:
        positions = [float(ds.ImagePositionPatient[2]) for ds in datasets]
        diffs = np.diff(sorted(positions))
        if not np.allclose(diffs, diffs[0], atol=1e-3):
            raise ValueError(f"Non-uniform slice spacing detected: {diffs[:5]}…")
        slice_sp = float(diffs[0])
        logger.info(f"Computed uniform slice spacing = {slice_sp} from ImagePositionPatient")

    # 6) Dual-mode ordering: con e senza ImagePositionPatient
    with_pos    = [ds for ds in datasets if hasattr(ds, 'ImagePositionPatient')]
    without_pos = [ds for ds in datasets if not hasattr(ds, 'ImagePositionPatient')]
    with_pos.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    without_pos.sort(key=lambda ds: int(getattr(ds, 'InstanceNumber', 0)))
    datasets = with_pos + without_pos

    # 7) Costruisci volume e converti in HU
    volume = np.stack([apply_modality_lut(ds.pixel_array, ds) for ds in datasets], axis=0)

    # 8) Metadata e timing minimale
    row_sp, col_sp = map(float, datasets[0].PixelSpacing)
    metadata = {
        'acquisition': acq,
        'positions': [float(ds.ImagePositionPatient[2]) for ds in datasets if hasattr(ds, 'ImagePositionPatient')],
        'spacing': (row_sp, col_sp, slice_sp),
        'slice_spacing': slice_sp
    }
    timings = {
        'memory_usage': psutil.Process().memory_info().rss
    }

    logger.info(f"CT volume loaded: shape={volume.shape}, spacing={metadata['spacing']}")
    return volume, metadata, timings


def load_segmentation(seg_path: str, ct_positions: List[float], label_name: str=None) -> np.ndarray:
    """
    Carica una segmentazione DICOM SEG e restituisce un volume etichettato
    (0=background, altrimenti numero di segmento).
    """
    ds = pydicom.dcmread(seg_path)
    n_frames = int(ds.NumberOfFrames)
    rows, cols = int(ds.Rows), int(ds.Columns)
    positions = np.array(ct_positions)
    # Leggi SegmentSequence (se presente)
    seg_map = {}
    if 'SegmentSequence' in ds:
        for item in ds.SegmentSequence:
            seg_map[int(item.SegmentNumber)] = getattr(item, 'SegmentLabel', '')
    default_num = 1 if label_name == 'liver' else 2

    # Prepara array di output
    mask_vol = np.zeros((len(positions), rows, cols), dtype=np.uint8)
    # Per-frame: estrai Z e numero di segmento
    for f, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
        z = float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
        segnum = default_num
        if hasattr(frame, 'SegmentIdentificationSequence'):
            segnum = int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        slice_idx = int(np.argmin(np.abs(positions - z)))
        mask = ds.pixel_array.reshape(n_frames, rows, cols)[f] > 0
        mask_vol[slice_idx][mask] = segnum

    logger.info(f"Loaded SEG: segments={list(seg_map.keys()) or [default_num]}")
    return mask_volImagePositionPatient