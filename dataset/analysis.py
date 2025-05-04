import os
import pandas as pd
import pydicom
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_fill_holes, label as ndi_label, center_of_mass
from scipy import stats

# Percorsi dataset
ct_datasets = {
    "10_AP_Ax2.50mm": "dataset/10_AP_Ax2.50mm/"
}
segmentation_files = {
    "Liver": "dataset/10_AP_Ax2.50mm_ManualROI_Liver.dcm",
    "Tumor": "dataset/10_AP_Ax2.50mm_ManualROI_Tumor.dcm"
}


def analyze_datasets(ct_datasets, segmentation_files):
    entries = []
    anomalies = []
    ct_meta = {}

    def compute_stats(arr):
        flat = arr.ravel()
        return {
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            'mean': float(np.mean(flat)),
            'median': float(np.median(flat)),
            'std': float(np.std(flat)),
            'sum': float(np.sum(flat))
        }

    # 1) CT Volumes
    for name, path in ct_datasets.items():
        if not os.path.isdir(path):
            anomalies.append(f"Percorso non valido per CT '{name}': {path}")
            continue
        dcm_files = sorted([f for f in os.listdir(path) if f.lower().endswith('.dcm')])
        if not dcm_files:
            anomalies.append(f"Nessun file DICOM in CT '{name}'")
            continue

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(path, series_ids[0]))
        img = reader.Execute()
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        direction = img.GetDirection()
        num_slices, ny, nx = arr.shape

        # Stats generali CT
        stats_ct = compute_stats(arr)
        voxel_vol_mm3 = np.prod(spacing)
        total_vol_cm3 = arr.size * voxel_vol_mm3 / 1000.0

        # Metadati DICOM
        dc0 = pydicom.dcmread(os.path.join(path, dcm_files[0]))
        slope = getattr(dc0, 'RescaleSlope', 1.0)
        intercept = getattr(dc0, 'RescaleIntercept', 0.0)
        wind_center = getattr(dc0, 'WindowCenter', None)
        wind_width = getattr(dc0, 'WindowWidth', None)

        ct_meta[name] = {
            'arr': arr,
            'spacing': spacing,
            'origin': origin,
            'direction': direction
        }

        entries.append({
            'Dataset': name,
            'Tipo': 'CT Volume',
            'Num_Slices': num_slices,
            'Shape': arr.shape,
            'Voxel_Size_mm': spacing,
            'Volume_cm3': round(total_vol_cm3, 2),
            **{f"CT_{k}": v for k, v in stats_ct.items()},
            'RescaleSlope': slope,
            'RescaleIntercept': intercept,
            'WindowCenter': wind_center,
            'WindowWidth': wind_width
        })

    # 2) Segmentazioni (Liver e Tumor)
    for label, dcm_path in segmentation_files.items():
        if not os.path.isfile(dcm_path):
            anomalies.append(f"File di segmentazione non trovato: {dcm_path}")
            continue

        ds = pydicom.dcmread(dcm_path)
        seg_img = sitk.ReadImage(dcm_path)
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.uint8)
        num_frames = getattr(ds, 'NumberOfFrames', seg_arr.shape[0])

        # Estrai indici slice via ReferencedImageSequence se presente
        pfseq = getattr(ds, 'PerFrameFunctionalGroupsSequence', [])
        frame_indices = []
        for frame in pfseq:
            # preferenza ReferencedImageSequence
            refseq = getattr(frame, 'PlanePositionSequence', None) or []
            if refseq and hasattr(refseq[0], 'ReferencedImageSequence'):
                uid = refseq[0].ReferencedImageSequence[0].ReferencedSOPInstanceUID
                # lookup slice index per UID
                # TODO: implement mapping from uid to slice index
                frame_indices.append(None)
            else:
                # fallback a DimensionIndexValues
                idx = int(frame.FrameContentSequence[0].DimensionIndexValues[0]) - 1
                frame_indices.append(idx)

        # Ricostruzione maschera 3D
        ref = ct_meta.get("10_AP_Ax2.50mm", {})
        mask = np.zeros_like(ref.get('arr'), dtype=np.uint8) if ref else None
        if mask is not None:
            for i, sl in enumerate(frame_indices):
                if sl is None or sl < 0 or sl >= mask.shape[0]:
                    anomalies.append(f"Slice index invalido per {label} frame {i}: {sl}")
                else:
                    mask[sl] = seg_arr[i]

        # Fill holes e componenti
        if mask is not None:
            filled = binary_fill_holes(mask)
            components, num_comp = ndi_label(filled)
            centroids = center_of_mass(filled, components, list(range(1, num_comp+1)))
        else:
            filled = mask
            num_comp = 0
            centroids = []

        # Statistiche maschera
        stats_mask = compute_stats(filled) if filled is not None else {}
        # Volume e percentuale
        mm3 = np.prod(ref.get('spacing', (1,1,1)))
        tumor_vox = stats_mask.get('sum', 0)
        tumor_vol_cm3 = tumor_vox * mm3 / 1000.0
        pct_vol = tumor_vox / filled.size * 100 if filled is not None else None

        # Statistiche intensità CT dentro la maschera
        ct_arr = ref.get('arr') if ref else None
        if ct_arr is not None and filled is not None:
            intensities = ct_arr[filled>0]
            stats_in = compute_stats(intensities)
        else:
            stats_in = {}

        entries.append({
            'Dataset': label,
            'Tipo': 'Segmentation',
            'Num_Frames': num_frames,
            'Frame_Indices': frame_indices,
            'Num_Components': num_comp,
            'Component_Centroids_voxel': centroids,
            'Tumor_Voxels': int(tumor_vox),
            'Tumor_Volume_cm3': round(tumor_vol_cm3, 2),
            'Tumor_Volume_pct': round(pct_vol, 2) if pct_vol is not None else None,
            **{f"Mask_{k}": v for k, v in stats_mask.items()},
            **{f"CTinMask_{k}": v for k, v in stats_in.items()},
            'SegmentDescription': getattr(ds.SegmentSequence[0], 'SegmentDescription', '')
        })

    # Stampa anomalie
    print("\n--- ANOMALIE RILEVATE ---")
    for a in anomalies:
        print(f"- {a}")
    if not anomalies:
        print("Nessuna anomalia rilevata.")

    df = pd.DataFrame(entries)
    df.to_csv("dataset_detailed_metrics.csv", index=False)
    print("\nCSV salvato in 'dataset_detailed_metrics.csv'")
    return df

if __name__ == "__main__":
    analyze_datasets(ct_datasets, segmentation_files)