import os
import pandas as pd
import pydicom
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_fill_holes, label
from scipy import stats

# Percorsi dataset
ct_datasets = {
    "10_AP_Ax2.50mm": "dataset/10_AP_Ax2.50mm/",
    "30_EQP_Ax2.50mm": "dataset/30_EQP_Ax2.50mm/"
}
segmentation_files = {
    "10_AP_Ax2.50mm": "dataset/10_AP_Ax2.50mm_ManualROI_Tumor.dcm"
}

def analyze_datasets(ct_datasets, segmentation_files):
    entries   = []
    anomalies = []
    ct_meta   = {}

    # Funzione di supporto per statistiche
    def compute_stats(arr):
        flat = arr.ravel()
        nonzero = flat[flat != 0] if np.any(flat != 0) else flat
        return {
            'min': float(np.min(flat)),
            'max': float(np.max(flat)),
            'mean': float(np.mean(flat)),
            'median': float(np.median(flat)),
            'std': float(np.std(flat)),
            'nonzero_count': int(np.count_nonzero(flat)),
            'nonzero_ratio': float(np.count_nonzero(flat) / flat.size)
        }

    # 1) Analisi CT volumes
    for name, path in ct_datasets.items():
        if not os.path.isdir(path):
            anomalies.append(f"Percorso non valido per CT '{name}': {path}")
            continue

        # Leggi file DICOM nella cartella
        dcm_files = sorted(f for f in os.listdir(path) if f.lower().endswith('.dcm'))
        if not dcm_files:
            anomalies.append(f"Nessun file DICOM in CT '{name}'")
            continue

        # Carica con SimpleITK
        reader = sitk.ImageSeriesReader()
        ids    = reader.GetGDCMSeriesIDs(path)
        files  = reader.GetGDCMSeriesFileNames(path, ids[0])
        reader.SetFileNames(files)
        img    = reader.Execute()
        arr    = sitk.GetArrayFromImage(img).astype(np.float32)  # [z,y,x]
        spacing = img.GetSpacing()  # (X, Y, Z)
        origin  = img.GetOrigin()
        direction = img.GetDirection()
        num_slices = arr.shape[0]

        # Statistiche intensità
        stats_ct = compute_stats(arr)

        # Volume fisico
        voxel_vol_mm3 = np.prod(spacing)
        total_vol_cm3 = arr.size * voxel_vol_mm3 / 1000.0

        # Metadati DICOM dal primo slice
        dc0            = pydicom.dcmread(os.path.join(path, dcm_files[0]))
        slope          = getattr(dc0, 'RescaleSlope', 1.0)
        intercept      = getattr(dc0, 'RescaleIntercept', 0.0)
        orientation    = getattr(dc0, 'ImageOrientationPatient', [])
        window_center  = getattr(dc0, 'WindowCenter', [])
        window_width   = getattr(dc0, 'WindowWidth', [])

        # Salva metadata per segmentazione
        ct_meta[name] = {
            'num_slices': num_slices,
            'spacing': spacing,
            'slope': slope,
            'intercept': intercept,
            'origin': origin,
            'direction': direction
        }

        entries.append({
            'Dataset': name,
            'Tipo': 'CT Volume',
            'Num_Slices': num_slices,
            'Shape': arr.shape,
            'Voxel_Size_mm': spacing,
            'Origin': origin,
            'Direction': direction,
            'Volume_cm3': round(total_vol_cm3, 2),
            **{f"CT_{k}": v for k, v in stats_ct.items()},
            'RescaleSlope': slope,
            'RescaleIntercept': intercept,
            'WindowCenter': window_center,
            'WindowWidth': window_width
        })

    # 2) Analisi Segmentazione
    for ref_name, dcm_path in segmentation_files.items():
        if not os.path.isfile(dcm_path):
            anomalies.append(f"File di segmentazione non trovato: {dcm_path}")
            continue

        ds      = pydicom.dcmread(dcm_path)
        seg_img = sitk.ReadImage(dcm_path)
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.uint8)
        num_frames = getattr(ds, 'NumberOfFrames', seg_arr.shape[0])

        # Indici delle slice a cui si riferiscono i frame
        pfseq = getattr(ds, 'PerFrameFunctionalGroupsSequence', None)
        frame_indices = []
        if pfseq:
            for frame in pfseq:
                idx = int(frame.FrameContentSequence[0].DimensionIndexValues[0]) - 1
                frame_indices.append(idx)
        else:
            anomalies.append(f"Manca PerFrameFunctionalGroupsSequence in {dcm_path}")

        # Ricostruisci maschera 3D (207 slice)
        ref = ct_meta.get(ref_name, {})
        mask = None
        if ref:
            nz, ny, nx = ref['num_slices'], seg_arr.shape[1], seg_arr.shape[2]
            mask = np.zeros((nz, ny, nx), dtype=np.uint8)
            for i, sl in enumerate(frame_indices):
                if 0 <= sl < nz:
                    mask[sl] = seg_arr[i]
                else:
                    anomalies.append(f"Frame index fuori range: {sl}")

        # Statistiche maschera
        stats_mask = compute_stats(mask) if mask is not None else {}
        # Volume tumorale
        mask_vol_cm3 = stats_mask.get('nonzero_count',0) * np.prod(ref.get('spacing', (1,1,1))) / 1000.0
        pct_tumor    = (stats_mask.get('nonzero_count',0) / mask.size * 100) if mask is not None else None

        entries.append({
            'Dataset': ref_name,
            'Tipo': 'Segmentation',
            'Num_Frames': num_frames,
            'Frame_Indices': frame_indices,
            'Mask_Shape': mask.shape if mask is not None else None,
            'Tumor_Voxels': stats_mask.get('nonzero_count'),
            'Tumor_Volume_cm3': round(mask_vol_cm3, 2),
            'Tumor_Volume_pct': round(pct_tumor, 4) if pct_tumor is not None else None,
            **{f"Mask_{k}": v for k, v in stats_mask.items()},
            'SegmentDescription': getattr(ds.SegmentSequence[0], 'SegmentDescription', '')
        })

    # Scrivi anomalie
    print("\n--- ANOMALIE RILEVATE ---")
    if anomalies:
        for a in anomalies:
            print(f"- {a}")
    else:
        print("Nessuna anomalia rilevata.")

    # Crea DataFrame e salva
    df = pd.DataFrame(entries)
    output_csv = "dataset_summary_full_metrics.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nTabella salvata in '{output_csv}'")

    # Genera leggenda
    legend = {
        'Dataset': 'Nome della serie (CT or Segmentation)',
        'Tipo': 'Tipo di dato: CT Volume o Segmentation',
        'Num_Slices': 'Numero di slice del volume CT',
        'Shape': 'Dimensioni dell\'array (z,y,x)',
        'Voxel_Size_mm': 'Spacing mm in (X, Y, Z)',
        'Origin': 'Origine fisica (mm) del volume (GetOrigin)',
        'Direction': 'Coseni di direzione (GetDirection)',
        'Volume_cm3': 'Volume fisico in cm^3 (total_voxels * voxel_vol)',
        'CT_min, CT_max, CT_mean, CT_median, CT_std': 'Statistiche intensità Hounsfield sul CT',
        'CT_nonzero_count': 'Numero voxel non-zero nel CT',
        'CT_nonzero_ratio': 'Percentuale di voxel non-zero sul totale',
        'RescaleSlope/Intercept': 'Parametri DICOM per conversione HU',
        'WindowCenter/WindowWidth': 'Valori di windowing consigliati',
        'Num_Frames': 'Numero di frame nel DICOM Segmentation Storage',
        'Frame_Indices': 'Indici di slice CT corrispondenti ai frame di segmentazione',
        'Mask_Shape': 'Dimensioni dell\'array di maschera (z,y,x)',
        'Tumor_Voxels': 'Numero di voxel label==1 nella maschera',
        'Tumor_Volume_cm3': 'Volume tumorale in cm^3',
        'Tumor_Volume_pct': 'Percentuale del volume tumorale sul volume CT',
        'Mask_min, Mask_max, Mask_mean, Mask_median, Mask_std': 'Statistiche sulla maschera binaria',
        'Mask_nonzero_count': 'Numero di voxel non-zero nella maschera',
        'Mask_nonzero_ratio': 'Percentuale di voxel non-zero nella maschera',
        'SegmentDescription': 'Descrizione organo/tumore nel DICOM SegmentSequence'
    }

    with open("dataset_summary_legend.txt", "w") as f:
        f.write("Legenda dei campi nel CSV di riepilogo:\n\n")
        for field, desc in legend.items():
            f.write(f"{field}: {desc}\n")
    print("Legenda salvata in 'dataset_summary_legend.txt'")

    return df

if __name__ == "__main__":
    analyze_datasets(ct_datasets, segmentation_files)