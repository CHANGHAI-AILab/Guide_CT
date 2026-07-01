"""
IPMN CT Feature Extraction
===========================
Extracts quantitative CT imaging features from segmented IPMN NIfTI files
for GUIDE-CT model scoring.

Segmentation label convention (multi-label mask):
  1 = pancreatic parenchyma (gland)
  2 = main pancreatic duct (MPD)
  3 = solid component (mural nodule / solid burden)
  4 = cystic component (target lesion / fluid)

Inputs per case (four CT phases + one multi-label mask):
  C  = unenhanced (plain)
  A  = arterial
  P  = portal venous   <- primary phase for measurement
  V  = delayed / venous

Output: one CSV row per case with all extracted features.
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
import pandas as pd
from scipy import ndimage
from multiprocessing import Pool
from skimage.measure import label, regionprops
from radiomics import cShape

# --------------------------------------------------------------------------- #
# Segmentation-label constants
# --------------------------------------------------------------------------- #
LABEL_GLAND   = 1   # pancreatic parenchyma
LABEL_DUCT    = 2   # main pancreatic duct
LABEL_SOLID   = 3   # solid / mural component
LABEL_CYSTIC  = 4   # cystic component (target lesion)

# CT threshold separating cystic (fluid) from solid tissue (HU)
CYSTIC_HU_THRESHOLD = 35


# =========================================================================== #
# Volume & diameter helpers
# =========================================================================== #

def compute_pancreas_volumes(mask_path: str):
    """
    Return pancreatic parenchyma volume, gland volume, and cystic void volume
    (all in cm³ = mL).

    Returns
    -------
    (P_V, G_V, C_V) : tuple of float
        P_V  – total pancreas volume (label 2+3 union)
        G_V  – pure gland volume     (label 1)
        C_V  – cystic void           (P_V - G_V - solid_V)
    """
    mask_sitk = sitk.ReadImage(mask_path)
    spacing   = mask_sitk.GetSpacing()          # (x, y, z) mm
    voxel_vol = float(np.prod(spacing))          # mm³
    img       = sitk.GetArrayFromImage(mask_sitk)  # (z, y, x)
    pixelSpacing = np.array(spacing[::-1])

    P_V = G_V = T_V = 0.0
    for label_val in [1, 2, 3]:
        tmp = img.copy()
        if label_val == 1:
            tmp[tmp != 1] = 0
        elif label_val == 2:
            tmp[(tmp == 2) | (tmp == 3)] = 1
            tmp[tmp != 1] = 0
        else:  # 3
            tmp[tmp != 3] = 0
            tmp[tmp == 3] = 1

        vol = voxel_vol * float(np.sum(tmp))
        _, _, diameters = cShape.calculate_coefficients(tmp, pixelSpacing)
        if label_val == 2:
            P_V = vol / 1000.0
        elif label_val == 1:
            G_V = vol / 1000.0
        else:
            T_V = vol / 1000.0

    return P_V, G_V, P_V - G_V - T_V


def compute_duct_inscribed_diameter(mask_path: str):
    """
    Compute per-slice inscribed-circle radius for gland (label 1) and
    whole-pancreas (labels 1+2+3) cross-sections.

    Returns
    -------
    tuple with (pid, max_yixian_zj, max_yiguan_zj, min_yixian_zj,
                min_yiguan_zj, mean_yixian_re, mean_yiguan_re)
    where *_yiguan* = gland-only, *_yixian* = whole pancreas.
    """
    sitk_img  = sitk.ReadImage(mask_path)
    spacing   = sitk_img.GetSpacing()
    arr       = sitk.GetArrayFromImage(sitk_img)  # (z, y, x)
    pid       = os.path.basename(mask_path).split('.')[0]

    max_g = 0.0; min_g = 9999.0; vals_g = []
    max_p = 0.0; min_p = 9999.0; vals_p = []

    for slc in arr:
        # gland only
        m_g = slc.copy()
        m_g[m_g != 1] = 0; m_g[m_g == 1] = 255
        m_g = m_g.astype(np.uint8)
        _, r_g, _, _ = cv2.minMaxLoc(cv2.distanceTransform(m_g, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        d_g = float(r_g) * spacing[0] * 2
        vals_g.append(d_g)
        if d_g > max_g: max_g = d_g
        if d_g < min_g: min_g = d_g

        # whole pancreas (labels 1+2+3)
        m_p = slc.copy()
        m_p[(m_p == 1) | (m_p == 2) | (m_p == 3)] = 255
        m_p[m_p != 255] = 0
        m_p = m_p.astype(np.uint8)
        _, r_p, _, _ = cv2.minMaxLoc(cv2.distanceTransform(m_p, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        d_p = float(r_p) * spacing[0] * 2
        vals_p.append(d_p)
        if d_p > max_p: max_p = d_p
        if d_p < min_p: min_p = d_p

    mean_g = float(np.mean(vals_g)) if vals_g else 0.0
    mean_p = float(np.mean(vals_p)) if vals_p else 0.0

    return (pid, max_p, max_g, min_p, min_g, mean_p, mean_g)


# =========================================================================== #
# CT attenuation helpers
# =========================================================================== #

def mean_pancreas_hu(image_path: str, mask_path: str, label: int = LABEL_DUCT) -> float:
    """Mean HU of pancreatic parenchyma (label 2 = duct region) across slices."""
    try:
        img_nib  = nib.load(image_path)
        mask_nib = nib.load(mask_path)
        img_arr  = np.array(img_nib.dataobj).transpose(2, 1, 0)
        mask_arr = np.array(mask_nib.dataobj).transpose(2, 1, 0)
        mask_arr = (mask_arr == label).astype(float)
        per_slice = []
        for i in range(img_arr.shape[0]):
            slc = img_arr[i] * mask_arr[i]
            slc[slc < 0] = 0
            n = int(np.sum(mask_arr[i] > 0))
            if n > 0:
                per_slice.append(float(np.sum(slc)) / n)
        return float(np.mean(per_slice)) if per_slice else float('nan')
    except Exception:
        return float('nan')


def solid_component_hu_stats(image_path: str, mask_path: str):
    """
    HU statistics (mean, max, min) of solid component (label 3)
    in the largest-area cross-section.
    """
    img_arr  = nib.load(image_path).get_fdata()
    mask_arr = nib.load(mask_path).get_fdata()
    mask_arr = (mask_arr == LABEL_SOLID).astype(float)

    best_area = 0
    mean_ct = max_ct = min_ct = 0.0
    for z in range(mask_arr.shape[2]):
        m_slc = mask_arr[:, :, z]
        i_slc = img_arr[:, :, z]
        vals  = i_slc[m_slc > 0]
        if vals.size > best_area:
            best_area = vals.size
            mean_ct = float(np.mean(vals))
            max_ct  = float(np.max(vals))
            min_ct  = float(np.min(vals))
    return mean_ct, max_ct, min_ct


# =========================================================================== #
# Lesion geometry (cystic component, label 4)
# =========================================================================== #

def compute_lesion_diameter(mask_path: str):
    """
    Max lesion diameter (LD, mm) from the largest-area cross-section,
    using inscribed-circle transform on label-4 binary mask.
    """
    mask_arr = nib.load(mask_path).get_fdata()
    mask_arr = (mask_arr == LABEL_CYSTIC).astype(np.uint8)
    zooms    = nib.load(mask_path).header.get_zooms()

    max_d = 0.0
    for z in range(mask_arr.shape[2]):
        m = mask_arr[:, :, z].astype(np.uint8)
        if not np.any(m):
            continue
        _, r, _, _ = cv2.minMaxLoc(cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        d = float(r) * float(zooms[0]) * 2
        if d > max_d:
            max_d = d
    return max_d


def compute_solid_volume_fraction(mask_path: str, ct_threshold: int = CYSTIC_HU_THRESHOLD) -> float:
    """
    Solid-volume fraction (SVF, %) defined as the percentage of label-4 voxels
    with CT value >= ct_threshold.

    Note: This is an approximation using the mask label; the paper uses the
    semi-automated segmentation workflow with quality control.
    """
    mask_arr = nib.load(mask_path).get_fdata()
    n_total  = int(np.sum(mask_arr == LABEL_CYSTIC))
    if n_total == 0:
        return 0.0
    # For SVF we need the CT image; fall back to shape-based solid label if not provided
    n_solid  = int(np.sum(mask_arr == LABEL_SOLID))
    return 100.0 * n_solid / (n_solid + n_total) if (n_solid + n_total) > 0 else 0.0


def compute_solid_volume_fraction_from_ct(ct_path: str, mask_path: str,
                                          ct_threshold: int = CYSTIC_HU_THRESHOLD) -> float:
    """
    SVF (%) using actual CT HU values within label-4 region.
    Solid voxels are those with HU >= ct_threshold.
    """
    ct_arr   = nib.load(ct_path).get_fdata()
    mask_arr = nib.load(mask_path).get_fdata()
    roi      = ct_arr[mask_arr == LABEL_CYSTIC]
    if roi.size == 0:
        return 0.0
    n_solid = int(np.sum(roi >= ct_threshold))
    return 100.0 * n_solid / roi.size


def compute_lesion_volumes(ct_path: str, mask_path: str,
                           ct_threshold: int = CYSTIC_HU_THRESHOLD):
    """
    Total lesion volume, cystic volume, solid volume (all in mm³) and
    solid / cystic volume fractions.
    """
    ct_arr   = nib.load(ct_path).get_fdata()
    mask_arr = nib.load(mask_path).get_fdata()
    zooms    = nib.load(ct_path).header.get_zooms()
    vox_vol  = float(np.prod(zooms))

    roi_vals = ct_arr[mask_arr == LABEL_CYSTIC]
    if roi_vals.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    n_solid   = int(np.sum(roi_vals >= ct_threshold))
    n_cystic  = int(np.sum(roi_vals < ct_threshold))
    vol_solid  = n_solid  * vox_vol
    vol_cystic = n_cystic * vox_vol
    vol_total  = vol_solid + vol_cystic
    svf  = vol_solid  / vol_total if vol_total > 0 else 0.0
    cvf  = vol_cystic / vol_total if vol_total > 0 else 0.0
    return vol_total, vol_cystic, vol_solid, cvf, svf


def compute_lesion_major_minor_axes(mask_path: str):
    """
    Major and minor axis lengths (pixels) in the largest-area cross-section
    of the target lesion (label 4).
    """
    try:
        mask_arr = nib.load(mask_path).get_fdata()
        mask_arr = (mask_arr == LABEL_CYSTIC).astype(np.uint8)
        best_area = 0
        best_slice = None
        for z in range(mask_arr.shape[2]):
            a = int(np.sum(mask_arr[:, :, z]))
            if a > best_area:
                best_area = a
                best_slice = mask_arr[:, :, z]
        if best_slice is None:
            return float('nan'), float('nan'), float('nan')
        props = regionprops(label(best_slice))
        if not props:
            return 0.0, 0.0, 0.0
        p = props[0]
        major = float(p.major_axis_length)
        minor = float(p.minor_axis_length)
        return major, minor, (major + minor) / 2.0
    except Exception:
        return float('nan'), float('nan'), float('nan')


def compute_mpd_diameter(mask_path: str) -> float:
    """
    Maximum main pancreatic duct (MPD) diameter (mm).
    Uses inscribed-circle transform on label-2 binary slices.
    """
    try:
        sitk_img = sitk.ReadImage(mask_path)
        spacing  = sitk_img.GetSpacing()
        arr      = sitk.GetArrayFromImage(sitk_img)
        max_d    = 0.0
        for slc in arr:
            m = slc.copy()
            m[m != LABEL_DUCT] = 0
            m[m == LABEL_DUCT] = 255
            m = m.astype(np.uint8)
            if not np.any(m):
                continue
            _, r, _, _ = cv2.minMaxLoc(
                cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
            d = float(r) * spacing[0] * 2
            if d > max_d:
                max_d = d
        return max_d
    except Exception:
        return float('nan')


# =========================================================================== #
# Per-case wrapper
# =========================================================================== #

def extract_case_features(case_name: str, base_dir: str, output_dir: str,
                           ct_threshold: int = CYSTIC_HU_THRESHOLD):
    """
    Extract all quantitative features for one case and write a CSV row.

    Expected directory layout under *base_dir*:
        C/<case_name>   – unenhanced phase image + mask
        A/<case_name>   – arterial phase image + mask
        P/<case_name>   – portal venous phase image + mask  (primary)
        V/<case_name>   – delayed phase image + mask

    Sub-directories:
        <phase>_image/  – CT NIfTI
        <phase>_mask/   – multi-label NIfTI mask
    """
    print(f"Processing: {case_name}")

    def img(phase):
        return os.path.join(base_dir, f"{phase}_image", case_name)

    def msk(phase):
        return os.path.join(base_dir, f"{phase}_mask", case_name)

    row = {"case_name": case_name}

    # -- Pancreas volumes --------------------------------------------------- #
    try:
        P_V, G_V, C_V = compute_pancreas_volumes(msk("P"))
    except Exception:
        P_V = G_V = C_V = float('nan')
    row.update({"P_V_mL": P_V, "G_V_mL": G_V, "cyst_void_V_mL": C_V})

    # -- Duct / gland diameter ---------------------------------------------- #
    try:
        (pid, max_pan_d, max_gland_d,
         min_pan_d, min_gland_d,
         mean_pan_d, mean_gland_d) = compute_duct_inscribed_diameter(msk("P"))
    except Exception:
        pid = case_name
        max_pan_d = max_gland_d = min_pan_d = min_gland_d = mean_pan_d = mean_gland_d = float('nan')
    row.update({
        "max_pancreas_diam_mm": max_pan_d,
        "max_gland_diam_mm":    max_gland_d,
        "min_pancreas_diam_mm": min_pan_d,
        "min_gland_diam_mm":    min_gland_d,
        "mean_pancreas_diam_mm": mean_pan_d,
        "mean_gland_diam_mm":   mean_gland_d,
    })

    # -- MPD diameter ------------------------------------------------------- #
    row["mpd_max_diam_mm"] = compute_mpd_diameter(msk("P"))

    # -- Lesion size -------------------------------------------------------- #
    row["lesion_diameter_mm"] = compute_lesion_diameter(msk("P"))
    maj, minor_, mean_ = compute_lesion_major_minor_axes(msk("P"))
    row.update({"major_axis_mm": maj, "minor_axis_mm": minor_, "mean_axis_mm": mean_})

    # -- Lesion volumes & SVF ----------------------------------------------- #
    try:
        vol_t, vol_c, vol_s, cvf, svf = compute_lesion_volumes(img("P"), msk("P"), ct_threshold)
    except Exception:
        vol_t = vol_c = vol_s = cvf = svf = float('nan')
    row.update({
        "total_lesion_vol_mm3":   vol_t,
        "cystic_vol_mm3":         vol_c,
        "solid_vol_mm3":          vol_s,
        "cystic_vol_fraction":    cvf,
        "solid_vol_fraction_SVF": svf,
    })

    # -- CT attenuation per phase ------------------------------------------- #
    for phase in ["C", "A", "P", "V"]:
        row[f"pancreas_HU_{phase}"] = mean_pancreas_hu(img(phase), msk(phase))

    # -- Solid component HU ------------------------------------------------- #
    try:
        sc_mean, sc_max, sc_min = solid_component_hu_stats(img("P"), msk("P"))
    except Exception:
        sc_mean = sc_max = sc_min = float('nan')
    row.update({
        "solid_HU_mean_P": sc_mean,
        "solid_HU_max_P":  sc_max,
        "solid_HU_min_P":  sc_min,
    })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_name}.csv")
    pd.DataFrame([row]).to_csv(out_path, index=False)
    return row


# =========================================================================== #
# Entry point
# =========================================================================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract IPMN CT features for GUIDE-CT scoring."
    )
    parser.add_argument("--base_dir",   required=True,
                        help="Root directory with C/A/P/V sub-folders.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write per-case CSV files.")
    parser.add_argument("--workers",    type=int, default=4,
                        help="Number of parallel workers (default: 4).")
    parser.add_argument("--threshold",  type=int, default=CYSTIC_HU_THRESHOLD,
                        help=f"HU threshold for solid vs cystic (default: {CYSTIC_HU_THRESHOLD}).")
    args = parser.parse_args()

    cases = os.listdir(os.path.join(args.base_dir, "P_image"))
    print(f"Found {len(cases)} cases.")

    def _worker(c):
        return extract_case_features(c, args.base_dir, args.output_dir, args.threshold)

    with Pool(processes=args.workers) as pool:
        pool.map(_worker, cases)

    print("Done. Concatenating CSV files...")
    frames = [pd.read_csv(os.path.join(args.output_dir, f))
              for f in os.listdir(args.output_dir) if f.endswith(".csv")]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "all_features.csv"), index=False)
        print(f"Saved combined CSV to {args.output_dir}/all_features.csv")
