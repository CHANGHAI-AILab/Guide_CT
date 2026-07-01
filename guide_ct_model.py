"""
GUIDE-CT Locked Model
=====================
Guideline-informed Unified Imaging for Ductal and Solid-Burden Evaluation with CT.

Reference
---------
[manuscript under review] CT-Centered Risk Stratification for
Guideline-Concerning Pancreatic Intraductal Papillary Mucinous Neoplasm.

Locked model equation (Table S4 of Supplementary Appendix)
-----------------------------------------------------------
Linear predictor:
  eta = -0.6223
        + 0.5204 * zLD
        + 0.5791 * zMPD
        + 1.4395 * zSVF
        + 1.4760 * zALARM

Predicted probability:
  p = 1 / (1 + exp(-eta))

Standardization constants (training-cohort means and SDs):
  LD    : mean = 28.0360, SD = 6.5604, median_impute = 27.8122
  MPD   : mean =  4.9938, SD = 1.6543, median_impute =  4.9304
  SVF   : mean = 35.0125, SD = 8.9536, median_impute = 34.8263
  ALARM : mean =  1.9713, SD = 0.9565, median_impute =  1.9744

Operating thresholds (locked in training):
  Balanced (primary)          : p >= 0.3925
  Safety-prioritized (high-Sn): p >= 0.1482

Predictor definitions
---------------------
LD    – lesion diameter (mm), largest cross-sectional measurement of the
        cystic component on portal venous phase CT.
MPD   – main pancreatic duct diameter (mm), maximum measurement.
SVF   – solid-volume fraction (%), percentage of the target lesion volume
        with CT attenuation >= 35 HU on portal venous phase.
ALARM – non-size alarm score; ordinal sum of binary guideline features
        (obstructive jaundice, thickened/enhancing cyst wall, abrupt
        duct-caliber change with distal atrophy, acute pancreatitis,
        CA19-9 elevation, new-onset/worsening diabetes);
        rescaled to training-range denominator when components are missing.

Outcome
-------
HGD/IC – high-grade dysplasia or invasive carcinoma (positive class).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# =========================================================================== #
# Locked constants – do NOT modify
# =========================================================================== #

_INTERCEPT = -0.6223

_COEF = {
    "zLD":    0.5204,
    "zMPD":   0.5791,
    "zSVF":   1.4395,
    "zALARM": 1.4760,
}

_STANDARDIZATION = {
    # variable: (mean, sd, median_impute_value)
    "LD":    (28.0360, 6.5604, 27.8122),
    "MPD":   ( 4.9938, 1.6543,  4.9304),
    "SVF":   (35.0125, 8.9536, 34.8263),
    "ALARM": ( 1.9713, 0.9565,  1.9744),
}

THRESHOLD_BALANCED           = 0.3925  # primary operating point (Table S4)
THRESHOLD_SAFETY_PRIORITIZED = 0.1482  # high-sensitivity operating point (Table S4)


# =========================================================================== #
# Standardization helper
# =========================================================================== #

def _standardize(value: Optional[float], variable: str) -> float:
    """
    Apply training-cohort standardization.
    Missing values (None / NaN) are replaced by the training median before
    standardization, matching the locked model protocol.
    """
    mean, sd, median_impute = _STANDARDIZATION[variable]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        value = median_impute
    return (float(value) - mean) / sd


# =========================================================================== #
# ALARM score helper
# =========================================================================== #

def compute_alarm_score(
    obstructive_jaundice:        Optional[int] = None,
    thickened_enhancing_wall:    Optional[int] = None,
    abrupt_duct_caliber_change:  Optional[int] = None,
    acute_pancreatitis:          Optional[int] = None,
    ca199_elevation:             Optional[int] = None,
    new_onset_diabetes:          Optional[int] = None,
) -> float:
    """
    Compute the non-size ALARM score.

    Each component is binary (0 = absent, 1 = present).
    Pass ``None`` for components not collected at your center; they will be
    excluded and the score rescaled to the training-range denominator (6).

    Returns
    -------
    float
        ALARM score rescaled to the full 6-component denominator.
    """
    components = {
        "obstructive_jaundice":       obstructive_jaundice,
        "thickened_enhancing_wall":   thickened_enhancing_wall,
        "abrupt_duct_caliber_change": abrupt_duct_caliber_change,
        "acute_pancreatitis":         acute_pancreatitis,
        "ca199_elevation":            ca199_elevation,
        "new_onset_diabetes":         new_onset_diabetes,
    }
    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return float('nan')
    raw_sum   = sum(available.values())
    n_avail   = len(available)
    # rescale to full denominator
    return raw_sum * 6.0 / n_avail


# =========================================================================== #
# Main scoring function
# =========================================================================== #

@dataclass
class GuideCTResult:
    """Output of ``score_case``."""
    LD:    float
    MPD:   float
    SVF:   float
    ALARM: float

    zLD:    float = field(init=False)
    zMPD:   float = field(init=False)
    zSVF:   float = field(init=False)
    zALARM: float = field(init=False)

    eta:         float = field(init=False)
    probability: float = field(init=False)

    # recommendations using locked thresholds
    recommend_surgery_balanced:          bool = field(init=False)
    recommend_surgery_safety_prioritized: bool = field(init=False)

    def __post_init__(self):
        self.zLD    = _standardize(self.LD,    "LD")
        self.zMPD   = _standardize(self.MPD,   "MPD")
        self.zSVF   = _standardize(self.SVF,   "SVF")
        self.zALARM = _standardize(self.ALARM, "ALARM")

        self.eta = (
            _INTERCEPT
            + _COEF["zLD"]    * self.zLD
            + _COEF["zMPD"]   * self.zMPD
            + _COEF["zSVF"]   * self.zSVF
            + _COEF["zALARM"] * self.zALARM
        )
        self.probability = 1.0 / (1.0 + math.exp(-self.eta))

        self.recommend_surgery_balanced = (
            self.probability >= THRESHOLD_BALANCED
        )
        self.recommend_surgery_safety_prioritized = (
            self.probability >= THRESHOLD_SAFETY_PRIORITIZED
        )

    def to_dict(self) -> dict:
        return {
            "LD_mm":         self.LD,
            "MPD_mm":        self.MPD,
            "SVF_pct":       self.SVF,
            "ALARM":         self.ALARM,
            "zLD":           self.zLD,
            "zMPD":          self.zMPD,
            "zSVF":          self.zSVF,
            "zALARM":        self.zALARM,
            "eta":           self.eta,
            "probability":   self.probability,
            "high_risk_balanced":           self.recommend_surgery_balanced,
            "high_risk_safety_prioritized": self.recommend_surgery_safety_prioritized,
        }


def score_case(
    LD:    Optional[float] = None,
    MPD:   Optional[float] = None,
    SVF:   Optional[float] = None,
    ALARM: Optional[float] = None,
) -> GuideCTResult:
    """
    Score a single IPMN case with the locked GUIDE-CT model.

    Parameters
    ----------
    LD : float or None
        Lesion diameter (mm). Missing values are median-imputed.
    MPD : float or None
        Main pancreatic duct diameter (mm). Missing values are median-imputed.
    SVF : float or None
        Solid-volume fraction (%). Missing values are median-imputed.
    ALARM : float or None
        Non-size alarm score. Missing values are median-imputed.

    Returns
    -------
    GuideCTResult

    Examples
    --------
    >>> r = score_case(LD=35.0, MPD=6.5, SVF=42.0, ALARM=2.0)
    >>> round(r.probability, 3)
    0.835
    """
    return GuideCTResult(LD=LD, MPD=MPD, SVF=SVF, ALARM=ALARM)


def score_dataframe(df: pd.DataFrame,
                    ld_col:    str = "LD",
                    mpd_col:   str = "MPD",
                    svf_col:   str = "SVF",
                    alarm_col: str = "ALARM") -> pd.DataFrame:
    """
    Score a DataFrame of cases.

    Adds columns: ``guide_ct_prob``, ``guide_ct_high_risk_balanced``,
    ``guide_ct_high_risk_safety_prioritized``.
    """
    out = df.copy()
    results = []
    for _, row in df.iterrows():
        r = score_case(
            LD=row.get(ld_col),
            MPD=row.get(mpd_col),
            SVF=row.get(svf_col),
            ALARM=row.get(alarm_col),
        )
        results.append(r.to_dict())

    res_df = pd.DataFrame(results)
    out["guide_ct_prob"]                       = res_df["probability"].values
    out["guide_ct_high_risk_balanced"]          = res_df["high_risk_balanced"].values
    out["guide_ct_high_risk_safety_prioritized"]= res_df["high_risk_safety_prioritized"].values
    return out


# =========================================================================== #
# CLI entry point
# =========================================================================== #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score IPMN cases with the locked GUIDE-CT model."
    )
    parser.add_argument("--input",  required=True,
                        help="CSV file with columns LD, MPD, SVF, ALARM.")
    parser.add_argument("--output", required=True,
                        help="Output CSV path.")
    parser.add_argument("--ld_col",    default="LD")
    parser.add_argument("--mpd_col",   default="MPD")
    parser.add_argument("--svf_col",   default="SVF")
    parser.add_argument("--alarm_col", default="ALARM")
    args = parser.parse_args()

    df_in  = pd.read_csv(args.input)
    df_out = score_dataframe(df_in,
                              ld_col=args.ld_col,
                              mpd_col=args.mpd_col,
                              svf_col=args.svf_col,
                              alarm_col=args.alarm_col)
    df_out.to_csv(args.output, index=False)
    print(f"Scored {len(df_out)} cases → {args.output}")

    # summary
    n_high_bal  = int(df_out["guide_ct_high_risk_balanced"].sum())
    n_high_safe = int(df_out["guide_ct_high_risk_safety_prioritized"].sum())
    print(f"  High-risk (balanced threshold {THRESHOLD_BALANCED}):           "
          f"{n_high_bal} / {len(df_out)}")
    print(f"  High-risk (safety-prioritized threshold {THRESHOLD_SAFETY_PRIORITIZED}): "
          f"{n_high_safe} / {len(df_out)}")
