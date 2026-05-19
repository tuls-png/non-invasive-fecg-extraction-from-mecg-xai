"""
xai/echo_master_table.py
Master Explainability Table Generator

Converts raw ECHO clinical explanation logs into a structured pandas DataFrame
with derived clinical and reliability metrics.

Workflow:
  1. Parse raw clinical_explanation text using regex
  2. Extract recording-level statistics and attribution values
  3. Handle missing values (e.g., "N/A" for morphology in NIFECGDB)
  4. Compute derived columns:
     - dominant_cue
     - physiological_plausibility
     - reliability_class
     - clinical_flag
  5. Export to CSV: echo_master_table_{dataset_name}.csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ECHOMasterTableGenerator:
    """
    Parse ECHO clinical explanations and build a structured Master Table.
    """

    # Target columns for the final Master Table
    MASTER_COLUMNS = [
        "Recording ID",
        "Method/Phase",
        "Total Fetal Beats Detected",
        "Mean Fetal Heart Rate (BPM)",
        "Fetal HR Standard Deviation (BPM)",
        "Mean Maternal Heart Rate (BPM)",
        "HR Separation Gap (BPM)",
        "Normal HR Beats (%)",
        "Bradycardic Beats (%)",
        "Tachycardic Beats (%)",
        "Mean Separation Confidence (%)",
        "HR Contrast Attribution (%)",
        "Temporal Independence Attribution (%)",
        "Morphological Consistency Attribution (%)",
        "Primary Separation Mechanism",
        "Physiological Plausibility",
        "Reliability Classification",
        "Clinical Decision Flag",
    ]

    def __init__(self):
        """Initialize the generator."""
        self.records: List[Dict] = []

    def parse_clinical_explanation(self, text: str) -> Dict:
        """
        Parse raw clinical_explanation text and extract structured metrics.

        Patterns extracted:
          - OVERALL CONFIDENCE: X%
          - Heart Rate Contrast [Y% attribution]
          - Temporal Independence [Z% attribution]
          - Morphological Consistency [W% attribution] or N/A

        Parameters
        ----------
        text : str
            Raw clinical explanation text from ECHO.

        Returns
        -------
        dict
            Extracted metrics: confidence, hr_attr, temporal_attr, morph_attr.
        """
        result = {
            "confidence": np.nan,
            "hr_attr": np.nan,
            "temporal_attr": np.nan,
            "morph_attr": np.nan,
        }

        if not text or not isinstance(text, str):
            return result

        # Extract OVERALL CONFIDENCE: X%
        conf_match = re.search(r"OVERALL CONFIDENCE:\s*([\d.]+)\s*%", text)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))

        # Extract Heart Rate Contrast [Y% attribution]
        hr_match = re.search(
            r"Heart Rate Contrast\s*\[([\d.]+)\s*%\s*attribution\]", text
        )
        if hr_match:
            result["hr_attr"] = float(hr_match.group(1))

        # Extract Temporal Independence [Z% attribution]
        temporal_match = re.search(
            r"Temporal Independence\s*\[([\d.]+)\s*%\s*attribution\]", text
        )
        if temporal_match:
            result["temporal_attr"] = float(temporal_match.group(1))

        # Extract Morphological Consistency [W% attribution]
        # Handle both "X% attribution" and "N/A -- no direct electrode reference"
        morph_match = re.search(
            r"Morphological Consistency\s*\[([\d.]+)\s*%\s*attribution\]", text
        )
        if morph_match:
            result["morph_attr"] = float(morph_match.group(1))
        elif "N/A -- no direct electrode reference" in text:
            result["morph_attr"] = np.nan  # Explicitly NaN for NIFECGDB

        return result

    def safe_float(self, value: any, default: float = np.nan) -> float:
        """
        Safely convert value to float, handling "N/A" and missing values.

        Parameters
        ----------
        value : any
            Value to convert.
        default : float
            Default value if conversion fails.

        Returns
        -------
        float
            Converted float or default.
        """
        if value is None or value == "" or value == "N/A":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def add_summary_record(
        self,
        recording: str,
        method: str,
        summary: Dict,
    ) -> None:
        """
        Add a recording-level summary record.

        Parameters
        ----------
        recording : str
            Recording ID (e.g., "r01", "ecgca102").
        method : str
            Method/phase (e.g., "PHASE_nifecgdb").
        summary : dict
            Raw summary dict from echo.generate_summary_dict().
            Must contain keys like n_beats, mean_fetal_hr, mean_confidence_pct, etc.
        """
        # Extract summary metrics
        total_beats = int(summary.get("n_beats", 0))
        mean_fhr = self.safe_float(summary.get("mean_fetal_hr"))
        fhr_std = self.safe_float(summary.get("fetal_hr_std"))
        maternal_hr = self.safe_float(summary.get("maternal_hr"))
        hr_separation = self.safe_float(summary.get("hr_separation"))
        normal_pct = self.safe_float(summary.get("normal_hr_pct"))
        brady_pct = self.safe_float(summary.get("bradycardia_pct"))
        tachy_pct = self.safe_float(summary.get("tachycardia_pct"))
        mean_conf = self.safe_float(summary.get("mean_confidence_pct"))

        # Parse clinical explanation to extract attribution percentages
        clinical_text = str(summary.get("clinical_explanation", ""))
        parsed = self.parse_clinical_explanation(clinical_text)

        hr_attr_pct = parsed.get("hr_attr")
        temporal_attr_pct = parsed.get("temporal_attr")
        morph_attr_pct = parsed.get("morph_attr")

        # Use confidence from summary if available; otherwise from parsed text
        if np.isnan(mean_conf) and not np.isnan(parsed.get("confidence")):
            mean_conf = parsed.get("confidence")

        # Compute derived columns
        dominant_cue = self._compute_dominant_cue(temporal_attr_pct, hr_attr_pct)
        physiological_plausibility = self._compute_physiological_plausibility(
            normal_pct
        )
        reliability_class = self._compute_reliability_class(mean_conf)
        clinical_flag = self._compute_clinical_flag(
            brady_pct, tachy_pct, mean_conf, normal_pct
        )

        record = {
            "Recording ID": recording,
            "Method/Phase": method,
            "Total Fetal Beats Detected": total_beats,
            "Mean Fetal Heart Rate (BPM)": round(mean_fhr, 1) if not np.isnan(mean_fhr) else np.nan,
            "Fetal HR Standard Deviation (BPM)": round(fhr_std, 1) if not np.isnan(fhr_std) else np.nan,
            "Mean Maternal Heart Rate (BPM)": round(maternal_hr, 1) if not np.isnan(maternal_hr) else np.nan,
            "HR Separation Gap (BPM)": round(hr_separation, 1) if not np.isnan(hr_separation) else np.nan,
            "Normal HR Beats (%)": round(normal_pct, 1) if not np.isnan(normal_pct) else np.nan,
            "Bradycardic Beats (%)": round(brady_pct, 1) if not np.isnan(brady_pct) else np.nan,
            "Tachycardic Beats (%)": round(tachy_pct, 1) if not np.isnan(tachy_pct) else np.nan,
            "Mean Separation Confidence (%)": round(mean_conf, 1) if not np.isnan(mean_conf) else np.nan,
            "HR Contrast Attribution (%)": round(hr_attr_pct, 1) if not np.isnan(hr_attr_pct) else np.nan,
            "Temporal Independence Attribution (%)": round(temporal_attr_pct, 1) if not np.isnan(temporal_attr_pct) else np.nan,
            "Morphological Consistency Attribution (%)": round(morph_attr_pct, 1) if not np.isnan(morph_attr_pct) else np.nan,
            "Primary Separation Mechanism": dominant_cue,
            "Physiological Plausibility": physiological_plausibility,
            "Reliability Classification": reliability_class,
            "Clinical Decision Flag": clinical_flag,
        }

        self.records.append(record)

    def _compute_dominant_cue(
        self, temporal_attr: float, hr_attr: float
    ) -> str:
        """
        Determine dominant separation cue.

        Rule:
          - if temporal attribution > HR contrast attribution: "Temporal Independence"
          - else: "HR Contrast"

        Parameters
        ----------
        temporal_attr : float
            Temporal independence attribution percentage.
        hr_attr : float
            HR contrast attribution percentage.

        Returns
        -------
        str
            Dominant cue label.
        """
        if np.isnan(temporal_attr) or np.isnan(hr_attr):
            return "Unknown"
        if temporal_attr > hr_attr:
            return "Temporal Independence"
        else:
            return "HR Contrast"

    def _compute_physiological_plausibility(self, normal_pct: float) -> str:
        """
        Classify physiological plausibility based on normal HR percentage.

        Rule:
          - normal_pct > 70 → "High"
          - 40–70 → "Moderate"
          - otherwise → "Low"

        Parameters
        ----------
        normal_pct : float
            Percentage of normal HR beats.

        Returns
        -------
        str
            Plausibility class.
        """
        if np.isnan(normal_pct):
            return "Unknown"
        if normal_pct > 70:
            return "High"
        elif normal_pct >= 40:
            return "Moderate"
        else:
            return "Low"

    def _compute_reliability_class(self, confidence: float) -> str:
        """
        Classify reliability based on mean confidence score.

        Rule:
          - confidence > 75 → "High Reliability"
          - 50–75 → "Moderate Reliability"
          - 25–50 → "Low Reliability"
          - below 25 → "Very Low Reliability"

        Parameters
        ----------
        confidence : float
            Mean confidence percentage (0-100).

        Returns
        -------
        str
            Reliability class.
        """
        if np.isnan(confidence):
            return "Unknown"
        if confidence > 75:
            return "High Reliability"
        elif confidence >= 50:
            return "Moderate Reliability"
        elif confidence >= 25:
            return "Low Reliability"
        else:
            return "Very Low Reliability"

    def _compute_clinical_flag(
        self,
        brady_pct: float,
        tachy_pct: float,
        confidence: float,
        normal_pct: float,
    ) -> str:
        """
        Determine clinical flag based on HR patterns and confidence.

        Rule (checked in order):
          - brady_pct > 40 → "Bradycardia Dominant"
          - tachy_pct > 25 → "Tachycardia Dominant"
          - confidence < 20 → "Physician Review Recommended"
          - otherwise → "Clinically Acceptable"

        Parameters
        ----------
        brady_pct : float
            Percentage of bradycardic beats.
        tachy_pct : float
            Percentage of tachycardic beats.
        confidence : float
            Mean confidence percentage.
        normal_pct : float
            Percentage of normal HR beats.

        Returns
        -------
        str
            Clinical flag.
        """
        if np.isnan(brady_pct):
            brady_pct = 0.0
        if np.isnan(tachy_pct):
            tachy_pct = 0.0
        if np.isnan(confidence):
            confidence = 0.0

        if brady_pct > 40:
            return "Bradycardia Dominant"
        elif tachy_pct > 25:
            return "Tachycardia Dominant"
        elif confidence < 20:
            return "Physician Review Recommended"
        else:
            return "Clinically Acceptable"

    def build_dataframe(self) -> pd.DataFrame:
        """
        Build a pandas DataFrame from accumulated records.

        Returns
        -------
        pd.DataFrame
            Master Explainability Table with all records.
        """
        if not self.records:
            print("[ECHOMasterTableGenerator] No records to build DataFrame.")
            return pd.DataFrame(columns=self.MASTER_COLUMNS)

        df = pd.DataFrame(self.records)

        # Ensure all columns are present
        for col in self.MASTER_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        # Return only target columns in correct order
        return df[self.MASTER_COLUMNS]

    def save_csv(self, dataset_name: str, output_dir: str = "results") -> Optional[str]:
        """
        Save the Master Explainability Table to CSV.

        Parameters
        ----------
        dataset_name : str
            Dataset name (e.g., "nifecgdb", "adfecgdb").
        output_dir : str
            Directory to save the CSV file.

        Returns
        -------
        str or None
            Path to saved CSV file, or None if no records.
        """
        if not self.records:
            print("[ECHOMasterTableGenerator] No records to save.")
            return None

        output_path = Path(output_dir) / f"echo_master_table_{dataset_name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.build_dataframe()
        df.to_csv(output_path, index=False)

        print(
            f"[ECHOMasterTableGenerator] Saved Master Explainability Table → {output_path}"
        )
        return str(output_path)

    def reset(self) -> None:
        """Clear all accumulated records."""
        self.records = []
