"""
test_echo_master_table.py
Test the ECHOMasterTableGenerator with sample ECHO data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from xai.echo_master_table import ECHOMasterTableGenerator
import pandas as pd


def test_master_table_generator():
    """Test the Master Table Generator with realistic sample data."""
    gen = ECHOMasterTableGenerator()

    # Sample data mimicking real ECHO summaries
    sample_summaries = [
        {
            "n_beats": 408,
            "mean_fetal_hr": 111.2,
            "fetal_hr_std": 44.2,
            "maternal_hr": 85.8,
            "hr_separation": 25.4,
            "normal_hr_pct": 38.0,
            "bradycardia_pct": 46.6,
            "tachycardia_pct": 15.4,
            "mean_hr_contrast_pct": 47.2,
            "mean_morphology_pct": "N/A",
            "mean_temporal_independence_pct": 52.8,
            "mean_confidence_pct": 44.5,
            "clinical_explanation": """
+--------------------------------------------------------------+
  ECHO Clinical Explanation -- Fetal Beat #1
+--------------------------------------------------------------+

  OVERALL CONFIDENCE: 4.1%

  SEPARATION RATIONALE:
  --------------------------------------------------------------
  [WARN] Heart Rate Contrast        [4.8% attribution]
     Instantaneous fetal HR = 86.6 BPM
     Status: OUTSIDE normal range (100-185 BPM)
     Maternal HR            = 85.8 BPM
     HR separation          = 0.8 BPM

  [ok]   Morphological Consistency  [N/A -- no direct electrode reference]
     Cannot assess: NIFECGDB does not provide a reference waveform.

  [ok] Temporal Independence      [95.2% attribution]
     This fetal beat is temporally separated from maternal QRS.
""",
        },
        {
            "n_beats": 350,
            "mean_fetal_hr": 142.5,
            "fetal_hr_std": 25.3,
            "maternal_hr": 78.2,
            "hr_separation": 64.3,
            "normal_hr_pct": 85.0,
            "bradycardia_pct": 5.0,
            "tachycardia_pct": 10.0,
            "mean_hr_contrast_pct": 60.0,
            "mean_morphology_pct": "N/A",
            "mean_temporal_independence_pct": 40.0,
            "mean_confidence_pct": 82.3,
            "clinical_explanation": """
+--------------------------------------------------------------+
  ECHO Clinical Explanation -- Fetal Beat #2
+--------------------------------------------------------------+

  OVERALL CONFIDENCE: 82.3%

  SEPARATION RATIONALE:
  [ok] Heart Rate Contrast        [60.0% attribution]
     Instantaneous fetal HR = 142.5 BPM
     Status: NORMAL (100-185 BPM)
     Maternal HR            = 78.2 BPM
     HR separation          = 64.3 BPM

  [ok]   Morphological Consistency  [N/A -- no direct electrode reference]
     Cannot assess: NIFECGDB does not provide a reference waveform.

  [ok] Temporal Independence      [40.0% attribution]
     This fetal beat is temporally separated from maternal QRS.
""",
        },
    ]

    recordings = ["ecgca102", "ecgca115"]
    for i, (rec, summary) in enumerate(zip(recordings, sample_summaries)):
        gen.add_summary_record(
            recording=rec,
            method="PHASE_nifecgdb",
            summary=summary,
        )

    # Build and display the dataframe
    df = gen.build_dataframe()
    print("\n" + "="*80)
    print("Master Explainability Table:")
    print("="*80)
    print(df.to_string())

    print("\n" + "="*80)
    print("Column-by-column verification:")
    print("="*80)
    for col in df.columns:
        print(f"{col:30s} → {df[col].tolist()}")

    # Test derived columns
    print("\n" + "="*80)
    print("Derived Columns Analysis:")
    print("="*80)
    print(f"dominant_cue:                {df['dominant_cue'].tolist()}")
    print(f"physiological_plausibility:  {df['physiological_plausibility'].tolist()}")
    print(f"reliability_class:           {df['reliability_class'].tolist()}")
    print(f"clinical_flag:               {df['clinical_flag'].tolist()}")

    # Save and verify
    filepath = gen.save_csv("test", output_dir=".")
    print(f"\n[SUCCESS] Master Table saved to: {filepath}")

    # Verify file was created
    if Path(filepath).exists():
        print(f"[VERIFIED] File exists and is readable.")
        loaded_df = pd.read_csv(filepath)
        print(f"[VERIFIED] Loaded {len(loaded_df)} rows, {len(loaded_df.columns)} columns")
        print("\nFirst few rows of saved CSV:")
        print(loaded_df.head().to_string())


if __name__ == "__main__":
    test_master_table_generator()
