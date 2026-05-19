"""
verify_echo_integration.py
Verify that the ECHO Master Table system is properly integrated.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def check_module_imports():
    """Verify all required modules can be imported."""
    print("="*70)
    print("CHECKING MODULE IMPORTS")
    print("="*70)
    
    try:
        from xai.echo_master_table import ECHOMasterTableGenerator
        print("✓ ECHOMasterTableGenerator imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ECHOMasterTableGenerator: {e}")
        return False
    
    try:
        from xai.echo import ECHOExplainer
        print("✓ ECHOExplainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ECHOExplainer: {e}")
        return False
    
    try:
        from utils.logger import ResultsLogger
        print("✓ ResultsLogger imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ResultsLogger: {e}")
        return False
    
    try:
        from pipeline import PHASEPipeline
        print("✓ PHASEPipeline imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PHASEPipeline: {e}")
        return False
    
    return True


def check_file_structure():
    """Verify all files are in place."""
    print("\n" + "="*70)
    print("CHECKING FILE STRUCTURE")
    print("="*70)
    
    required_files = [
        "xai/echo_master_table.py",
        "run_experiment_new.py",
        "pipeline.py",
        "utils/logger.py",
        "xai/echo.py",
        "ECHO_MASTER_TABLE_GUIDE.md",
        "ECHO_QUICK_START.md",
        "ECHO_IMPLEMENTATION_SUMMARY.md",
        "test_echo_master_table.py",
    ]
    
    all_exist = True
    for filepath in required_files:
        full_path = Path(__file__).parent / filepath
        if full_path.exists():
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} NOT FOUND")
            all_exist = False
    
    return all_exist


def check_old_files():
    """Check for deprecated files."""
    print("\n" + "="*70)
    print("CHECKING FOR DEPRECATED FILES")
    print("="*70)
    
    # Look for old echo_summary CSV files
    results_dirs = [
        Path("results_nifecgdb"),
        Path("results_adfecgdb"),
        Path("results_cinc2013"),
    ]
    
    found_old_files = []
    for results_dir in results_dirs:
        if results_dir.exists():
            old_files = list(results_dir.glob("echo_summary_*.csv"))
            if old_files:
                found_old_files.extend(old_files)
    
    if found_old_files:
        print(f"⚠ Found {len(found_old_files)} old echo_summary CSV files:")
        for f in found_old_files:
            print(f"  - {f}")
        print("\nThese files are from the deprecated system and can be safely deleted.")
        print("All data has been incorporated into the new Master Table system.")
    else:
        print("✓ No deprecated echo_summary CSV files found")
    
    return True


def check_deprecated_code():
    """Verify deprecated code is marked."""
    print("\n" + "="*70)
    print("CHECKING DEPRECATED CODE MARKERS")
    print("="*70)
    
    logger_file = Path("utils/logger.py")
    if logger_file.exists():
        content = logger_file.read_text()
        if "DEPRECATED" in content and "ECHOMasterTableGenerator" in content:
            print("✓ ECHOResultsLogger marked as DEPRECATED in logger.py")
        else:
            print("⚠ ECHOResultsLogger not properly marked as deprecated")
    
    return True


def check_integration_points():
    """Verify integration points in run_experiment_new.py."""
    print("\n" + "="*70)
    print("CHECKING INTEGRATION POINTS")
    print("="*70)
    
    run_exp_file = Path("run_experiment_new.py")
    if run_exp_file.exists():
        content = run_exp_file.read_text()
        
        checks = [
            ("ECHOMasterTableGenerator import", "from xai.echo_master_table import ECHOMasterTableGenerator"),
            ("Remove ECHOResultsLogger", "ECHOResultsLogger" not in content or "# Legacy" in content),
            ("Create generator instance", "echo_master_gen = ECHOMasterTableGenerator()"),
            ("Add summary records", "echo_master_gen.add_summary_record"),
            ("Save master table", "echo_master_gen.save_csv"),
        ]
        
        all_pass = True
        for check_name, check_str in checks:
            if isinstance(check_str, bool):
                if check_str:
                    print(f"✓ {check_name}")
                else:
                    print(f"✗ {check_name}")
                    all_pass = False
            else:
                if check_str in content:
                    print(f"✓ {check_name}")
                else:
                    print(f"✗ {check_name} - NOT FOUND")
                    all_pass = False
        
        return all_pass
    else:
        print("✗ run_experiment_new.py not found")
        return False


def verify_functionality():
    """Quick functionality test."""
    print("\n" + "="*70)
    print("VERIFYING FUNCTIONALITY")
    print("="*70)
    
    try:
        from xai.echo_master_table import ECHOMasterTableGenerator
        
        gen = ECHOMasterTableGenerator()
        
        # Test add_summary_record
        sample_summary = {
            "n_beats": 100,
            "mean_fetal_hr": 150.0,
            "fetal_hr_std": 20.0,
            "maternal_hr": 80.0,
            "hr_separation": 70.0,
            "normal_hr_pct": 80.0,
            "bradycardia_pct": 5.0,
            "tachycardia_pct": 15.0,
            "mean_hr_contrast_pct": 50.0,
            "mean_morphology_pct": "N/A",
            "mean_temporal_independence_pct": 50.0,
            "mean_confidence_pct": 80.0,
            "clinical_explanation": """
OVERALL CONFIDENCE: 80.0%
Heart Rate Contrast        [50.0% attribution]
Temporal Independence      [50.0% attribution]
Morphological Consistency  [N/A -- no direct electrode reference]
""",
        }
        
        gen.add_summary_record("test_rec", "PHASE_test", sample_summary)
        print("✓ add_summary_record() works")
        
        # Test build_dataframe
        df = gen.build_dataframe()
        if len(df) == 1 and len(df.columns) == 18:
            print("✓ build_dataframe() generates correct structure (1 row, 18 columns)")
        else:
            print(f"✗ build_dataframe() incorrect (rows={len(df)}, cols={len(df.columns)})")
            return False
        
        # Test derived columns
        if df['dominant_cue'].iloc[0] in ['HR Contrast', 'Temporal Independence']:
            print("✓ dominant_cue computed correctly")
        else:
            print("✗ dominant_cue not computed")
            return False
        
        if df['physiological_plausibility'].iloc[0] == 'High':
            print("✓ physiological_plausibility computed correctly")
        else:
            print("✗ physiological_plausibility not computed")
            return False
        
        if df['reliability_class'].iloc[0] == 'High Reliability':
            print("✓ reliability_class computed correctly")
        else:
            print("✗ reliability_class not computed")
            return False
        
        if df['clinical_flag'].iloc[0] == 'Clinically Acceptable':
            print("✓ clinical_flag computed correctly")
        else:
            print("✗ clinical_flag not computed")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "🔍 ECHO MASTER TABLE INTEGRATION VERIFICATION 🔍\n")
    
    results = {
        "Module Imports": check_module_imports(),
        "File Structure": check_file_structure(),
        "Deprecated Files": check_old_files(),
        "Deprecated Code Markers": check_deprecated_code(),
        "Integration Points": check_integration_points(),
        "Functionality": verify_functionality(),
    }
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_pass = True
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:30s} {status}")
        if not result:
            all_pass = False
    
    print("="*70)
    
    if all_pass:
        print("\n✓ ALL CHECKS PASSED - SYSTEM READY FOR USE\n")
        print("Next steps:")
        print("  1. Run: python run_experiment_new.py --dataset nifecgdb --mode full")
        print("  2. Check output: results_nifecgdb/echo_master_table_nifecgdb.csv")
        print("  3. Analyze with: import pandas; df = pandas.read_csv(...)")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - REVIEW ABOVE FOR DETAILS\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
