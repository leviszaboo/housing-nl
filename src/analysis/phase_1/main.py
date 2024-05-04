from src.analysis.phase_1.tables import create_tables

def run_phase_1():
    print("Running Phase 1 analysis...")

    # Create tables for the summary statistics of the main dataset
    create_tables()

    print("Phase 1 analysis complete")
    return 