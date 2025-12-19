import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# STEP 1: DATA LOADING
# ==========================================

def load_data(filepath):
    """Load data from CSV file"""
    print("=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} records from {filepath}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    return df

# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

def clean_data(df):
    """Clean dataset: handle missing values, duplicates, inconsistencies"""
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

    initial_rows = len(df)

    # 2.1 Handle duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicates} duplicate records")

    # 2.2 Fix inconsistent operational_status (standardize to title case)
    df['operational_status'] = df['operational_status'].str.title()
    print(f"✓ Standardized operational_status to title case")

    # 2.3 Convert date columns to datetime FIRST
    df['usage_date'] = pd.to_datetime(df['usage_date'])
    df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'], errors='coerce') # Use coerce to turn unparseable dates into NaT

    # 2.4 Handle missing values
    print("\nMissing Values Before Cleaning:")
    missing_before = df.isnull().sum()
    print(missing_before[missing_before > 0])

    # Fill missing usage_hours with median by equipment_type
    df['usage_hours'] = df.groupby('equipment_type')['usage_hours'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill missing maintenance dates with earliest date in dataset
    if df['last_maintenance_date'].isnull().any():
        # Find the minimum *valid* date
        earliest_date = df['last_maintenance_date'].min()
        df['last_maintenance_date'].fillna(earliest_date, inplace=True)

    print("\nMissing Values After Cleaning:")
    missing_after = df.isnull().sum()
    print(missing_after[missing_after > 0] if missing_after.any() else "No missing values")

    print(f"\n✓ Data cleaning complete: {initial_rows} \u2192 {len(df)} records")

    return df

# ==========================================
# STEP 3: DATA VALIDATION
# ==========================================

def validate_data(df):
    """Perform data quality checks"""
    print("\n" + "=" * 60)
    print("STEP 3: DATA VALIDATION")
    print("=" * 60)

    issues = []

    # Check 1: Usage hours should not exceed capacity
    invalid_usage = df[df['usage_hours'] > df['max_daily_capacity']]
    if len(invalid_usage) > 0:
        issues.append(f"Found {len(invalid_usage)} records where usage > capacity")
        df.loc[df['usage_hours'] > df['max_daily_capacity'], 'usage_hours'] = \
            df.loc[df['usage_hours'] > df['max_daily_capacity'], 'max_daily_capacity']

    # Check 2: Valid operational status values
    valid_statuses = ['Active', 'Idle', 'Under Maintenance', 'Faulty']
    invalid_status = df[~df['operational_status'].isin(valid_statuses)]
    if len(invalid_status) > 0:
        issues.append(f"Found {len(invalid_status)} records with invalid status")

    # Check 3: Maintenance flag should be 0, 1, or 2
    invalid_flags = df[~df['maintenance_flag'].isin([0, 1, 2])]
    if len(invalid_flags) > 0:
        issues.append(f"Found {len(invalid_flags)} records with invalid maintenance_flag")

    # Check 4: Usage dates should not be in the future
    future_dates = df[df['usage_date'] > datetime.now()]
    if len(future_dates) > 0:
        issues.append(f"Found {len(future_dates)} records with future dates")

    if issues:
        print("\u26a0 Validation Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All validation checks passed")

    return df

# ==========================================
# STEP 4: FEATURE ENGINEERING
# ==========================================

def engineer_features(df):
    """Create derived features for analysis"""
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 60)

    # Feature 1: Utilization rate (percentage)
    df['utilization_rate'] = (df['usage_hours'] / df['max_daily_capacity']) * 100

    # Feature 2: Idle indicator (binary)
    df['is_idle'] = (df['utilization_rate'] < 20).astype(int)

    # Feature 3: Overused indicator (binary)
    df['is_overused'] = (df['utilization_rate'] > 80).astype(int)

    # Feature 4: Days since last maintenance
    df['days_since_maintenance'] = (df['usage_date'] - df['last_maintenance_date']).dt.days

    # Feature 5: Maintenance risk score (0-10)
    df['maintenance_risk_score'] = np.clip(df['days_since_maintenance'] / 15, 0, 10)

    # Feature 6: Daily operational cost
    df['daily_cost'] = df['usage_hours'] * df['cost_per_hour']

    # Feature 7: Month and quarter for time-based analysis
    df['month'] = df['usage_date'].dt.month
    df['quarter'] = df['usage_date'].dt.quarter

    print("✓ Created features:")
    print("  - utilization_rate: Usage as % of capacity")
    print("  - is_idle: Flag for underutilized equipment (<20%)")
    print("  - is_overused: Flag for overworked equipment (>80%)")
    print("  - days_since_maintenance: Time since last service")
    print("  - maintenance_risk_score: Predictive maintenance score (0-10)")
    print("  - daily_cost: Operational cost per day")
    print("  - month, quarter: Time period identifiers")

    return df

# ==========================================
# STEP 5: ANALYSIS
# ==========================================

def perform_analysis(df):
    """Perform comprehensive equipment utilization analysis"""
    print("\n" + "=" * 60)
    print("STEP 5: ANALYSIS & INSIGHTS")
    print("=" * 60)

    # Analysis 1: Overall utilization statistics
    print("\n1. OVERALL UTILIZATION STATISTICS")
    print("-" * 60)
    print(f"Average Utilization Rate: {df['utilization_rate'].mean():.2f}%")
    print(f"Median Utilization Rate: {df['utilization_rate'].median():.2f}%")
    print(f"Std Dev: {df['utilization_rate'].std():.2f}%")

    # Analysis 2: Equipment type performance
    print("\n2. UTILIZATION BY EQUIPMENT TYPE")
    print("-" * 60)
    type_analysis = df.groupby('equipment_type').agg({
        'utilization_rate': 'mean',
        'usage_hours': 'sum',
        'daily_cost': 'sum',
        'equipment_id': 'count'
    }).round(2)
    type_analysis.columns = ['Avg Util %', 'Total Hours', 'Total Cost $', 'Records']
    type_analysis = type_analysis.sort_values('Avg Util %', ascending=False)
    print(type_analysis)

    # Analysis 3: Identify underutilized equipment
    print("\n3. UNDERUTILIZED EQUIPMENT (< 30% utilization)")
    print("-" * 60)
    underutilized = df.groupby('equipment_id').agg({
        'equipment_type': 'first',
        'department': 'first',
        'utilization_rate': 'mean'
    }).round(2)
    underutilized = underutilized[underutilized['utilization_rate'] < 30]
    underutilized = underutilized.sort_values('utilization_rate')
    print(underutilized.head(10))

    # Analysis 4: Identify overused equipment
    print("\n4. OVERUSED EQUIPMENT (> 80% utilization)")
    print("-" * 60)
    overused = df.groupby('equipment_id').agg({
        'equipment_type': 'first',
        'department': 'first',
        'utilization_rate': 'mean'
    }).round(2)
    overused = overused[overused['utilization_rate'] > 80]
    overused = overused.sort_values('utilization_rate', ascending=False)
    print(overused.head(10))

    # Analysis 5: Maintenance risk assessment
    print("\n5. MAINTENANCE RISK ASSESSMENT")
    print("-" * 60)
    maintenance_risk = df.groupby(['equipment_id', 'maintenance_flag']).agg({
        'equipment_type': 'first',
        'days_since_maintenance': 'mean'
    }).round(0)
    high_risk = maintenance_risk[maintenance_risk.index.get_level_values('maintenance_flag') >= 1]
    print(f"Equipment requiring attention: {len(high_risk)} items")
    print(high_risk.head(10))

    # Analysis 6: Department-wise analysis
    print("\n6. DEPARTMENT-WISE ANALYSIS")
    print("-" * 60)
    dept_analysis = df.groupby('department').agg({
        'utilization_rate': 'mean',
        'daily_cost': 'sum',
        'equipment_id': 'nunique'
    }).round(2)
    dept_analysis.columns = ['Avg Util %', 'Total Cost $', 'Unique Equipment']
    print(dept_analysis.sort_values('Total Cost $', ascending=False))

    return df

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INTELLIGENT EQUIPMENT MONITORING - PROCEDURAL PIPELINE")
    print("=" * 60)

    # Execute pipeline
    df = load_data('equipment_usage_data.csv')
    df = clean_data(df)
    df = validate_data(df)
    df = engineer_features(df)
    df = perform_analysis(df)

    # Save processed data
    output_file = 'equipment_data_processed.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Processed data saved to: {output_file}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
