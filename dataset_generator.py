import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_EQUIPMENT = 20
NUM_RECORDS = 500
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 15)

# Equipment master data
equipment_types = ['MRI Scanner', 'CT Scanner', 'Ultrasound', 'X-Ray', 
                   'Centrifuge', 'Spectrophotometer', 'PCR Machine', 
                   'Ventilator', 'ECG Machine', 'Blood Analyzer']

departments = ['Radiology', 'Pathology', 'Cardiology', 'Emergency', 
               'Research Lab', 'ICU', 'Diagnostics']

# Equipment master list
equipment_list = []
for i in range(NUM_EQUIPMENT):
    eq_type = random.choice(equipment_types)
    
    # Define capacity based on equipment type
    if eq_type in ['MRI Scanner', 'CT Scanner', 'Ventilator']:
        capacity = 24  # 24/7 equipment
    elif eq_type in ['Ultrasound', 'X-Ray', 'ECG Machine']:
        capacity = 12  # Standard shift
    else:
        capacity = 8   # Research equipment
    
    # Cost varies by equipment sophistication
    if eq_type in ['MRI Scanner', 'CT Scanner']:
        cost = np.random.uniform(150, 300)
    elif eq_type in ['PCR Machine', 'Blood Analyzer']:
        cost = np.random.uniform(50, 100)
    else:
        cost = np.random.uniform(20, 80)
    
    equipment_list.append({
        'equipment_id': f'EQ-{i+1:03d}',
        'equipment_type': eq_type,
        'department': random.choice(departments),
        'max_daily_capacity': capacity,
        'cost_per_hour': round(cost, 2)
    })

equipment_df = pd.DataFrame(equipment_list)

# Generate usage records
records = []

for _ in range(NUM_RECORDS):
    equipment = equipment_list[random.randint(0, NUM_EQUIPMENT-1)]
    usage_date = START_DATE + timedelta(days=random.randint(0, (END_DATE - START_DATE).days))
    
    # Generate realistic usage patterns
    eq_type = equipment['equipment_type']
    capacity = equipment['max_daily_capacity']
    
    # High-utilization equipment
    if eq_type in ['MRI Scanner', 'CT Scanner', 'Ventilator']:
        usage_hours = np.random.uniform(0.7 * capacity, 0.95 * capacity)
    # Medium utilization
    elif eq_type in ['Ultrasound', 'X-Ray', 'Blood Analyzer']:
        usage_hours = np.random.uniform(0.4 * capacity, 0.75 * capacity)
    # Low utilization (research equipment)
    else:
        usage_hours = np.random.uniform(0.1 * capacity, 0.5 * capacity)
    
    # Add some realistic noise
    usage_hours = max(0, usage_hours + np.random.normal(0, 0.5))
    
    # Operational status distribution
    status_choices = ['Active', 'Idle', 'Under Maintenance', 'Faulty']
    status_weights = [0.75, 0.15, 0.07, 0.03]
    operational_status = random.choices(status_choices, weights=status_weights)[0]
    
    # Maintenance logic
    last_maint = usage_date - timedelta(days=random.randint(10, 180))
    days_since_maint = (usage_date - last_maint).days
    
    if days_since_maint > 120:
        maintenance_flag = 2  # Overdue
    elif days_since_maint > 90:
        maintenance_flag = 1  # Due soon
    else:
        maintenance_flag = 0  # OK
    
    records.append({
        'equipment_id': equipment['equipment_id'],
        'equipment_type': equipment['equipment_type'],
        'department': equipment['department'],
        'usage_date': usage_date.strftime('%Y-%m-%d'),
        'usage_hours': round(usage_hours, 2),
        'max_daily_capacity': capacity,
        'operational_status': operational_status,
        'maintenance_flag': maintenance_flag,
        'last_maintenance_date': last_maint.strftime('%Y-%m-%d'),
        'cost_per_hour': equipment['cost_per_hour']
    })

# Create DataFrame
df = pd.DataFrame(records)

# Introduce realistic data quality issues
# 5% missing values in usage_hours
missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[missing_indices, 'usage_hours'] = np.nan

# 3% missing maintenance dates
missing_maint = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
df.loc[missing_maint, 'last_maintenance_date'] = None

# 2% duplicate records
duplicate_rows = df.sample(n=int(0.02 * len(df)))
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Some inconsistent status entries (typos)
inconsistent_indices = np.random.choice(df.index, size=5, replace=False)
df.loc[inconsistent_indices, 'operational_status'] = 'active'  # lowercase inconsistency

# Save to CSV
output_file = 'equipment_usage_data.csv'
df.to_csv(output_file, index=False)

print(f"✓ Generated dataset with {len(df)} records")
print(f"✓ {NUM_EQUIPMENT} unique equipment items")
print(f"✓ Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
print(f"✓ Saved to: {output_file}")
print(f"\nDataset Preview:")
print(df.head(10))
print(f"\nData Quality Issues Introduced:")
print(f"  - Missing values: ~5% in usage_hours, ~3% in maintenance dates")
print(f"  - Duplicate records: ~2%")
print(f"  - Inconsistent entries: 5 records with lowercase status")