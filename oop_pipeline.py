import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# BASE CLASS: DataProcessor
# ==========================================

class DataProcessor:
    """Base class for data processing operations"""
    
    def __init__(self):
        self.df = None
        self.metadata = {}
    
    def log(self, message: str, level: str = "INFO"):
        """Logging utility"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

# ==========================================
# CLASS 1: DataLoader
# ==========================================

class DataLoader(DataProcessor):
    """Handles data loading and initial inspection"""
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        self.log("Loading data from CSV...", "INFO")
        
        try:
            self.df = pd.read_csv(filepath)
            self.metadata['source_file'] = filepath
            self.metadata['initial_records'] = len(self.df)
            self.metadata['columns'] = list(self.df.columns)
            
            self.log(f"✓ Loaded {len(self.df)} records", "SUCCESS")
            return self.df
        
        except FileNotFoundError:
            self.log(f"File not found: {filepath}", "ERROR")
            raise
        except Exception as e:
            self.log(f"Error loading data: {str(e)}", "ERROR")
            raise
    
    def get_summary(self) -> Dict:
        """Get data summary statistics"""
        if self.df is None:
            raise ValueError("No data loaded")
        
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_kb': self.df.memory_usage(deep=True).sum() / 1024,
            'dtypes': self.df.dtypes.value_counts().to_dict()
        }

# ==========================================
# CLASS 2: DataCleaner
# ==========================================

class DataCleaner(DataProcessor):
    """Handles data cleaning operations"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.copy()
        self.cleaning_report = {}
    
    def remove_duplicates(self) -> 'DataCleaner':
        """Remove duplicate records"""
        initial = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial - len(self.df)
        
        self.cleaning_report['duplicates_removed'] = removed
        self.log(f"✓ Removed {removed} duplicates", "SUCCESS")
        
        return self
    
    def standardize_text_columns(self, columns: List[str]) -> 'DataCleaner':
        """Standardize text columns to title case"""
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.title()
        
        self.log(f"✓ Standardized {len(columns)} text columns", "SUCCESS")
        return self
    
    def handle_missing_values(self, strategy: Dict[str, str]) -> 'DataCleaner':
        """Handle missing values based on strategy
        
        Args:
            strategy: Dict mapping column names to strategies 
                     ('median', 'mean', 'mode', 'forward_fill', 'drop')
        """
        missing_before = self.df.isnull().sum().sum()
        
        for col, method in strategy.items():
            if col not in self.df.columns:
                continue
            
            if method == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif method == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif method == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif method == 'forward_fill':
                self.df[col].fillna(method='ffill', inplace=True)
            elif method == 'group_median':
                # Special case: fill by group median
                if 'equipment_type' in self.df.columns:
                    self.df[col] = self.df.groupby('equipment_type')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
        
        missing_after = self.df.isnull().sum().sum()
        self.cleaning_report['missing_values_handled'] = missing_before - missing_after
        
        self.log(f"✓ Handled {missing_before - missing_after} missing values", "SUCCESS")
        return self
    
    def convert_date_columns(self, columns: List[str]) -> 'DataCleaner':
        """Convert columns to datetime format"""
        for col in columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
        
        self.log(f"✓ Converted {len(columns)} columns to datetime", "SUCCESS")
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return cleaned dataframe"""
        return self.df
    
    def get_report(self) -> Dict:
        """Get cleaning report"""
        return self.cleaning_report

# ==========================================
# CLASS 3: DataValidator
# ==========================================

class DataValidator(DataProcessor):
    """Performs data quality validation"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.copy()
        self.validation_issues = []
    
    def validate_range(self, column: str, min_val: float, max_val: float) -> 'DataValidator':
        """Validate that column values are within range"""
        out_of_range = self.df[(self.df[column] < min_val) | (self.df[column] > max_val)]
        
        if len(out_of_range) > 0:
            self.validation_issues.append({
                'check': 'range_validation',
                'column': column,
                'failed_records': len(out_of_range),
                'details': f"Values outside [{min_val}, {max_val}]"
            })
        
        return self
    
    def validate_usage_capacity(self) -> 'DataValidator':
        """Validate that usage doesn't exceed capacity"""
        invalid = self.df[self.df['usage_hours'] > self.df['max_daily_capacity']]
        
        if len(invalid) > 0:
            self.validation_issues.append({
                'check': 'usage_capacity',
                'failed_records': len(invalid),
                'details': 'Usage hours exceed capacity'
            })
            # Auto-fix: cap at capacity
            self.df.loc[self.df['usage_hours'] > self.df['max_daily_capacity'], 'usage_hours'] = \
                self.df.loc[self.df['usage_hours'] > self.df['max_daily_capacity'], 'max_daily_capacity']
        
        return self
    
    def validate_categorical(self, column: str, valid_values: List[str]) -> 'DataValidator':
        """Validate categorical column values"""
        invalid = self.df[~self.df[column].isin(valid_values)]
        
        if len(invalid) > 0:
            self.validation_issues.append({
                'check': 'categorical_validation',
                'column': column,
                'failed_records': len(invalid),
                'details': f"Invalid values in {column}"
            })
        
        return self
    
    def validate_future_dates(self, column: str) -> 'DataValidator':
        """Check for future dates"""
        future = self.df[self.df[column] > datetime.now()]
        
        if len(future) > 0:
            self.validation_issues.append({
                'check': 'future_dates',
                'column': column,
                'failed_records': len(future),
                'details': 'Records with future dates found'
            })
        
        return self
    
    def is_valid(self) -> bool:
        """Check if data passed all validations"""
        return len(self.validation_issues) == 0
    
    def get_issues(self) -> List[Dict]:
        """Get list of validation issues"""
        return self.validation_issues
    
    def get_validated_data(self) -> pd.DataFrame:
        """Return validated dataframe"""
        return self.df

# ==========================================
# CLASS 4: FeatureEngineer
# ==========================================

class FeatureEngineer(DataProcessor):
    """Creates derived features for analysis"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.copy()
        self.features_created = []
    
    def create_utilization_rate(self) -> 'FeatureEngineer':
        """Calculate utilization rate as percentage"""
        self.df['utilization_rate'] = (self.df['usage_hours'] / self.df['max_daily_capacity']) * 100
        self.features_created.append('utilization_rate')
        return self
    
    def create_idle_indicator(self, threshold: float = 20) -> 'FeatureEngineer':
        """Binary flag for underutilized equipment"""
        self.df['is_idle'] = (self.df['utilization_rate'] < threshold).astype(int)
        self.features_created.append('is_idle')
        return self
    
    def create_overused_indicator(self, threshold: float = 80) -> 'FeatureEngineer':
        """Binary flag for overused equipment"""
        self.df['is_overused'] = (self.df['utilization_rate'] > threshold).astype(int)
        self.features_created.append('is_overused')
        return self
    
    def create_maintenance_features(self) -> 'FeatureEngineer':
        """Create maintenance-related features"""
        self.df['days_since_maintenance'] = (
            self.df['usage_date'] - self.df['last_maintenance_date']
        ).dt.days
        
        self.df['maintenance_risk_score'] = np.clip(
            self.df['days_since_maintenance'] / 15, 0, 10
        )
        
        self.features_created.extend(['days_since_maintenance', 'maintenance_risk_score'])
        return self
    
    def create_cost_features(self) -> 'FeatureEngineer':
        """Calculate operational costs"""
        self.df['daily_cost'] = self.df['usage_hours'] * self.df['cost_per_hour']
        self.features_created.append('daily_cost')
        return self
    
    def create_temporal_features(self) -> 'FeatureEngineer':
        """Extract time-based features"""
        self.df['month'] = self.df['usage_date'].dt.month
        self.df['quarter'] = self.df['usage_date'].dt.quarter
        self.df['day_of_week'] = self.df['usage_date'].dt.dayofweek
        self.features_created.extend(['month', 'quarter', 'day_of_week'])
        return self
    
    def get_features(self) -> List[str]:
        """Get list of created features"""
        return self.features_created
    
    def get_engineered_data(self) -> pd.DataFrame:
        """Return dataframe with new features"""
        return self.df

# ==========================================
# CLASS 5: EquipmentAnalyzer
# ==========================================

class EquipmentAnalyzer(DataProcessor):
    """Performs equipment utilization analysis"""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.analysis_results = {}
    
    def overall_statistics(self) -> Dict:
        """Calculate overall utilization statistics"""
        stats = {
            'mean_utilization': self.df['utilization_rate'].mean(),
            'median_utilization': self.df['utilization_rate'].median(),
            'std_utilization': self.df['utilization_rate'].std(),
            'total_usage_hours': self.df['usage_hours'].sum(),
            'total_cost': self.df['daily_cost'].sum(),
            'total_records': len(self.df)
        }
        
        self.analysis_results['overall'] = stats
        return stats
    
    def equipment_type_analysis(self) -> pd.DataFrame:
        """Analyze utilization by equipment type"""
        type_stats = self.df.groupby('equipment_type').agg({
            'utilization_rate': 'mean',
            'usage_hours': 'sum',
            'daily_cost': 'sum',
            'equipment_id': 'nunique'
        }).round(2)
        
        type_stats.columns = ['Avg Util %', 'Total Hours', 'Total Cost $', 'Unique Items']
        type_stats = type_stats.sort_values('Avg Util %', ascending=False)
        
        self.analysis_results['by_type'] = type_stats
        return type_stats
    
    def identify_underutilized(self, threshold: float = 30) -> pd.DataFrame:
        """Identify underutilized equipment"""
        equipment_util = self.df.groupby('equipment_id').agg({
            'equipment_type': 'first',
            'department': 'first',
            'utilization_rate': 'mean'
        }).round(2)
        
        underutilized = equipment_util[equipment_util['utilization_rate'] < threshold]
        underutilized = underutilized.sort_values('utilization_rate')
        
        self.analysis_results['underutilized'] = underutilized
        return underutilized
    
    def identify_overused(self, threshold: float = 80) -> pd.DataFrame:
        """Identify overused equipment"""
        equipment_util = self.df.groupby('equipment_id').agg({
            'equipment_type': 'first',
            'department': 'first',
            'utilization_rate': 'mean'
        }).round(2)
        
        overused = equipment_util[equipment_util['utilization_rate'] > threshold]
        overused = overused.sort_values('utilization_rate', ascending=False)
        
        self.analysis_results['overused'] = overused
        return overused
    
    def maintenance_risk_assessment(self) -> pd.DataFrame:
        """Assess maintenance risks"""
        risk_analysis = self.df[self.df['maintenance_flag'] >= 1].groupby('equipment_id').agg({
            'equipment_type': 'first',
            'department': 'first',
            'maintenance_flag': 'max',
            'days_since_maintenance': 'mean'
        }).round(0)
        
        risk_analysis = risk_analysis.sort_values('maintenance_flag', ascending=False)
        
        self.analysis_results['maintenance_risk'] = risk_analysis
        return risk_analysis
    
    def department_analysis(self) -> pd.DataFrame:
        """Analyze by department"""
        dept_stats = self.df.groupby('department').agg({
            'utilization_rate': 'mean',
            'daily_cost': 'sum',
            'equipment_id': 'nunique'
        }).round(2)
        
        dept_stats.columns = ['Avg Util %', 'Total Cost $', 'Unique Equipment']
        dept_stats = dept_stats.sort_values('Total Cost $', ascending=False)
        
        self.analysis_results['by_department'] = dept_stats
        return dept_stats
    
    def get_all_results(self) -> Dict:
        """Get all analysis results"""
        return self.analysis_results

# ==========================================
# CLASS 6: ReportGenerator
# ==========================================

class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    def __init__(self, analyzer: EquipmentAnalyzer):
        self.analyzer = analyzer
    
    def generate_text_report(self) -> str:
        """Generate formatted text report"""
        report = []
        report.append("=" * 70)
        report.append("EQUIPMENT UTILIZATION & QUALITY MONITORING REPORT")
        report.append("=" * 70)
        
        # Overall stats
        stats = self.analyzer.overall_statistics()
        report.append("\n1. OVERALL STATISTICS")
        report.append("-" * 70)
        report.append(f"Mean Utilization Rate: {stats['mean_utilization']:.2f}%")
        report.append(f"Total Usage Hours: {stats['total_usage_hours']:.2f}")
        report.append(f"Total Operational Cost: ${stats['total_cost']:.2f}")
        report.append(f"Records Analyzed: {stats['total_records']}")
        
        # Equipment type analysis
        report.append("\n2. UTILIZATION BY EQUIPMENT TYPE")
        report.append("-" * 70)
        type_analysis = self.analyzer.equipment_type_analysis()
        report.append(type_analysis.to_string())
        
        # Underutilized
        report.append("\n3. UNDERUTILIZED EQUIPMENT (<30% utilization)")
        report.append("-" * 70)
        underutil = self.analyzer.identify_underutilized()
        report.append(f"Total underutilized items: {len(underutil)}")
        report.append(underutil.head(5).to_string())
        
        # Overused
        report.append("\n4. OVERUSED EQUIPMENT (>80% utilization)")
        report.append("-" * 70)
        overused = self.analyzer.identify_overused()
        report.append(f"Total overused items: {len(overused)}")
        report.append(overused.head(5).to_string())
        
        # Maintenance risks
        report.append("\n5. MAINTENANCE RISK ASSESSMENT")
        report.append("-" * 70)
        maint_risk = self.analyzer.maintenance_risk_assessment()
        report.append(f"Equipment requiring attention: {len(maint_risk)}")
        report.append(maint_risk.head(5).to_string())
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """Save report to file"""
        report = self.generate_text_report()
        with open(filepath, 'w') as f:
            f.write(report)

# ==========================================
# MAIN PIPELINE ORCHESTRATOR
# ==========================================

class EquipmentMonitoringPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.loader = None
        self.cleaner = None
        self.validator = None
        self.engineer = None
        self.analyzer = None
        self.reporter = None
        self.df = None
    
    def run(self, input_file: str, output_file: str = None):
        """Execute complete pipeline"""
        print("\n" + "=" * 70)
        print("EQUIPMENT MONITORING SYSTEM - OOP PIPELINE")
        print("=" * 70 + "\n")
        
        # Step 1: Load
        self.loader = DataLoader()
        self.df = self.loader.load_from_csv(input_file)
        
        # Step 2: Clean
        print("\n" + "=" * 70)
        print("DATA CLEANING")
        print("=" * 70)
        self.cleaner = DataCleaner(self.df)
        self.df = (self.cleaner
                   .remove_duplicates()
                   .standardize_text_columns(['operational_status'])
                   .handle_missing_values({
                       'usage_hours': 'group_median',
                       'last_maintenance_date': 'forward_fill'
                   })
                   .convert_date_columns(['usage_date', 'last_maintenance_date'])
                   .get_cleaned_data())
        
        # Step 3: Validate
        print("\n" + "=" * 70)
        print("DATA VALIDATION")
        print("=" * 70)
        self.validator = DataValidator(self.df)
        self.df = (self.validator
                   .validate_usage_capacity()
                   .validate_categorical('operational_status', 
                                        ['Active', 'Idle', 'Under Maintenance', 'Faulty'])
                   .validate_future_dates('usage_date')
                   .get_validated_data())
        
        if not self.validator.is_valid():
            print("⚠ Validation issues found:")
            for issue in self.validator.get_issues():
                print(f"  - {issue}")
        else:
            print("✓ All validation checks passed")
        
        # Step 4: Feature Engineering
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING")
        print("=" * 70)
        self.engineer = FeatureEngineer(self.df)
        self.df = (self.engineer
                   .create_utilization_rate()
                   .create_idle_indicator()
                   .create_overused_indicator()
                   .create_maintenance_features()
                   .create_cost_features()
                   .create_temporal_features()
                   .get_engineered_data())
        
        print(f"✓ Created {len(self.engineer.get_features())} new features:")
        for feat in self.engineer.get_features():
            print(f"  - {feat}")
        
        # Step 5: Analysis
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        self.analyzer = EquipmentAnalyzer(self.df)
        self.analyzer.overall_statistics()
        self.analyzer.equipment_type_analysis()
        self.analyzer.identify_underutilized()
        self.analyzer.identify_overused()
        self.analyzer.maintenance_risk_assessment()
        self.analyzer.department_analysis()
        
        # Step 6: Generate Report
        self.reporter = ReportGenerator(self.analyzer)
        print(self.reporter.generate_text_report())
        
        # Save processed data
        if output_file:
            self.df.to_csv(output_file, index=False)
            print(f"\n✓ Processed data saved to: {output_file}")
        
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        
        return self.df

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    pipeline = EquipmentMonitoringPipeline()
    processed_data = pipeline.run(
        input_file='equipment_usage_data.csv',
        output_file='equipment_data_processed_oop.csv'
    )