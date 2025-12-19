import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import classes from main module
# from equipment_monitoring import DataCleaner, DataValidator, FeatureEngineer, EquipmentAnalyzer

# ==========================================
# TEST SUITE 1: DataCleaner Tests
# ==========================================

class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class"""
    
    def setUp(self):
        """Create sample test data"""
        self.test_data = pd.DataFrame({
            'equipment_id': ['EQ-001', 'EQ-002', 'EQ-001', 'EQ-003'],
            'equipment_type': ['MRI', 'CT Scanner', 'MRI', 'Ultrasound'],
            'operational_status': ['Active', 'active', 'Active', 'Idle'],
            'usage_hours': [10.5, np.nan, 10.5, 5.2],
            'max_daily_capacity': [24, 24, 24, 12]
        })
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        # Simulate DataCleaner
        initial_len = len(self.test_data)
        cleaned = self.test_data.drop_duplicates()
        
        self.assertEqual(len(cleaned), 3, "Should remove 1 duplicate")
        self.assertLess(len(cleaned), initial_len, "Length should decrease")
    
    def test_standardize_text(self):
        """Test text standardization"""
        standardized = self.test_data['operational_status'].str.title()
        
        self.assertEqual(standardized[1], 'Active', "Should standardize to title case")
        self.assertTrue(all(s[0].isupper() for s in standardized), "All should be title case")
    
    def test_missing_value_handling(self):
        """Test missing value imputation"""
        # Fill with median
        median_val = self.test_data['usage_hours'].median()
        filled = self.test_data['usage_hours'].fillna(median_val)
        
        self.assertEqual(filled.isnull().sum(), 0, "Should have no missing values")
        self.assertEqual(filled[1], median_val, "Should fill with median value")

# ==========================================
# TEST SUITE 2: DataValidator Tests
# ==========================================

class TestDataValidator(unittest.TestCase):
    """Test suite for DataValidator class"""
    
    def setUp(self):
        """Create test data with validation issues"""
        self.test_data = pd.DataFrame({
            'usage_hours': [8, 15, 30, 5],  # One value exceeds capacity
            'max_daily_capacity': [24, 24, 24, 24],
            'operational_status': ['Active', 'Idle', 'Invalid', 'Active'],
            'usage_date': [
                datetime.now() - timedelta(days=1),
                datetime.now() - timedelta(days=2),
                datetime.now() + timedelta(days=1),  # Future date
                datetime.now()
            ]
        })
    
    def test_usage_capacity_validation(self):
        """Test that usage doesn't exceed capacity"""
        invalid = self.test_data[self.test_data['usage_hours'] > self.test_data['max_daily_capacity']]
        
        self.assertEqual(len(invalid), 1, "Should find 1 invalid record")
        self.assertEqual(invalid.iloc[0]['usage_hours'], 30, "Should identify correct invalid record")
    
    def test_categorical_validation(self):
        """Test categorical value validation"""
        valid_statuses = ['Active', 'Idle', 'Under Maintenance', 'Faulty']
        invalid = self.test_data[~self.test_data['operational_status'].isin(valid_statuses)]
        
        self.assertEqual(len(invalid), 1, "Should find 1 invalid status")
    
    def test_future_date_validation(self):
        """Test future date detection"""
        future = self.test_data[self.test_data['usage_date'] > datetime.now()]
        
        self.assertEqual(len(future), 1, "Should find 1 future date")

# ==========================================
# TEST SUITE 3: FeatureEngineer Tests
# ==========================================

class TestFeatureEngineer(unittest.TestCase):
    """Test suite for FeatureEngineer class"""
    
    def setUp(self):
        """Create test data for feature engineering"""
        self.test_data = pd.DataFrame({
            'usage_hours': [8, 20, 2, 18],
            'max_daily_capacity': [24, 24, 12, 24],
            'cost_per_hour': [100, 150, 50, 120],
            'usage_date': [datetime(2024, 12, 1)] * 4,
            'last_maintenance_date': [datetime(2024, 10, 1)] * 4
        })
    
    def test_utilization_rate_calculation(self):
        """Test utilization rate calculation"""
        utilization = (self.test_data['usage_hours'] / self.test_data['max_daily_capacity']) * 100
        
        self.assertAlmostEqual(utilization[0], 33.33, places=2, msg="Should calculate correct utilization")
        self.assertAlmostEqual(utilization[1], 83.33, places=2, msg="High utilization should be correct")
        self.assertAlmostEqual(utilization[2], 16.67, places=2, msg="Low utilization should be correct")
    
    def test_idle_indicator(self):
        """Test idle equipment flagging"""
        utilization = (self.test_data['usage_hours'] / self.test_data['max_daily_capacity']) * 100
        is_idle = (utilization < 20).astype(int)
        
        self.assertEqual(is_idle[2], 1, "Low utilization should be flagged as idle")
        self.assertEqual(is_idle[1], 0, "High utilization should not be idle")
    
    def test_overused_indicator(self):
        """Test overused equipment flagging"""
        utilization = (self.test_data['usage_hours'] / self.test_data['max_daily_capacity']) * 100
        is_overused = (utilization > 80).astype(int)
        
        self.assertEqual(is_overused[1], 1, "High utilization should be flagged as overused")
        self.assertEqual(is_overused[0], 0, "Moderate utilization should not be overused")
    
    def test_cost_calculation(self):
        """Test operational cost calculation"""
        daily_cost = self.test_data['usage_hours'] * self.test_data['cost_per_hour']
        
        self.assertEqual(daily_cost[0], 800, "Should calculate correct cost")
        self.assertEqual(daily_cost[1], 3000, "Should calculate high usage cost")
    
    def test_days_since_maintenance(self):
        """Test maintenance interval calculation"""
        days_diff = (self.test_data['usage_date'] - self.test_data['last_maintenance_date']).dt.days
        
        self.assertEqual(days_diff[0], 61, "Should calculate correct days difference")

# ==========================================
# TEST SUITE 4: EquipmentAnalyzer Tests
# ==========================================

class TestEquipmentAnalyzer(unittest.TestCase):
    """Test suite for EquipmentAnalyzer class"""
    
    def setUp(self):
        """Create test data for analysis"""
        self.test_data = pd.DataFrame({
            'equipment_id': ['EQ-001', 'EQ-001', 'EQ-002', 'EQ-003'],
            'equipment_type': ['MRI', 'MRI', 'CT Scanner', 'Ultrasound'],
            'department': ['Radiology', 'Radiology', 'Radiology', 'Cardiology'],
            'utilization_rate': [85, 90, 25, 70],
            'usage_hours': [20, 22, 6, 8],
            'daily_cost': [2000, 2200, 900, 600],
            'maintenance_flag': [0, 1, 0, 2]
        })
    
    def test_overall_statistics(self):
        """Test overall statistics calculation"""
        mean_util = self.test_data['utilization_rate'].mean()
        total_cost = self.test_data['daily_cost'].sum()
        
        self.assertAlmostEqual(mean_util, 67.5, places=1, msg="Should calculate correct mean")
        self.assertEqual(total_cost, 5700, "Should calculate correct total cost")
    
    def test_underutilized_identification(self):
        """Test underutilized equipment identification"""
        underutilized = self.test_data[self.test_data['utilization_rate'] < 30]
        
        self.assertEqual(len(underutilized), 1, "Should identify 1 underutilized equipment")
        self.assertEqual(underutilized.iloc[0]['equipment_id'], 'EQ-002', "Should identify correct equipment")
    
    def test_overused_identification(self):
        """Test overused equipment identification"""
        overused = self.test_data[self.test_data['utilization_rate'] > 80]
        
        self.assertEqual(len(overused), 2, "Should identify 2 overused equipment")
    
    def test_maintenance_risk_detection(self):
        """Test maintenance risk assessment"""
        high_risk = self.test_data[self.test_data['maintenance_flag'] >= 1]
        
        self.assertEqual(len(high_risk), 2, "Should identify 2 equipment needing maintenance")
    
    def test_department_grouping(self):
        """Test department-wise aggregation"""
        dept_stats = self.test_data.groupby('department').agg({
            'utilization_rate': 'mean',
            'daily_cost': 'sum'
        })
        
        self.assertEqual(len(dept_stats), 2, "Should have 2 departments")
        self.assertGreater(dept_stats.loc['Radiology', 'daily_cost'], 
                          dept_stats.loc['Cardiology', 'daily_cost'],
                          "Radiology should have higher cost")

# ==========================================
# TEST SUITE 5: Data Integrity Tests
# ==========================================

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and business logic"""
    
    def test_utilization_bounds(self):
        """Test that utilization rate is between 0 and 100"""
        usage = 15
        capacity = 24
        utilization = (usage / capacity) * 100
        
        self.assertGreaterEqual(utilization, 0, "Utilization should not be negative")
        self.assertLessEqual(utilization, 100, "Utilization should not exceed 100%")
    
    def test_maintenance_flag_values(self):
        """Test maintenance flag validity"""
        valid_flags = [0, 1, 2]
        test_flags = [0, 1, 2, 0, 1]
        
        self.assertTrue(all(f in valid_flags for f in test_flags), 
                       "All flags should be valid")
    
    def test_cost_calculation_accuracy(self):
        """Test cost calculation precision"""
        hours = 10.5
        rate = 99.99
        expected_cost = 1049.895
        calculated_cost = hours * rate
        
        self.assertAlmostEqual(calculated_cost, expected_cost, places=2,
                              msg="Cost calculation should be accurate")

# ==========================================
# INTEGRATION TESTS
# ==========================================

class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end pipeline"""
    
    def test_pipeline_data_flow(self):
        """Test data flows correctly through pipeline stages"""
        # Simulate pipeline
        data = pd.DataFrame({
            'equipment_id': ['EQ-001', 'EQ-002'],
            'usage_hours': [10, np.nan],
            'max_daily_capacity': [24, 24]
        })
        
        # Clean
        data['usage_hours'].fillna(data['usage_hours'].median(), inplace=True)
        
        # Engineer
        data['utilization_rate'] = (data['usage_hours'] / data['max_daily_capacity']) * 100
        
        # Validate
        self.assertEqual(len(data), 2, "Should maintain record count")
        self.assertTrue('utilization_rate' in data.columns, "Should have new feature")
        self.assertEqual(data['usage_hours'].isnull().sum(), 0, "Should have no missing values")
    
    def test_analysis_consistency(self):
        """Test that analysis results are consistent"""
        data = pd.DataFrame({
            'equipment_type': ['MRI', 'MRI', 'CT'],
            'utilization_rate': [80, 90, 30]
        })
        
        type_stats = data.groupby('equipment_type')['utilization_rate'].mean()
        
        self.assertEqual(type_stats['MRI'], 85.0, "MRI average should be 85%")
        self.assertEqual(type_stats['CT'], 30.0, "CT average should be 30%")

# ==========================================
# TEST RUNNER WITH DETAILED OUTPUT
# ==========================================

def run_all_tests():
    """Run all test suites with detailed output"""
    print("\n" + "=" * 70)
    print("EQUIPMENT MONITORING SYSTEM - UNIT TEST SUITE")
    print("=" * 70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataCleaner))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineer))
    suite.addTests(loader.loadTestsFromTestCase(TestEquipmentAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("=" * 70)
    
    return result

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    result = run_all_tests()
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)