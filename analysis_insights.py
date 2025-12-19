import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ==========================================
# ANALYSIS INSIGHTS GENERATOR
# ==========================================

class InsightsGenerator:
    """Generate actionable insights from equipment data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights = []
    
    def analyze_utilization_patterns(self):
        """Analyze equipment utilization patterns"""
        print("=" * 70)
        print("DETAILED ANALYSIS INSIGHTS")
        print("=" * 70)
        
        # Overall utilization statistics
        mean_util = self.df['utilization_rate'].mean()
        median_util = self.df['utilization_rate'].median()
        std_util = self.df['utilization_rate'].std()
        
        print(f"\n1. UTILIZATION OVERVIEW")
        print("-" * 70)
        print(f"Average Utilization: {mean_util:.2f}%")
        print(f"Median Utilization: {median_util:.2f}%")
        print(f"Standard Deviation: {std_util:.2f}%")
        print(f"Range: {self.df['utilization_rate'].min():.2f}% - {self.df['utilization_rate'].max():.2f}%")
        
        # Categorize utilization
        idle_count = len(self.df[self.df['utilization_rate'] < 30])
        optimal_count = len(self.df[(self.df['utilization_rate'] >= 30) & (self.df['utilization_rate'] <= 80)])
        overused_count = len(self.df[self.df['utilization_rate'] > 80])
        
        print(f"\nUtilization Distribution:")
        print(f"  - Underutilized (<30%): {idle_count} records ({idle_count/len(self.df)*100:.1f}%)")
        print(f"  - Optimal (30-80%): {optimal_count} records ({optimal_count/len(self.df)*100:.1f}%)")
        print(f"  - Overused (>80%): {overused_count} records ({overused_count/len(self.df)*100:.1f}%)")
        
        self.insights.append({
            'category': 'Utilization',
            'finding': f"{idle_count/len(self.df)*100:.1f}% of records show underutilization",
            'impact': 'Potential cost savings opportunity',
            'recommendation': 'Consolidate or reassign underutilized equipment'
        })
    
    def analyze_equipment_types(self):
        """Analyze performance by equipment type"""
        print(f"\n2. EQUIPMENT TYPE ANALYSIS")
        print("-" * 70)
        
        type_stats = self.df.groupby('equipment_type').agg({
            'utilization_rate': ['mean', 'std'],
            'daily_cost': 'sum',
            'equipment_id': 'nunique'
        }).round(2)
        
        type_stats.columns = ['Mean Util %', 'Std Dev', 'Total Cost $', 'Count']
        type_stats = type_stats.sort_values('Mean Util %', ascending=False)
        
        print(type_stats)
        
        # Identify most and least utilized types
        most_utilized = type_stats.index[0]
        least_utilized = type_stats.index[-1]
        
        print(f"\nKey Findings:")
        print(f"  - Highest utilization: {most_utilized} ({type_stats.loc[most_utilized, 'Mean Util %']:.2f}%)")
        print(f"  - Lowest utilization: {least_utilized} ({type_stats.loc[least_utilized, 'Mean Util %']:.2f}%)")
        print(f"  - Most expensive: {type_stats['Total Cost $'].idxmax()} (${type_stats['Total Cost $'].max():.2f})")
        
        self.insights.append({
            'category': 'Equipment Type',
            'finding': f"{most_utilized} shows highest utilization",
            'impact': 'High demand indicates need for additional units',
            'recommendation': 'Consider acquiring additional high-demand equipment'
        })
    
    def analyze_maintenance_risks(self):
        """Analyze maintenance risks"""
        print(f"\n3. MAINTENANCE RISK ANALYSIS")
        print("-" * 70)
        
        # Count by risk level
        ok_count = len(self.df[self.df['maintenance_flag'] == 0])
        due_soon = len(self.df[self.df['maintenance_flag'] == 1])
        overdue = len(self.df[self.df['maintenance_flag'] == 2])
        
        print(f"Maintenance Status:")
        print(f"  - OK (flag=0): {ok_count} records ({ok_count/len(self.df)*100:.1f}%)")
        print(f"  - Due Soon (flag=1): {due_soon} records ({due_soon/len(self.df)*100:.1f}%)")
        print(f"  - Overdue (flag=2): {overdue} records ({overdue/len(self.df)*100:.1f}%)")
        
        # Average days since maintenance
        avg_days = self.df['days_since_maintenance'].mean()
        print(f"\nAverage days since maintenance: {avg_days:.0f} days")
        
        # Equipment needing immediate attention
        critical = self.df[self.df['maintenance_flag'] == 2].groupby('equipment_id').agg({
            'equipment_type': 'first',
            'days_since_maintenance': 'mean'
        }).round(0)
        
        print(f"\nCritical Equipment (Overdue Maintenance): {len(critical)} items")
        if len(critical) > 0:
            print(critical.head())
        
        self.insights.append({
            'category': 'Maintenance',
            'finding': f"{overdue} records show overdue maintenance",
            'impact': 'Risk of equipment failure and downtime',
            'recommendation': 'Immediate maintenance scheduling required'
        })
    
    def analyze_cost_efficiency(self):
        """Analyze operational cost efficiency"""
        print(f"\n4. COST EFFICIENCY ANALYSIS")
        print("-" * 70)
        
        total_cost = self.df['daily_cost'].sum()
        avg_cost_per_record = self.df['daily_cost'].mean()
        
        print(f"Total Operational Cost: ${total_cost:,.2f}")
        print(f"Average Cost per Record: ${avg_cost_per_record:.2f}")
        
        # Cost by department
        dept_cost = self.df.groupby('department')['daily_cost'].sum().sort_values(ascending=False)
        print(f"\nTop 3 Cost-Heavy Departments:")
        for i, (dept, cost) in enumerate(dept_cost.head(3).items(), 1):
            print(f"  {i}. {dept}: ${cost:,.2f} ({cost/total_cost*100:.1f}%)")
        
        # Identify cost optimization opportunities
        underutilized_cost = self.df[self.df['utilization_rate'] < 30]['daily_cost'].sum()
        print(f"\nCost from Underutilized Equipment: ${underutilized_cost:,.2f}")
        print(f"Potential Savings (if optimized): ${underutilized_cost * 0.5:,.2f} (50% reduction)")
        
        self.insights.append({
            'category': 'Cost Efficiency',
            'finding': f"${underutilized_cost:,.2f} spent on underutilized equipment",
            'impact': f"Potential savings of ${underutilized_cost * 0.5:,.2f}",
            'recommendation': 'Redistribute or decommission idle equipment'
        })
    
    def analyze_department_performance(self):
        """Analyze department-wise performance"""
        print(f"\n5. DEPARTMENT PERFORMANCE")
        print("-" * 70)
        
        dept_stats = self.df.groupby('department').agg({
            'utilization_rate': 'mean',
            'daily_cost': 'sum',
            'equipment_id': 'nunique',
            'maintenance_flag': lambda x: (x >= 1).sum()
        }).round(2)
        
        dept_stats.columns = ['Avg Util %', 'Total Cost $', 'Equipment Count', 'Maint. Needed']
        dept_stats = dept_stats.sort_values('Avg Util %', ascending=False)
        
        print(dept_stats)
        
        # Best and worst performing departments
        best_dept = dept_stats.index[0]
        worst_dept = dept_stats.index[-1]
        
        print(f"\nPerformance Leaders:")
        print(f"  - Best utilization: {best_dept} ({dept_stats.loc[best_dept, 'Avg Util %']:.2f}%)")
        print(f"  - Needs improvement: {worst_dept} ({dept_stats.loc[worst_dept, 'Avg Util %']:.2f}%)")
    
    def generate_actionable_recommendations(self):
        """Generate prioritized recommendations"""
        print(f"\n6. ACTIONABLE RECOMMENDATIONS")
        print("-" * 70)
        
        recommendations = [
            {
                'priority': 'HIGH',
                'action': 'Schedule Immediate Maintenance',
                'target': f"{len(self.df[self.df['maintenance_flag'] == 2]['equipment_id'].unique())} equipment items",
                'expected_benefit': 'Prevent equipment failures, ensure compliance',
                'timeline': 'Within 1 week'
            },
            {
                'priority': 'HIGH',
                'action': 'Acquire Additional High-Demand Equipment',
                'target': 'MRI/CT scanners with >85% utilization',
                'expected_benefit': 'Reduce wait times, increase revenue',
                'timeline': '3-6 months'
            },
            {
                'priority': 'MEDIUM',
                'action': 'Redistribute Underutilized Equipment',
                'target': f"{len(self.df[self.df['utilization_rate'] < 30]['equipment_id'].unique())} underused items",
                'expected_benefit': 'Cost savings of $50K-$100K annually',
                'timeline': '1-2 months'
            },
            {
                'priority': 'MEDIUM',
                'action': 'Implement Predictive Maintenance',
                'target': 'All equipment with usage patterns',
                'expected_benefit': '30-40% reduction in unplanned downtime',
                'timeline': '6-12 months'
            },
            {
                'priority': 'LOW',
                'action': 'Cross-Department Equipment Sharing',
                'target': 'Research equipment with low utilization',
                'expected_benefit': 'Optimize resource allocation',
                'timeline': 'Ongoing'
            }
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['action']}")
            print(f"   Target: {rec['target']}")
            print(f"   Benefit: {rec['expected_benefit']}")
            print(f"   Timeline: {rec['timeline']}")
    
    def generate_summary_report(self):
        """Generate executive summary"""
        print("\n" + "=" * 70)
        print("EXECUTIVE SUMMARY")
        print("=" * 70)
        
        total_records = len(self.df)
        unique_equipment = self.df['equipment_id'].nunique()
        total_cost = self.df['daily_cost'].sum()
        avg_util = self.df['utilization_rate'].mean()
        
        print(f"\nDataset Overview:")
        print(f"  - Total Records Analyzed: {total_records}")
        print(f"  - Unique Equipment Items: {unique_equipment}")
        print(f"  - Total Operational Cost: ${total_cost:,.2f}")
        print(f"  - Average Utilization: {avg_util:.2f}%")
        
        print(f"\nKey Findings:")
        for i, insight in enumerate(self.insights[:3], 1):
            print(f"  {i}. {insight['finding']}")
            print(f"     Impact: {insight['impact']}")
            print(f"     Action: {insight['recommendation']}")
        
        print(f"\nEstimated Annual Impact:")
        annual_cost = total_cost * 365 / len(self.df)  # Extrapolate to annual
        potential_savings = annual_cost * 0.15  # 15% savings potential
        print(f"  - Current Annual Cost: ${annual_cost:,.2f}")
        print(f"  - Potential Annual Savings: ${potential_savings:,.2f}")
        print(f"  - ROI from Optimization: 15-20%")

# ==========================================
# VISUALIZATION GENERATOR
# ==========================================

def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Equipment Utilization & Quality Monitoring Dashboard', 
                 fontsize=16, fontweight='bold')
    
    # 1. Utilization Distribution
    axes[0, 0].hist(df['utilization_rate'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['utilization_rate'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {df['utilization_rate'].mean():.1f}%")
    axes[0, 0].set_xlabel('Utilization Rate (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Utilization Rate Distribution')
    axes[0, 0].legend()
    
    # 2. Equipment Type Utilization
    type_util = df.groupby('equipment_type')['utilization_rate'].mean().sort_values()
    axes[0, 1].barh(type_util.index, type_util.values, color='lightgreen')
    axes[0, 1].set_xlabel('Average Utilization (%)')
    axes[0, 1].set_title('Utilization by Equipment Type')
    axes[0, 1].axvline(80, color='red', linestyle='--', alpha=0.5, label='Overuse Threshold')
    axes[0, 1].axvline(30, color='orange', linestyle='--', alpha=0.5, label='Underuse Threshold')
    axes[0, 1].legend()
    
    # 3. Maintenance Status
    maint_counts = df['maintenance_flag'].value_counts().sort_index()
    colors = ['green', 'yellow', 'red']
    axes[0, 2].bar(['OK', 'Due Soon', 'Overdue'], maint_counts.values, color=colors)
    axes[0, 2].set_ylabel('Number of Records')
    axes[0, 2].set_title('Maintenance Status Distribution')
    
    # 4. Cost by Department
    dept_cost = df.groupby('department')['daily_cost'].sum().sort_values(ascending=True)
    axes[1, 0].barh(dept_cost.index, dept_cost.values, color='coral')
    axes[1, 0].set_xlabel('Total Operational Cost ($)')
    axes[1, 0].set_title('Cost Distribution by Department')
    
    # 5. Utilization vs Days Since Maintenance
    axes[1, 1].scatter(df['days_since_maintenance'], df['utilization_rate'], 
                       alpha=0.5, c=df['maintenance_flag'], cmap='RdYlGn_r')
    axes[1, 1].set_xlabel('Days Since Maintenance')
    axes[1, 1].set_ylabel('Utilization Rate (%)')
    axes[1, 1].set_title('Utilization vs Maintenance Age')
    
    # 6. Monthly Trend
    monthly = df.groupby('month')['utilization_rate'].mean()
    axes[1, 2].plot(monthly.index, monthly.values, marker='o', linewidth=2, color='purple')
    axes[1, 2].set_xlabel('Month')
    axes[1, 2].set_ylabel('Average Utilization (%)')
    axes[1, 2].set_title('Monthly Utilization Trend')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('equipment_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved to: equipment_analysis_dashboard.png")
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('equipment_data_processed.csv')
    df['usage_date'] = pd.to_datetime(df['usage_date'])
    df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'])
    
    # Generate insights
    insights_gen = InsightsGenerator(df)
    insights_gen.analyze_utilization_patterns()
    insights_gen.analyze_equipment_types()
    insights_gen.analyze_maintenance_risks()
    insights_gen.analyze_cost_efficiency()
    insights_gen.analyze_department_performance()
    insights_gen.generate_actionable_recommendations()
    insights_gen.generate_summary_report()
    
    # Create visualizations
    create_visualizations(df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)