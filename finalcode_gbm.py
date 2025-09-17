import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import pickle
import optuna
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import pickle
import optuna
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


    
import pandas as pd

class AdvancedPedestrianPredictor:
    def __init__(self, use_gpu=True,plot_dir="model_plots",save_plots=True):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.use_gpu = use_gpu
        self.feature_cols = None
        self.location_clusters = None
        self.location_stats = None
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.gpu_available = False

                # Create plots directory
        if self.save_plots and not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            print(f"Created directory: {self.plot_dir}")

        if self.use_gpu:
            self.gpu_available = self._test_gpu_availability()
            if not self.gpu_available:
                print("⚠️  GPU not available or incompatible, falling back to CPU")
                self.use_gpu = False

    def _test_gpu_availability(self):
        """Safely test if GPU acceleration is available"""
        try:
            # Create small test dataset
            X_test = np.random.randn(100, 5)
            y_test = np.random.randn(100)
            
            train_data = lgb.Dataset(X_test, label=y_test)
            
            # Test GPU parameters
            gpu_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'device_type': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'num_boost_round': 1,
                'verbose': -1,
                'force_col_wise': True,
                'num_leaves': 31,
                'learning_rate': 0.1
            }
            
            # Try training with GPU
            test_model = lgb.train(
                gpu_params,
                train_data,
                num_boost_round=1,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Test prediction
            test_model.predict(X_test[:10])
            
            print("✅ GPU acceleration available and working")
            return True
            
        except Exception as e:
            print(f"❌ GPU test failed: {str(e)}")
            return False
            
    def load_and_preprocess_data(self, df):
        """Load and preprocess the dataset with advanced feature engineering"""
        print("Loading dataset...")
        
        # Handle missing values smartly
        df['dir1'] = df['dir1'].fillna(df['dir1'].median())
        df['dir2'] = df['dir2'].fillna(df['dir2'].median())
        
        # Remove rows with missing essential data
        df = df.dropna(subset=['latitude', 'longitude', 'count'])
        
        # Remove extreme outliers more carefully (99th percentile)
        count_99th = df['count'].quantile(0.99)
        df = df[df['count'] <= count_99th]
        

        
    
        print(f"Dataset loaded: {len(df)} records after cleaning")
        return df
    
        
    def create_data_exploration_plots(self, df):
        """Create comprehensive data exploration visualizations"""
        print("Creating data exploration plots...")
        
        # 1. Distribution of pedestrian counts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Count distribution
        axes[0,0].hist(df['count'], bins=50, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Distribution of Pedestrian Counts')
        axes[0,0].set_xlabel('Count')
        axes[0,0].set_ylabel('Frequency')
        
        # Log-transformed count distribution
        axes[0,1].hist(np.log1p(df['count']), bins=50, alpha=0.7, color='lightcoral')
        axes[0,1].set_title('Distribution of Log-Transformed Counts')
        axes[0,1].set_xlabel('Log(Count + 1)')
        axes[0,1].set_ylabel('Frequency')
        
        # Box plot by hour
        hourly_data = [df[df['hour'] == h]['count'].values for h in range(24)]
        axes[1,0].boxplot(hourly_data, labels=range(24))
        axes[1,0].set_title('Count Distribution by Hour of Day')
        axes[1,0].set_xlabel('Hour')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Box plot by day of week
        dow_data = [df[df['dow'] == d]['count'].values for d in range(7)]
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1,1].boxplot(dow_data, labels=dow_labels)
        axes[1,1].set_title('Count Distribution by Day of Week')
        axes[1,1].set_xlabel('Day of Week')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/01_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average count by hour
        hourly_avg = df.groupby('hour')['count'].mean()
        axes[0,0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
        axes[0,0].set_title('Average Pedestrian Count by Hour')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Average Count')
        axes[0,0].grid(True, alpha=0.3)
        
        # Average count by day of week
        dow_avg = df.groupby('dow')['count'].mean()
        axes[0,1].bar(dow_labels, dow_avg.values, color='lightgreen')
        axes[0,1].set_title('Average Pedestrian Count by Day of Week')
        axes[0,1].set_xlabel('Day of Week')
        axes[0,1].set_ylabel('Average Count')
        
        # Heatmap: Hour vs Day of Week
        pivot_data = df.groupby(['hour', 'dow'])['count'].mean().unstack()
        im = axes[1,0].imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        axes[1,0].set_title('Average Count: Hour vs Day of Week')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Hour of Day')
        axes[1,0].set_xticks(range(7))
        axes[1,0].set_xticklabels(dow_labels)
        plt.colorbar(im, ax=axes[1,0])
        
        # Monthly trends
        monthly_avg = df.groupby('month')['count'].mean()
        axes[1,1].plot(monthly_avg.index, monthly_avg.values, marker='s', linewidth=2, color='purple')
        axes[1,1].set_title('Average Pedestrian Count by Month')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_ylabel('Average Count')
        axes[1,1].set_xticks(range(1, 13))
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/02_temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Spatial analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot of locations colored by average count
        location_avg = df.groupby(['latitude', 'longitude'])['count'].mean().reset_index()
        scatter = axes[0].scatter(location_avg['longitude'], location_avg['latitude'], 
                                c=location_avg['count'], cmap='viridis', s=50, alpha=0.7)
        axes[0].set_title('Sensor Locations Colored by Average Count')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(scatter, ax=axes[0])
        
        # Count distribution by sensor
        top_locations = df.groupby('sensor_name')['count'].mean().nlargest(10)
        axes[1].barh(range(len(top_locations)), top_locations.values, color='coral')
        axes[1].set_yticks(range(len(top_locations)))
        axes[1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                               for name in top_locations.index], fontsize=9)
        axes[1].set_title('Top 10 Locations by Average Count')
        axes[1].set_xlabel('Average Count')
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/03_spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Data exploration plots saved!")

    def create_model_summary_report(self, results):
        """Create a comprehensive model summary report"""
        print("Creating model summary report...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Model metrics summary (text)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        metrics_text = f"""
        MODEL PERFORMANCE SUMMARY
        
        Mean Absolute Error (MAE): {results['mae']:.2f}
        Root Mean Square Error (RMSE): {results['rmse']:.2f}
        R-squared (R²): {results['r2']:.4f}
        
        Mean Actual Count: {results['y_test'].mean():.2f}
        Mean Predicted Count: {results['y_pred'].mean():.2f}
        
        Best Hyperparameters:
        {chr(10).join([f"  • {k}: {v}" for k, v in results['best_params'].items()])}
        """
        
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Top 10 features (horizontal bar)
        ax2 = fig.add_subplot(gs[1, :2])
        top_10_features = results['feature_importance'].head(10)
        bars = ax2.barh(range(len(top_10_features)), top_10_features['importance'], 
                        color='steelblue')
        ax2.set_yticks(range(len(top_10_features)))
        ax2.set_yticklabels(top_10_features['feature'])
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 10 Most Important Features')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.0f}', ha='left', va='center', fontsize=9)
        
        # Actual vs Predicted mini plot
        ax3 = fig.add_subplot(gs[1, 2])
        max_val = max(results['y_test'].max(), results['y_pred'].max())
        ax3.scatter(results['y_test'], results['y_pred'], alpha=0.6, s=15, color='coral')
        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        ax3.set_xlabel('Actual')
        ax3.set_ylabel('Predicted')
        ax3.set_title(f'Actual vs Predicted\n(R² = {results["r2"]:.3f})')
        ax3.grid(True, alpha=0.3)
        
        # Error distribution
        ax4 = fig.add_subplot(gs[2, 0])
        residuals = results['y_test'] - results['y_pred']
        ax4.hist(residuals, bins=30, alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Residuals Distribution')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Feature importance pie chart (top 8)
        ax5 = fig.add_subplot(gs[2, 1])
        top_8_features = results['feature_importance'].head(8)
        other_importance = results['feature_importance']['importance'][8:].sum()
        
        pie_data = list(top_8_features['importance']) + [other_importance]
        pie_labels = list(top_8_features['feature']) + ['Others']
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
        ax5.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=colors, 
                textprops={'fontsize': 8})
        ax5.set_title('Feature Importance Distribution')
        
        # Performance by count range
        ax6 = fig.add_subplot(gs[2, 2])
        
        # Create error analysis by count bins
        error_df = pd.DataFrame({
            'actual': results['y_test'].values,
            'predicted': results['y_pred'],
            'error': np.abs(results['y_test'].values - results['y_pred'])
        })
        
        bins = [0, 10, 50, 100, 500, float('inf')]
        labels = ['0-10', '11-50', '51-100', '101-500', '500+']
        error_df['count_bin'] = pd.cut(error_df['actual'], bins=bins, labels=labels, include_lowest=True)
        bin_errors = error_df.groupby('count_bin')['error'].mean()
        
        bars = ax6.bar(range(len(bin_errors)), bin_errors.values, color='orange', alpha=0.7)
        ax6.set_xticks(range(len(bin_errors)))
        ax6.set_xticklabels(bin_errors.index, rotation=45)
        ax6.set_ylabel('Mean Absolute Error')
        ax6.set_title('MAE by Count Range')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Melbourne Pedestrian Count Prediction Model - Summary Report', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(f'{self.plot_dir}/11_model_summary_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Model summary report saved!")

    def create_prediction_visualization(self, latitude, longitude, timestamp_str, prediction):
        """Create visualization for individual predictions"""
        if not self.save_plots:
            return
            
        print("Creating prediction visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Parse timestamp for context
        timestamp = pd.to_datetime(timestamp_str)
        
        # 1. Location on map context (if we have historical data for nearby locations)
        ax1 = axes[0, 0]
        ax1.scatter(longitude, latitude, color='red', s=100, marker='*', 
                    label=f'Prediction Location\nLat: {latitude:.4f}\nLng: {longitude:.4f}')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Prediction Location')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time context visualization
        ax2 = axes[0, 1]
        hour_labels = [f'{h:02d}:00' for h in range(0, 24, 3)]
        hour_values = [h for h in range(0, 24, 3)]
        ax2.bar(hour_values, [1]*len(hour_values), alpha=0.3, color='lightblue')
        ax2.bar([timestamp.hour], [1], color='red', alpha=0.8)
        ax2.set_xticks(hour_values)
        ax2.set_xticklabels(hour_labels, rotation=45)
        ax2.set_ylabel('Selected Time')
        ax2.set_title(f'Time Context: {timestamp.strftime("%Y-%m-%d %H:%M")}')
        ax2.set_ylim(0, 1.2)
        
        # 3. Day of week context
        ax3 = axes[1, 0]
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_colors = ['lightblue' if i != timestamp.dayofweek else 'red' for i in range(7)]
        ax3.bar(range(7), [1]*7, color=dow_colors, alpha=0.8)
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(dow_names)
        ax3.set_ylabel('Selected Day')
        ax3.set_title(f'Day of Week: {dow_names[timestamp.dayofweek]}')
        ax3.set_ylim(0, 1.2)
        
        # 4. Prediction result
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.7, f'PREDICTED COUNT', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.text(0.5, 0.5, f'{prediction:,}', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=36, fontweight='bold', color='red')
        ax4.text(0.5, 0.3, 'pedestrians', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Add border around prediction
        rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='black', 
                            linewidth=2, transform=ax4.transAxes)
        ax4.add_patch(rect)
        
        plt.tight_layout()
        
        # Save with timestamp in filename
        timestamp_str_clean = timestamp.strftime("%Y%m%d_%H%M")
        plt.savefig(f'{self.plot_dir}/prediction_{timestamp_str_clean}_{latitude}_{longitude}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction visualization saved!")

    def generate_all_plots_summary(self):
        """Generate a summary of all created plots"""
        if not self.save_plots:
            return
            
        plot_files = [f for f in os.listdir(self.plot_dir) if f.endswith('.png')]
        plot_files.sort()
        
        print(f"\n{'='*60}")
        print("VISUALIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total plots created: {len(plot_files)}")
        print(f"Plots saved in directory: {self.plot_dir}")
        print("\nGenerated plots:")
        
        plot_descriptions = {
            '01_data_distribution.png': 'Data distribution and basic statistics',
            '02_temporal_patterns.png': 'Temporal patterns (hourly, daily, monthly)',
            '03_spatial_analysis.png': 'Spatial analysis and location patterns',
            '04_correlation_matrix.png': 'Feature correlation matrix',
            '05_feature_analysis.png': 'Advanced feature analysis',
            '06_model_performance.png': 'Model performance evaluation',
            '07_feature_importance.png': 'Feature importance ranking',
            '08_error_analysis.png': 'Detailed error analysis',
            '09_optimization_history.png': 'Hyperparameter optimization history',
            '10_param_importance.png': 'Hyperparameter importance',
            '11_model_summary_report.png': 'Comprehensive model summary report'
        }
        
        for i, plot_file in enumerate(plot_files, 1):
            description = plot_descriptions.get(plot_file, 'Individual prediction visualization')
            print(f"{i:2d}. {plot_file:<35} - {description}")
        
        print(f"{'='*60}\n")
        
    def get_safe_gpu_params(self, base_params):
            """Get GPU parameters with safe fallback"""
            if self.use_gpu and self.gpu_available:
                try:
                    gpu_params = base_params.copy()
                    gpu_params.update({
                        'device_type': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0,
                        'force_col_wise': True,
                        'max_bin': 255,  # Reduce memory usage
                        'gpu_use_dp': False,  # Use single precision
                    })
                    return gpu_params, "GPU"
                except Exception as e:
                    print(f"⚠️  GPU setup failed: {e}")
                    self.use_gpu = False
                    
            # CPU fallback
            cpu_params = base_params.copy()
            cpu_params.update({
                'device_type': 'cpu',
                'force_col_wise': True,
                'num_threads': -1,  # Use all available CPU cores
            })
            return cpu_params, "CPU"

    def create_feature_analysis_plots(self, df):
        """Create feature analysis plots after feature engineering"""
        print("Creating feature analysis plots...")
        
        # Feature correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/04_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cyclical features visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Hour cyclical features
        axes[0,0].scatter(df['hour_sin'], df['hour_cos'], c=df['hour'], cmap='hsv', alpha=0.6)
        axes[0,0].set_title('Hour Cyclical Features')
        axes[0,0].set_xlabel('Hour Sin')
        axes[0,0].set_ylabel('Hour Cos')
        
        # Day of week cyclical features
        axes[0,1].scatter(df['dow_sin'], df['dow_cos'], c=df['dow'], cmap='Set1', alpha=0.6)
        axes[0,1].set_title('Day of Week Cyclical Features')
        axes[0,1].set_xlabel('DOW Sin')
        axes[0,1].set_ylabel('DOW Cos')
        
        # Distance from CBD distribution
        axes[1,0].hist(df['distance_from_cbd'], bins=30, alpha=0.7, color='lightblue')
        axes[1,0].set_title('Distance from CBD Distribution')
        axes[1,0].set_xlabel('Distance from CBD')
        axes[1,0].set_ylabel('Frequency')
        
        # Location clusters
        if hasattr(df, 'location_cluster'):
            cluster_counts = df['location_cluster'].value_counts().sort_index()
            axes[1,1].bar(cluster_counts.index, cluster_counts.values, color='lightgreen')
            axes[1,1].set_title('Location Clusters Distribution')
            axes[1,1].set_xlabel('Cluster')
            axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/05_feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature analysis plots saved!")
    
    def create_training_plots(self, results):
        """Create training and evaluation plots"""
        print("Creating training evaluation plots...")
        
        y_test = results['y_test']
        y_pred = results['y_pred']
        feature_imp = results['feature_importance']
        
        # 1. Model performance plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted scatter plot
        max_val = max(y_test.max(), y_pred.max())
        axes[0,0].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[0,0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        axes[0,0].set_xlabel('Actual Count')
        axes[0,0].set_ylabel('Predicted Count')
        axes[0,0].set_title(f'Actual vs Predicted (R² = {results["r2"]:.3f})')
        axes[0,0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Count')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residual Plot')
        axes[0,1].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[1,0].hist(residuals, bins=50, alpha=0.7, color='lightcoral')
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Residuals')
        axes[1,0].axvline(residuals.mean(), color='red', linestyle='--', 
                         label=f'Mean: {residuals.mean():.2f}')
        axes[1,0].legend()
        
        # Q-Q plot for residuals normality
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot of Residuals')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/06_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance plot
        plt.figure(figsize=(12, 8))
        top_features = feature_imp.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/07_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error analysis by different segments
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Create a temporary dataframe for error analysis
        error_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'error': np.abs(y_test.values - y_pred),
            'relative_error': np.abs(y_test.values - y_pred) / (y_test.values + 1)
        })
        
        # Error by count bins
        bins = [0, 10, 50, 100, 500, float('inf')]
        labels = ['0-10', '11-50', '51-100', '101-500', '500+']
        error_df['count_bin'] = pd.cut(error_df['actual'], bins=bins, labels=labels, include_lowest=True)
        bin_errors = error_df.groupby('count_bin')['error'].mean()
        
        axes[0,0].bar(range(len(bin_errors)), bin_errors.values, color='lightblue')
        axes[0,0].set_xticks(range(len(bin_errors)))
        axes[0,0].set_xticklabels(bin_errors.index)
        axes[0,0].set_title('Mean Absolute Error by Count Range')
        axes[0,0].set_xlabel('Actual Count Range')
        axes[0,0].set_ylabel('Mean Absolute Error')
        
        # Relative error distribution
        axes[0,1].hist(error_df['relative_error'], bins=50, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('Distribution of Relative Errors')
        axes[0,1].set_xlabel('Relative Error')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(error_df['relative_error'].median(), color='red', 
                         linestyle='--', label=f'Median: {error_df["relative_error"].median():.3f}')
        axes[0,1].legend()
        
        # Error vs actual count
        axes[1,0].scatter(error_df['actual'], error_df['error'], alpha=0.5, s=20)
        axes[1,0].set_xlabel('Actual Count')
        axes[1,0].set_ylabel('Absolute Error')
        axes[1,0].set_title('Error vs Actual Count')
        axes[1,0].grid(True, alpha=0.3)
        
        # Cumulative error plot
        sorted_errors = np.sort(error_df['error'])
        cumulative_errors = np.cumsum(sorted_errors) / np.sum(sorted_errors)
        axes[1,1].plot(range(len(sorted_errors)), cumulative_errors)
        axes[1,1].set_xlabel('Samples (sorted by error)')
        axes[1,1].set_ylabel('Cumulative Error Proportion')
        axes[1,1].set_title('Cumulative Error Distribution')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/08_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Training evaluation plots saved!")
    
    def create_hyperparameter_plots(self, study):
        """Create hyperparameter optimization plots"""
        print("Creating hyperparameter optimization plots...")
        
        # Optimization history
        plt.figure(figsize=(12, 6))
        trials_df = study.trials_dataframe()
        plt.plot(trials_df['number'], trials_df['value'])
        plt.xlabel('Trial Number')
        plt.ylabel('RMSE')
        plt.title('Hyperparameter Optimization History')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/09_optimization_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter importance (if available)
        try:
            param_importance = optuna.importance.get_param_importances(study)
            if param_importance:
                plt.figure(figsize=(10, 6))
                params = list(param_importance.keys())
                importances = list(param_importance.values())
                plt.barh(params, importances)
                plt.xlabel('Importance')
                plt.title('Hyperparameter Importance')
                plt.tight_layout()
                plt.savefig(f'{self.plot_dir}/10_param_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        except:
            print("Parameter importance plot not available")
        
        print("Hyperparameter optimization plots saved!")

    def create_advanced_features(self, df):
        """Create advanced features for better prediction"""
        df = df.copy()
        
        # Melbourne CBD center coordinates
        cbd_lat, cbd_lng = -37.8136, 144.9631
        
        # Distance from CBD
        df['distance_from_cbd'] = np.sqrt(
            (df['latitude'] - cbd_lat)**2 + (df['longitude'] - cbd_lng)**2
        )
        
        # Create location clusters
        if self.location_clusters is None:
            coords = df[['latitude', 'longitude']].drop_duplicates()
            self.location_clusters = KMeans(n_clusters=20, random_state=42)
            self.location_clusters.fit(coords)
        
        df['location_cluster'] = self.location_clusters.predict(df[['latitude', 'longitude']])
        
        # Time features (more comprehensive)
        df['month'] = df['timestamp_local'].dt.month
        df['year'] = df['timestamp_local'].dt.year
        df['day'] = df['timestamp_local'].dt.day
        df['quarter'] = df['timestamp_local'].dt.quarter
        df['week_of_year'] = df['timestamp_local'].dt.isocalendar().week
        
        # Advanced cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Rush hour indicators
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 14)).astype(int)
        
        # Business hours
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                  (df['dow'] < 5)).astype(int)
        
        # Season (Australian seasons)
        def get_season(month):
            if month in [12, 1, 2]: return 0  # Summer
            elif month in [3, 4, 5]: return 1  # Autumn
            elif month in [6, 7, 8]: return 2  # Winter
            else: return 3  # Spring
        
        df['season'] = df['month'].apply(get_season)
        
        # Location-based statistics (historical averages)
        if self.location_stats is None:
            self.location_stats = df.groupby(['location_id', 'hour', 'dow'])['count'].agg([
                'mean', 'std', 'median'
            ]).reset_index()
            self.location_stats.columns = ['location_id', 'hour', 'dow', 
                                         'location_hour_dow_mean', 'location_hour_dow_std', 
                                         'location_hour_dow_median']
        
        df = df.merge(self.location_stats, on=['location_id', 'hour', 'dow'], how='left')
        df[['location_hour_dow_mean', 'location_hour_dow_std', 'location_hour_dow_median']] = \
            df[['location_hour_dow_mean', 'location_hour_dow_std', 'location_hour_dow_median']].fillna(0)
        
        # Directional features
        df['total_direction'] = df['dir1'] + df['dir2']
        df['direction_ratio'] = np.where(df['dir2'] != 0, df['dir1'] / df['dir2'], 0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        self.feature_cols = [
            'latitude', 'longitude', 'hour', 'dow', 'is_weekend',
            'month', 'year', 'day', 'quarter', 'week_of_year',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
            'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'is_morning_rush', 'is_evening_rush', 'is_lunch_time',
            'is_business_hours', 'season', 'location_cluster',
            'distance_from_cbd', 'dir1', 'dir2', 'total_direction',
            'direction_ratio', 'location_hour_dow_mean',
            'location_hour_dow_std', 'location_hour_dow_median'
        ]
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_cols].copy()
        y = df['count'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Log transform the target to handle skewed distribution
        y_log = np.log1p(y)  # log(1+x) to handle zeros
        
        return X, y, y_log
    
    def objective(self, trial, X, y_log):
        """Objective function for hyperparameter optimization"""
        # Set parameters with safe GPU handling
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'verbose': -1,
            'random_state': 42,
            'max_bin': 255,  # Reduce memory usage
        }
        
        # Get safe GPU/CPU parameters
        params, device_type = self.get_safe_gpu_params(params)
        
        # Cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            
            # Try with smaller dataset for memory efficiency
            try:
                model = lgb.train(
                    params, train_data, 
                    valid_sets=[val_data],
                    num_boost_round=500,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            except Exception as e:
                print(f"⚠️  Training failed with {device_type}, trying CPU fallback: {e}")
                # Force CPU fallback
                params_cpu = {k: v for k, v in params.items() if 'gpu' not in k}
                params_cpu['device_type'] = 'cpu'
                params_cpu['num_threads'] = -1
                
                model = lgb.train(
                    params_cpu, train_data, 
                    valid_sets=[val_data],
                    num_boost_round=500,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            
            y_pred_log = model.predict(X_val_scaled, num_iteration=model.best_iteration)
            y_pred = np.expm1(y_pred_log)  # Transform back from log
            y_val_orig = np.expm1(y_val)
            
            rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)
    
    def train_model(self, df_t, test_size=0.2, n_trials=50, random_state=42):
        """Train the optimized LightGBM model"""
        # Load and preprocess data
        df = self.load_and_preprocess_data(df_t)
        
        df = self.create_advanced_features(df)
        
        if self.save_plots:
            self.create_data_exploration_plots(df)
            
        X, y, y_log = self.prepare_features(df)
               # Create feature analysis plots
               
        if self.save_plots:
            self.create_feature_analysis_plots(df)

        print(f"Training with {len(self.feature_cols)} features")
        
        # Hyperparameter optimization
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X, y_log), 
                      n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        print(f"Best RMSE: {study.best_value:.2f}")
              # Create hyperparameter optimization plots
        if self.save_plots:
            self.create_hyperparameter_plots(study)
        # Split data
        X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(
            X, y, y_log, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Final model training with best parameters
        final_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': random_state,
            'max_bin': 255,
            'force_col_wise': True,
            **best_params
        }
        
        if self.use_gpu:
            try:
                final_params['device_type'] = 'gpu'
                print("Using GPU acceleration")
            except:
                final_params['device_type'] = 'cpu'

                # Get safe GPU/CPU parameters
        final_params, device_type = self.get_safe_gpu_params(final_params)
        print(f"Training final model on {device_type}...")
        
        train_data = lgb.Dataset(X_train_scaled, label=y_log_train)
        valid_data = lgb.Dataset(X_test_scaled, label=y_log_test, reference=train_data)
        
        
        try:
            self.model = lgb.train(
                final_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=100)
                ]
            )
        except Exception as e:
            print(f"⚠️  Final training failed with {device_type}, using CPU: {e}")
            # Emergency CPU fallback
            cpu_params = {k: v for k, v in final_params.items() if 'gpu' not in k}
            cpu_params.update({
                'device_type': 'cpu',
                'num_threads': -1,
                'force_col_wise': True
            })
            
            self.model = lgb.train(
                cpu_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),
                    lgb.log_evaluation(period=100)
                ]
            )
        
        # Evaluate model (transform back from log space)
        y_pred_log = self.model.predict(X_test_scaled, num_iteration=self.model.best_iteration)
        y_pred = np.expm1(y_pred_log)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal Model Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"Mean actual count: {y_test.mean():.2f}")
        print(f"Mean predicted count: {y_pred.mean():.2f}")
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_imp.head(10))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': feature_imp,
            'best_params': best_params
        }
    
    def predict(self, latitude, longitude, timestamp_str):
        """Make prediction for given location and time"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        timestamp = pd.to_datetime(timestamp_str)
        
        # Create a temporary dataframe for feature engineering
        temp_df = pd.DataFrame({
            'location_id': [1],  # Dummy location_id
            'latitude': [latitude],
            'longitude': [longitude],
            'timestamp_local': [timestamp],
            'hour': [timestamp.hour],
            'dow': [timestamp.dayofweek],
            'is_weekend': [1 if timestamp.dayofweek >= 5 else 0],
            'dir1': [0],
            'dir2': [0],
            'count': [0]  # Dummy count for processing
        })
        
        # Apply feature engineering
        temp_df = self.create_advanced_features(temp_df)
        
        # Extract features
        features = temp_df[self.feature_cols].iloc[0:1]
        features = features.fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction (in log space)
        prediction_log = self.model.predict(features_scaled, 
                                          num_iteration=self.model.best_iteration)[0]
        
        # Transform back from log space
        prediction = np.expm1(prediction_log)
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        return round(prediction)
    
    def save_model(self, model_path='advanced_pedestrian_model.pkl'):
        """Save the trained model and all components"""
        if self.model is None or self.scaler is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'location_clusters': self.location_clusters,
            'location_stats': self.location_stats,
            'label_encoders': self.label_encoders,
            'use_gpu': self.use_gpu,
            'save_plots': self.save_plots,
            'plot_dir': self.plot_dir
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Advanced model saved to {model_path}")
    
    def load_model(self, model_path='advanced_pedestrian_model.pkl'):
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_cols = model_data['feature_cols']
        self.location_clusters = model_data['location_clusters']
        self.location_stats = model_data['location_stats']
        self.label_encoders = model_data['label_encoders']
        self.use_gpu = model_data['use_gpu']
        self.save_plots = model_data.get('save_plots', True)
        self.plot_dir = model_data.get('plot_dir', 'model_plots')
        
        print(f"Advanced model loaded from {model_path}")



    

predictor2 = AdvancedPedestrianPredictor(use_gpu=False, save_plots=True, plot_dir="pedestrian_model_plots_new")

# Read data efficiently
df = pd.read_csv(r"data\processed\melbourne_pedestrian_hourly_joined.csv", 
            parse_dates=['timestamp_local', 'timestamp_utc', 'date'],
            low_memory=False)
import copy as cp
#copy_hist = cp.deepcopy(hist)
df['timestamp_local'] = (
    pd.to_datetime(df['timestamp_local'], utc=True)
      .dt.tz_convert('Australia/Melbourne')
      .dt.tz_localize(None)
)

results = predictor2.train_model(df, n_trials=50)

    
    # Load model for predictions
#predictor2.load_model('advanced_pedestrian_model.pkl')
predictor2.save_model('advanced_pedestrian_model.pkl')

predictor2.generate_all_plots_summary()

predictor2.create_model_summary_report(results)
predictor2.create_training_plots(results)

lat, lng = -37.8199817, 144.96872865
timestamp = "2025-07-30 13:00:00"
predicted_count = predictor2.predict(lat, lng, timestamp)
print(f"Predicted pedestrian count: {predicted_count}")
predictor2.create_prediction_visualization(lat, lng, timestamp, predicted_count)
