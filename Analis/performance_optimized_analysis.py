# -*- coding: utf-8 -*-
"""
Performance-Optimized Welding Quality Analysis
Focus: Speed and computational efficiency with extended models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Performance settings
plt.ioff()  # Turn off interactive plotting for speed

class PerformanceAnalyzer:
    def __init__(self, data_path="./Dataset/Dataset.csv"):
        self.data_path = data_path
        self.feature_cols = ['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def load_and_prepare_data(self):
        """Fast data loading and preparation"""
        print("Loading data...")
        start_time = time.time()
        
        # Load only necessary columns
        self.df = pd.read_csv(self.data_path, usecols=self.feature_cols + ['FIN_JGMT'])
        
        # Quick data check
        print(f"Dataset size: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Class distribution
        class_counts = self.df['FIN_JGMT'].value_counts()
        print(f"Class distribution - Normal: {class_counts[1]}, Defects: {class_counts[0]}")
        print(f"Imbalance ratio: {class_counts[1]/class_counts[0]:.2f}:1")
        
        # Prepare features
        self.X = self.df[self.feature_cols]
        self.y = self.df['FIN_JGMT']
        
        load_time = time.time() - start_time
        print(f"Data loading completed in {load_time:.2f} seconds")
        
    def fast_model_comparison(self):
        """Quick model comparison with balancing methods"""
        print("\n=== FAST MODEL COMPARISON WITH BALANCING ===")
        start_time = time.time()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Fast models with optimized parameters
        models = {
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=500, solver='liblinear'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=50, max_depth=6, random_state=42, 
                eval_metric='logloss', verbosity=0
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=50, max_depth=6, random_state=42, 
                verbosity=-1, force_col_wise=True
            ),
            'CatBoost': CatBoostClassifier(
                iterations=50, depth=6, random_state=42, 
                verbose=False, allow_writing_files=False
            ),
            'SVM': SVC(
                kernel='rbf', random_state=42, probability=True
            )
        }
        
        # Balancing methods
        balancing_methods = {
            'Original': None,
            'SMOTE': SMOTE(random_state=42),
            'UnderSample': RandomUnderSampler(random_state=42)
        }
        
        results_data = []
        
        for balance_name, balancer in balancing_methods.items():
            print(f"\nTesting with {balance_name} balancing:")
            
            # Apply balancing
            if balancer is None:
                X_train_balanced, y_train_balanced = X_train, y_train
            else:
                balance_start = time.time()
                X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)
                balance_time = time.time() - balance_start
                print(f"  Balancing time: {balance_time:.2f}s")
                print(f"  New class distribution: {np.bincount(y_train_balanced)}")
            
            for model_name, model in models.items():
                try:
                    model_start = time.time()
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    
                    # Train and predict
                    pipeline.fit(X_train_balanced, y_train_balanced)
                    y_pred = pipeline.predict(X_test)
                    
                    # Evaluate
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, pos_label=0)
                    
                    model_time = time.time() - model_start
                    
                    config_key = f"{model_name}_{balance_name}"
                    self.results[config_key] = {
                        'model_name': model_name,
                        'balance_name': balance_name,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'training_time': model_time,
                        'pipeline': pipeline
                    }
                    
                    results_data.append({
                        'Model': model_name,
                        'Balancing': balance_name,
                        'F1_Score': f1,
                        'Accuracy': accuracy,
                        'Time': model_time
                    })
                    
                    print(f"  {model_name}: F1={f1:.4f}, Acc={accuracy:.4f}, Time={model_time:.2f}s")
                    
                    # Track best model
                    if f1 > self.best_score:
                        self.best_score = f1
                        self.best_model = config_key
                        
                except Exception as e:
                    print(f"  {model_name}: Error - {str(e)}")
        
        total_time = time.time() - start_time
        print(f"\nModel comparison completed in {total_time:.2f} seconds")
        print(f"Best model: {self.best_model} (F1: {self.best_score:.4f})")
        
        return pd.DataFrame(results_data)
    
    def create_production_pipeline(self):
        """Create optimized pipeline for production"""
        print("\n=== CREATING PRODUCTION PIPELINE ===")
        
        best_pipeline = self.results[self.best_model]['pipeline']
        
        # Save pipeline with metadata
        pipeline_data = {
            'pipeline': best_pipeline,
            'model_config': self.best_model,
            'f1_score': self.best_score,
            'feature_columns': self.feature_cols
        }
        
        pipeline_filename = 'performance_optimized_pipeline.joblib'
        joblib.dump(pipeline_data, pipeline_filename)
        print(f"Pipeline saved: {pipeline_filename}")
        print(f"Best configuration: {self.best_model}")
        print(f"F1-Score: {self.best_score:.4f}")
        
        return best_pipeline
    
    def create_visualizations(self, results_df):
        """Create comprehensive visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Class distribution
        ax1 = plt.subplot(2, 4, 1)
        class_counts = self.df['FIN_JGMT'].value_counts()
        ax1.bar(['Defect (0)', 'Normal (1)'], [class_counts[0], class_counts[1]], 
                color=['red', 'green'], alpha=0.7)
        ax1.set_title('Class Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        for i, v in enumerate([class_counts[0], class_counts[1]]):
            ax1.text(i, v + 50, f'{v}\n({v/len(self.df)*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. F1-Score heatmap by model and balancing
        ax2 = plt.subplot(2, 4, 2)
        pivot_f1 = results_df.pivot(index='Model', columns='Balancing', values='F1_Score')
        sns.heatmap(pivot_f1, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax2)
        ax2.set_title('F1-Score Heatmap', fontweight='bold')
        
        # 3. Top 5 models by F1-Score
        ax3 = plt.subplot(2, 4, 3)
        top_5 = results_df.nlargest(5, 'F1_Score')
        model_labels = [f"{row['Model']}\n({row['Balancing']})" for _, row in top_5.iterrows()]
        bars = ax3.bar(range(len(top_5)), top_5['F1_Score'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(top_5))))
        ax3.set_title('Top 5 Models by F1-Score', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_xticks(range(len(top_5)))
        ax3.set_xticklabels(model_labels, rotation=45, ha='right')
        # Add values on bars
        for i, v in enumerate(top_5['F1_Score']):
            ax3.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance vs Speed scatter
        ax4 = plt.subplot(2, 4, 4)
        scatter = ax4.scatter(results_df['Time'], results_df['F1_Score'], 
                             c=results_df.index, cmap='tab10', s=100, alpha=0.7)
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('F1-Score')
        ax4.set_title('Performance vs Speed Trade-off', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model comparison by balancing method
        ax5 = plt.subplot(2, 4, 5)
        for balance_method in results_df['Balancing'].unique():
            subset = results_df[results_df['Balancing'] == balance_method]
            ax5.plot(subset['Model'], subset['F1_Score'], 
                    marker='o', label=balance_method, linewidth=2, markersize=8)
        ax5.set_title('F1-Score by Balancing Method', fontweight='bold')
        ax5.set_ylabel('F1-Score')
        ax5.legend()
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Accuracy vs F1-Score
        ax6 = plt.subplot(2, 4, 6)
        for balance_method in results_df['Balancing'].unique():
            subset = results_df[results_df['Balancing'] == balance_method]
            ax6.scatter(subset['Accuracy'], subset['F1_Score'], 
                       label=balance_method, s=100, alpha=0.7)
        ax6.set_xlabel('Accuracy')
        ax6.set_ylabel('F1-Score')
        ax6.set_title('Accuracy vs F1-Score', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Best model metrics
        ax7 = plt.subplot(2, 4, 7)
        best_result = self.results[self.best_model]
        metrics = ['F1-Score', 'Accuracy']
        values = [best_result['f1_score'], best_result['accuracy']]
        bars = ax7.bar(metrics, values, color=['lightgreen', 'skyblue'], alpha=0.8)
        ax7.set_title(f'Best Model Metrics\n{self.best_model}', fontweight='bold')
        ax7.set_ylabel('Score')
        ax7.set_ylim(0, 1)
        for i, v in enumerate(values):
            ax7.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Training time comparison
        ax8 = plt.subplot(2, 4, 8)
        avg_times = results_df.groupby('Model')['Time'].mean().sort_values()
        bars = ax8.barh(range(len(avg_times)), avg_times.values, 
                       color=plt.cm.plasma(np.linspace(0, 1, len(avg_times))))
        ax8.set_yticks(range(len(avg_times)))
        ax8.set_yticklabels(avg_times.index)
        ax8.set_xlabel('Average Training Time (seconds)')
        ax8.set_title('Training Time by Model', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total models tested: {len(results_df)}")
        print(f"Best F1-Score: {results_df['F1_Score'].max():.4f}")
        print(f"Average F1-Score: {results_df['F1_Score'].mean():.4f}")
        print(f"Fastest model: {results_df.loc[results_df['Time'].idxmin(), 'Model']} ({results_df['Time'].min():.2f}s)")
        print(f"Best balancing method: {results_df.loc[results_df['F1_Score'].idxmax(), 'Balancing']}")
        
    def run_analysis(self):
        """Run complete performance-optimized analysis"""
        total_start = time.time()
        
        self.load_and_prepare_data()
        results_df = self.fast_model_comparison()
        pipeline = self.create_production_pipeline()
        self.create_visualizations(results_df)
        
        total_time = time.time() - total_start
        
        print(f"\n=== PERFORMANCE ANALYSIS COMPLETED ===")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Best model: {self.best_model}")
        print(f"F1-Score: {self.best_score:.4f}")
        
        return pipeline, results_df

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    pipeline, results = analyzer.run_analysis()