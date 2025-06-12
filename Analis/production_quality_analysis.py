# -*- coding: utf-8 -*-
"""
Production-Quality Welding Analysis
Focus: Robustness, reliability, and production readiness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import xgboost as xgb
import joblib
import logging
import warnings
from datetime import datetime
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionQualityAnalyzer:
    def __init__(self, data_path="./Dataset/Dataset.csv", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.feature_cols = ['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']
        self.results = {}
        self.validation_results = {}
        
        # Create results directory
        self.results_dir = f"production_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Production Quality Analyzer initialized")
        logger.info(f"Results directory: {self.results_dir}")
    
    def validate_data(self):
        """Comprehensive data validation"""
        logger.info("Starting data validation...")
        
        try:
            # Load data
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.df.shape}")
            
            # Check for required columns
            required_cols = self.feature_cols + ['FIN_JGMT']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for missing values
            missing_values = self.df[required_cols].isnull().sum()
            if missing_values.sum() > 0:
                logger.warning(f"Missing values detected: {missing_values.to_dict()}")
                # Handle missing values
                self.df = self.df.dropna(subset=required_cols)
                logger.info(f"Rows with missing values removed. New shape: {self.df.shape}")
            
            # Check data types
            for col in self.feature_cols:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    logger.warning(f"Non-numeric data in {col}, attempting conversion")
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Check target variable
            unique_targets = self.df['FIN_JGMT'].unique()
            if not set(unique_targets).issubset({0, 1}):
                raise ValueError(f"Invalid target values: {unique_targets}")
            
            # Check for outliers
            self._detect_outliers()
            
            # Prepare features
            self.X = self.df[self.feature_cols]
            self.y = self.df['FIN_JGMT']
            
            logger.info("Data validation completed successfully")
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def _detect_outliers(self):
        """Detect and log outliers"""
        outlier_info = {}
        
        for col in self.feature_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                       (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_info[col] = outliers
        
        total_outliers = sum(outlier_info.values())
        logger.info(f"Outliers detected: {outlier_info} (Total: {total_outliers})")
    
    def robust_model_evaluation(self):
        """Comprehensive model evaluation with cross-validation"""
        logger.info("Starting robust model evaluation...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, 
            stratify=self.y
        )
        
        # Models with production-ready configurations
        models = {
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state, max_iter=1000,
                class_weight='balanced', solver='liblinear'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state, eval_metric='logloss',
                scale_pos_weight=10, n_estimators=100
            )
        }
        
        # Scalers for robustness
        scalers = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler()
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        best_score = 0
        best_config = None
        
        for scaler_name, scaler in scalers.items():
            for model_name, model in models.items():
                try:
                    # Create pipeline
                    pipeline = Pipeline([
                        ('scaler', scaler),
                        ('model', model)
                    ])
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        pipeline, X_train, y_train, cv=cv, 
                        scoring='f1', n_jobs=-1
                    )
                    
                    # Train on full training set
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
                    
                    # Calculate metrics
                    test_f1 = f1_score(y_test, y_pred, pos_label=0)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    test_roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                    
                    config_key = f"{model_name}_{scaler_name}"
                    self.results[config_key] = {
                        'model_name': model_name,
                        'scaler_name': scaler_name,
                        'cv_f1_mean': cv_scores.mean(),
                        'cv_f1_std': cv_scores.std(),
                        'test_f1': test_f1,
                        'test_accuracy': test_accuracy,
                        'test_roc_auc': test_roc_auc,
                        'pipeline': pipeline,
                        'cv_scores': cv_scores
                    }
                    
                    logger.info(f"{config_key}: CV F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test F1={test_f1:.4f}")
                    
                    # Track best model
                    if cv_scores.mean() > best_score:
                        best_score = cv_scores.mean()
                        best_config = config_key
                        
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} with {scaler_name}: {str(e)}")
        
        logger.info(f"Best configuration: {best_config} (CV F1: {best_score:.4f})")
        return best_config, X_test, y_test
    
    def create_ensemble_model(self, top_n=3):
        """Create ensemble from top performing models"""
        logger.info("Creating ensemble model...")
        
        # Sort models by CV performance
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['cv_f1_mean'], 
            reverse=True
        )
        
        # Select top models
        top_models = sorted_results[:top_n]
        
        # Create ensemble
        estimators = []
        for config_name, result in top_models:
            model_name = result['model_name']
            scaler_name = result['scaler_name']
            pipeline = result['pipeline']
            estimators.append((f"{model_name}_{scaler_name}", pipeline))
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        logger.info(f"Ensemble created with {len(estimators)} models")
        return ensemble
    
    def comprehensive_validation(self, model, X_test, y_test):
        """Comprehensive model validation"""
        logger.info("Starting comprehensive validation...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, pos_label=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
        }
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log results
        logger.info(f"Validation metrics: {metrics}")
        
        # Save detailed results
        self._save_validation_results(metrics, class_report, conf_matrix, y_test, y_pred, y_pred_proba)
        
        return metrics
    
    def _save_validation_results(self, metrics, class_report, conf_matrix, y_test, y_pred, y_pred_proba):
        """Save comprehensive validation results"""
        # Create validation plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confusion Matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Defect', 'Normal'], yticklabels=['Defect', 'Normal'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Metrics bar plot
        ax2.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen', 'salmon'])
        ax2.set_title('Model Performance Metrics')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        for i, (k, v) in enumerate(metrics.items()):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ax3.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
            ax3.plot([0, 1], [0, 1], 'k--', label='Random')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve')
            ax3.legend()
            ax3.grid(True)
        
        # Precision-Recall Curve
        if y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba, pos_label=0)
            ax4.plot(recall, precision)
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save metrics to file
        with open(f'{self.results_dir}/metrics.txt', 'w') as f:
            f.write("PRODUCTION MODEL VALIDATION RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            
            f.write("METRICS:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nCLASSIFICATION REPORT:\n")
            f.write(classification_report(y_test, y_pred))
    
    def create_production_pipeline(self, best_config):
        """Create final production pipeline"""
        logger.info("Creating production pipeline...")
        
        best_pipeline = self.results[best_config]['pipeline']
        
        # Retrain on full dataset for production
        production_pipeline = clone(best_pipeline)
        production_pipeline.fit(self.X, self.y)
        
        # Save pipeline with metadata
        pipeline_data = {
            'pipeline': production_pipeline,
            'feature_columns': self.feature_cols,
            'model_config': best_config,
            'training_timestamp': datetime.now(),
            'validation_metrics': self.results[best_config]
        }
        
        pipeline_filename = f'{self.results_dir}/production_pipeline.joblib'
        joblib.dump(pipeline_data, pipeline_filename)
        
        logger.info(f"Production pipeline saved: {pipeline_filename}")
        return production_pipeline
    
    def run_production_analysis(self):
        """Run complete production-quality analysis"""
        try:
            logger.info("Starting production quality analysis...")
            
            # Step 1: Data validation
            self.validate_data()
            
            # Step 2: Model evaluation
            best_config, X_test, y_test = self.robust_model_evaluation()
            
            # Step 3: Create ensemble (optional)
            ensemble = self.create_ensemble_model()
            ensemble.fit(self.X, self.y)  # Train ensemble on full data
            
            # Step 4: Validate best single model
            best_model = self.results[best_config]['pipeline']
            single_metrics = self.comprehensive_validation(best_model, X_test, y_test)
            
            # Step 5: Validate ensemble
            ensemble_metrics = self.comprehensive_validation(ensemble, X_test, y_test)
            
            # Step 6: Choose final model
            if ensemble_metrics['f1_score'] > single_metrics['f1_score']:
                final_model = ensemble
                final_metrics = ensemble_metrics
                model_type = "Ensemble"
            else:
                final_model = best_model
                final_metrics = single_metrics
                model_type = best_config
            
            # Step 7: Create production pipeline
            production_pipeline = self.create_production_pipeline(best_config)
            
            logger.info(f"Production analysis completed successfully")
            logger.info(f"Final model type: {model_type}")
            logger.info(f"Final F1-Score: {final_metrics['f1_score']:.4f}")
            
            return production_pipeline, final_metrics
            
        except Exception as e:
            logger.error(f"Production analysis failed: {str(e)}")
            raise

if __name__ == "__main__":
    analyzer = ProductionQualityAnalyzer()
    pipeline, metrics = analyzer.run_production_analysis()
    
    print("\n=== PRODUCTION ANALYSIS COMPLETED ===")
    print(f"Results saved in: {analyzer.results_dir}")
    print(f"Final F1-Score: {metrics['f1_score']:.4f}")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")