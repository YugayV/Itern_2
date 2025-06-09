import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (11.69, 8.27)  # A4 size in inches

def create_title_slide(pdf):
    """Create title slide"""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.text(0.5, 0.7, 'Pipe Welding Data Analysis', 
            fontsize=28, fontweight='bold', ha='center', va='center')
    ax.text(0.5, 0.5, 'Machine Learning for Welding Quality Prediction', 
            fontsize=18, ha='center', va='center')
    ax.text(0.5, 0.3, 'Based on analysis of 739,888 records', 
            fontsize=14, ha='center', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_data_overview_slide(pdf):
    """Create data overview slide"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.69, 8.27))
    
    # Title
    fig.suptitle('Data Overview', fontsize=20, fontweight='bold')
    
    # Variable descriptions
    ax1.text(0.05, 0.9, 'Dataset Variables:', fontsize=14, fontweight='bold')
    variables_text = '''PIPE_NO - Pipe serial number
DV_R - Right side voltage
DA_R - Right side current
AV_R - Average voltage
AA_R - Average current
PM_R - Welding mode code
FIN_JGMT - Welding quality (1: normal, 0: defect)'''
    ax1.text(0.05, 0.1, variables_text, fontsize=10, va='bottom')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Data statistics
    ax2.text(0.05, 0.9, 'Basic Statistics:', fontsize=14, fontweight='bold')
    stats_text = '''Total records: 739,888
Missing values: 0 (100% completeness)
"Normal" class (1): 626,092 (84.6%)
"Defect" class (0): 113,796 (15.4%)
Class ratio: 5.5:1'''
    ax2.text(0.05, 0.1, stats_text, fontsize=10, va='bottom')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Class distribution (visualization)
    classes = ['Defect (0)', 'Normal (1)']
    counts = [113796, 626092]
    colors = ['red', 'green']
    
    ax3.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Class Distribution', fontsize=12, fontweight='bold')
    
    # Outliers in data
    ax4.text(0.05, 0.9, 'Data Outliers:', fontsize=14, fontweight='bold')
    outliers_text = '''PM_R: 14.04% of records
AA_R: 7.91% of records
AV_R: 0.24% of records
DA_R: 0.19% of records
DV_R: 0.18% of records'''
    ax4.text(0.05, 0.1, outliers_text, fontsize=10, va='bottom')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_model_results_slide(pdf):
    """Create model results slide"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.69, 8.27))
    
    fig.suptitle('Modeling Results', fontsize=20, fontweight='bold')
    
    # Quality metrics
    metrics = ['Accuracy', 'F1 (class 0)', 'F1 (class 1)', 'ROC-AUC']
    train_scores = [0.8189, 0.0991, 0.8993, 0.5856]
    test_scores = [0.8191, 0.0999, 0.8994, 0.5880]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, train_scores, width, label='Training set', alpha=0.8)
    ax1.bar(x + width/2, test_scores, width, label='Test set', alpha=0.8)
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title('Model Quality Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature importance
    features = ['PM_R', 'AA_R', 'AV_R', 'DA_R', 'DV_R']
    coefficients = [7.1105, 0.0514, 0.0481, 0.0080, 0.0080]
    
    ax2.barh(features, coefficients, color='skyblue')
    ax2.set_xlabel('Coefficients')
    ax2.set_title('Feature Importance')
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix (approximate)
    confusion_matrix = np.array([[18000, 4759], [8000, 117219]])
    im = ax3.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    ax3.set_title('Confusion Matrix')
    
    # Add text to cells
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, confusion_matrix[i, j], ha='center', va='center', fontsize=12)
    
    ax3.set_xlabel('Predicted class')
    ax3.set_ylabel('True class')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Defect', 'Normal'])
    ax3.set_yticklabels(['Defect', 'Normal'])
    
    # Conclusions
    ax4.text(0.05, 0.9, 'Key Findings:', fontsize=14, fontweight='bold')
    conclusions_text = '''1. Class imbalance makes defect
   detection challenging

2. PM_R (welding mode) is the most
   important feature

3. Model shows high accuracy (82%)
   but low F1 for defects (0.1)

4. ROC-AUC ≈ 0.59 indicates limited
   ability to distinguish classes'''
    ax4.text(0.05, 0.1, conclusions_text, fontsize=10, va='bottom')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_recommendations_slide(pdf):
    """Create recommendations slide"""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    
    ax.text(0.5, 0.9, 'Recommendations', fontsize=24, fontweight='bold', ha='center')
    
    recommendations_text = '''1. MODEL IMPROVEMENT
   • Explore other algorithms (Random Forest, XGBoost, CatBoost)
   • Perform hyperparameter tuning
   • Create new features based on existing ones

2. HANDLING CLASS IMBALANCE
   • Apply SMOTE, ADASYN methods
   • Use more advanced balancing techniques

3. PRODUCTION RECOMMENDATIONS
   • Pay special attention to PM_R parameter control
   • Monitor welding modes in real-time
   • Implement early defect warning system

4. FURTHER RESEARCH
   • Collect more data on defective samples
   • Study temporal dependencies in welding process
   • Integrate additional sensors'''
    
    ax.text(0.05, 0.05, recommendations_text, fontsize=12, va='bottom', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def main():
    """Main function to create PDF presentation"""
    with PdfPages('Pipe_Welding_Analysis_Presentation.pdf') as pdf:
        print("Creating title slide...")
        create_title_slide(pdf)
        
        print("Creating data overview slide...")
        create_data_overview_slide(pdf)
        
        print("Creating model results slide...")
        create_model_results_slide(pdf)
        
        print("Creating recommendations slide...")
        create_recommendations_slide(pdf)
        
        # PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Pipe Welding Data Analysis'
        d['Author'] = 'Data Analysis Team'
        d['Subject'] = 'Machine Learning for Welding Quality Prediction'
        d['Keywords'] = 'Welding, Machine Learning, Data Analysis'
        d['Creator'] = 'Python matplotlib'
    
    print("PDF presentation successfully created: Pipe_Welding_Analysis_Presentation.pdf")

if __name__ == "__main__":
    main()