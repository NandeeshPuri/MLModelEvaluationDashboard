"""
Functions for loading and processing model data.
"""
import os
import pandas as pd
import numpy as np
import pickle
from config import MODEL_DIR

def check_model_directory():
    """Ensure the model directory exists."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Directory '{MODEL_DIR}' not found. Please make sure you've extracted model_outputs.zip here.")

def load_model_results():
    """Load the model results from CSV files."""
    check_model_directory()
    
    try:
        initial_results = pd.read_csv(os.path.join(MODEL_DIR, 'initial_model_results.csv'), index_col=0)
        tuned_results = pd.read_csv(os.path.join(MODEL_DIR, 'tuned_model_results.csv'), index_col=0)
        comparison_df = pd.read_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'), index_col=0)
        
        # Get model names
        model_names = comparison_df.index.tolist()
        
        # Define models with feature importance
        feature_importance_models = [
            name for name in model_names 
            if name in ['Random Forest', 'Decision Tree', 'Gradient Boosting'] and
            os.path.exists(os.path.join(MODEL_DIR, f'feature_importance_{name.replace(" ", "_").lower()}.csv'))
        ]
        
        return {
            'initial_results': initial_results,
            'tuned_results': tuned_results,
            'comparison_df': comparison_df,
            'model_names': model_names,
            'feature_importance_models': feature_importance_models
        }
        
    except Exception as e:
        raise Exception(f"Error loading model results: {str(e)}")

def load_test_data():
    """Load the test data if available."""
    check_model_directory()
    
    test_data = None
    test_data_path = os.path.join(MODEL_DIR, 'test_data.pkl')
    if os.path.exists(test_data_path):
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
    
    return test_data

def load_confusion_matrix(model_name):
    """Load the confusion matrix for a specific model."""
    cm_path = os.path.join(MODEL_DIR, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.csv')
    
    if os.path.exists(cm_path):
        return np.loadtxt(cm_path, delimiter=',')
    return None

def load_roc_data():
    """Load ROC curve data for all models."""
    check_model_directory()
    
    roc_data = {}
    model_results = load_model_results()
    model_names = model_results['model_names']
    
    for name in model_names:
        # Load ROC data if available
        roc_data_path = os.path.join(MODEL_DIR, f'roc_data_{name.replace(" ", "_").lower()}.csv')
        auc_path = os.path.join(MODEL_DIR, f'auc_{name.replace(" ", "_").lower()}.txt')
        
        if os.path.exists(roc_data_path) and os.path.exists(auc_path):
            curve_data = pd.read_csv(roc_data_path)
            
            with open(auc_path, 'r') as f:
                roc_auc = float(f.read().strip())
            
            roc_data[name] = {
                'data': curve_data,
                'auc': roc_auc
            }
    
    return roc_data

def load_feature_importance(model_name):
    """Load feature importance data for a specific model."""
    fi_path = os.path.join(MODEL_DIR, f'feature_importance_{model_name.replace(" ", "_").lower()}.csv')
    
    if os.path.exists(fi_path):
        return pd.read_csv(fi_path, index_col=0)
    return None

def load_model_details():
    """Load the best parameters for each model."""
    params_path = os.path.join(MODEL_DIR, 'best_parameters.csv')
    
    if os.path.exists(params_path):
        return pd.read_csv(params_path, index_col=0)
    return None