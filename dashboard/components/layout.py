"""
Dashboard layout definition.
"""
from dash import html, dcc
from components.data_loader import load_model_results
from config import METRICS, STYLES

def create_header():
    """Create the dashboard header."""
    return html.Div([
        html.H1("Machine Learning Model Comparison Dashboard", 
                style=STYLES['header'])
    ])

def create_metrics_panel():
    """Create the panel for model performance metrics."""
    model_data = load_model_results()
    
    return html.Div([
        html.H2("Model Performance Metrics", 
                style=STYLES['section_header']),
        
        # Dropdown to select metric
        html.Label("Select Metric:", style=STYLES['dropdown_label']),
        dcc.Dropdown(
            id='metric-dropdown',
            options=METRICS,
            value='Accuracy',
            clearable=False,
            style=STYLES['dropdown']
        ),
        
        # Graph to show original vs tuned performance
        dcc.Graph(id='performance-graph', style={'marginBottom': '30px'})
    ])

def create_roc_panel():
    """Create the panel for ROC curves."""
    return html.Div([
        html.H2("ROC Curves", 
                style=STYLES['section_header']),
        dcc.Graph(id='roc-graph', style={'marginBottom': '30px'})
    ])

def create_confusion_panel():
    """Create the panel for confusion matrices."""
    model_data = load_model_results()
    
    return html.Div([
        html.H2("Confusion Matrices", 
                style=STYLES['section_header']),
        
        # Dropdown to select model for confusion matrix
        html.Label("Select Model:", style=STYLES['dropdown_label']),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': name, 'value': name} for name in model_data['model_names']],
            value=model_data['model_names'][0] if model_data['model_names'] else None,
            clearable=False,
            style=STYLES['dropdown']
        ),
        
        # Graph for confusion matrix
        dcc.Graph(id='confusion-matrix', style={'marginBottom': '30px'})
    ])

def create_feature_importance_panel():
    """Create the panel for feature importance."""
    model_data = load_model_results()
    
    return html.Div([
        html.H2("Feature Importance", 
                style=STYLES['section_header']),
        html.Label("Select Model:", style=STYLES['dropdown_label']),
        dcc.Dropdown(
            id='feature-model-dropdown',
            options=[{'label': name, 'value': name} for name in model_data['feature_importance_models']],
            value=model_data['feature_importance_models'][0] if model_data['feature_importance_models'] else None,
            clearable=False,
            style=STYLES['dropdown']
        ),
        dcc.Graph(id='feature-importance-graph')
    ], style={'display': 'block' if model_data['feature_importance_models'] else 'none'})

def create_model_details_panel():
    """Create the panel for model hyperparameters."""
    return html.Div([
        html.H2("Model Hyperparameters", 
                style=STYLES['section_header']),
        html.Div(id='model-details', style={'marginTop': '20px'})
    ], style={'clear': 'both', 'padding': '20px', **STYLES['panel']})

def create_layout():
    """Create the main dashboard layout."""
    return html.Div([
        create_header(),
        
        html.Div([
            # Left section - Model Performance Metrics and ROC Curves
            html.Div([
                create_metrics_panel(),
                create_roc_panel()
            ], style=STYLES['half_panel']),
            
            # Right section - Confusion Matrix and Feature Importance
            html.Div([
                create_confusion_panel(),
                create_feature_importance_panel()
            ], style=STYLES['right_panel'])
        ], style={'marginBottom': '50px'}),
        
        # Footer section with model details
        create_model_details_panel()
    ], style=STYLES['container'])