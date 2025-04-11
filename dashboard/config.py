"""
Configuration settings for the dashboard application.
"""
import os

# Model directory - this should point to where the model outputs are stored
MODEL_DIR = '../model_outputs'

# Dashboard settings
DASHBOARD_TITLE = "Machine Learning Model Comparison Dashboard"

# Available metrics for comparison
METRICS = [
    {'label': 'Accuracy', 'value': 'Accuracy'},
    {'label': 'Precision', 'value': 'Precision'},
    {'label': 'Recall', 'value': 'Recall'},
    {'label': 'F1 Score', 'value': 'F1 Score'}
]

# Colors for charts
CHART_COLORS = {
    'bar_original': '#3498db',
    'bar_tuned': '#2ecc71',
    'line_random': '#95a5a6',
    'model_colors': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
}

# Style settings
STYLES = {
    'container': {
        'maxWidth': '1400px', 
        'margin': 'auto', 
        'padding': '20px', 
        'fontFamily': 'Arial, sans-serif'
    },
    'panel': {
        'boxShadow': '0px 0px 10px #ccc',
        'padding': '20px',
        'borderRadius': '5px',
        'marginBottom': '30px'
    },
    'half_panel': {
        'width': '48%',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'boxShadow': '0px 0px 10px #ccc',
        'padding': '20px',
        'borderRadius': '5px'
    },
    'right_panel': {
        'width': '48%',
        'display': 'inline-block',
        'verticalAlign': 'top',
        'float': 'right',
        'boxShadow': '0px 0px 10px #ccc',
        'padding': '20px',
        'borderRadius': '5px'
    },
    'header': {
        'textAlign': 'center',
        'marginBottom': '30px',
        'marginTop': '20px',
        'fontFamily': 'Arial, sans-serif',
        'color': '#2c3e50'
    },
    'section_header': {
        'textAlign': 'center',
        'marginBottom': '20px',
        'color': '#2c3e50'
    },
    'dropdown_label': {
        'fontWeight': 'bold',
        'marginBottom': '10px'
    },
    'dropdown': {
        'marginBottom': '20px'
    }
}