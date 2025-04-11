"""
All dashboard callbacks in one file.
"""
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from dash import html
import numpy as np
import pandas as pd
import json
import os

from components.data_loader import (
    load_model_results, 
    load_confusion_matrix, 
    load_roc_data,
    load_feature_importance,
    load_model_details
)
from config import CHART_COLORS

def register_callbacks(app):
    """Register all callbacks with the Dash app."""
    
    # Performance graph callback
    @app.callback(
        Output('performance-graph', 'figure'),
        [Input('metric-dropdown', 'value')]
    )
    def update_performance_graph(selected_metric):
        """Update the performance comparison graph based on selected metric."""
        model_data = load_model_results()
        comparison_df = model_data['comparison_df']
        initial_results = model_data['initial_results']
        tuned_results = model_data['tuned_results']
        
        # Get original and tuned values for the selected metric
        if selected_metric == 'Accuracy':
            original_values = comparison_df['Original Accuracy']
            tuned_values = comparison_df['Tuned Accuracy']
        elif selected_metric == 'F1 Score':
            original_values = comparison_df['Original F1']
            tuned_values = comparison_df['Tuned F1']
        else:
            # For Precision and Recall, extract directly from the DataFrames
            original_values = initial_results[selected_metric]
            tuned_values = tuned_results[selected_metric]
        
        fig = go.Figure(data=[
            go.Bar(
                name='Original', 
                x=comparison_df.index, 
                y=original_values, 
                marker_color=CHART_COLORS['bar_original']
            ),
            go.Bar(
                name='Tuned', 
                x=comparison_df.index, 
                y=tuned_values, 
                marker_color=CHART_COLORS['bar_tuned']
            )
        ])
        
        fig.update_layout(
            title=f'{selected_metric}: Original vs Tuned Models',
            xaxis_title='Model',
            yaxis_title=selected_metric,
            barmode='group',
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=40, r=40, t=60, b=60),
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=18)
        )
        
        # Improve appearance with grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        
        return fig
    
    # ROC curve callback
    @app.callback(
        Output('roc-graph', 'figure'),
        [Input('metric-dropdown', 'value')]  # Just a trigger, not actually used
    )
    def update_roc_graph(_):
        """Update the ROC curves graph."""
        roc_data = load_roc_data()
        model_data = load_model_results()
        model_names = model_data['model_names']
        
        fig = go.Figure()
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color=CHART_COLORS['line_random'])
        ))
        
        # Add ROC curve for each model
        colors = CHART_COLORS['model_colors']
        
        for i, name in enumerate(model_names):
            if name in roc_data:
                model_roc = roc_data[name]
                roc_curve_data = model_roc['data']
                roc_auc = model_roc['auc']
                
                fig.add_trace(go.Scatter(
                    x=roc_curve_data['fpr'], 
                    y=roc_curve_data['tpr'],
                    mode='lines',
                    name=f'{name} (AUC = {roc_auc:.3f})',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title='ROC Curves for Tuned Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.01, y=0.01, bordercolor='#e8e8e8', borderwidth=1),
            margin=dict(l=40, r=40, t=60, b=60),
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=18)
        )
        
        # Improve appearance with grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        
        return fig
    
    # Confusion matrix callback
    @app.callback(
        Output('confusion-matrix', 'figure'),
        [Input('model-dropdown', 'value')]
    )
    def update_confusion_matrix(selected_model):
        """Update the confusion matrix based on selected model."""
        cm = load_confusion_matrix(selected_model)
        
        if cm is not None:
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Blues',
                showscale=False
            ))
            
            # Add text annotations
            annotations = []
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    annotations.append(dict(
                        text=str(int(cm[i][j])),
                        x=['Predicted 0', 'Predicted 1'][j],
                        y=['Actual 0', 'Actual 1'][i],
                        font=dict(color='white' if cm[i][j] > np.max(cm)/2 else 'black', size=14),
                        showarrow=False
                    ))
            
            fig.update_layout(
                title=f'Confusion Matrix: {selected_model}',
                annotations=annotations,
                margin=dict(l=40, r=40, t=60, b=60),
                plot_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=18)
            )
            
            return fig
        
        # Return empty figure if no data
        return go.Figure()
    
    # Feature importance callback
    @app.callback(
        Output('feature-importance-graph', 'figure'),
        [Input('feature-model-dropdown', 'value')]
    )
    def update_feature_importance(selected_model):
        """Update the feature importance graph based on selected model."""
        try:
            if not selected_model:
                # Return empty figure with message
                fig = go.Figure()
                fig.update_layout(
                    title='Feature Importance',
                    annotations=[{
                        'text': 'Please select a model to view feature importance',
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'showarrow': False,
                        'font': {'size': 16}
                    }]
                )
                return fig
                
            feature_importance = load_feature_importance(selected_model)
            
            if feature_importance is None:
                # Return empty figure with message
                fig = go.Figure()
                fig.update_layout(
                    title=f'No feature importance data available for {selected_model}',
                    annotations=[{
                        'text': 'No feature importance data available',
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'showarrow': False,
                        'font': {'size': 16}
                    }]
                )
                return fig
            
            # If we have a DataFrame with one column, convert to series
            if isinstance(feature_importance, pd.DataFrame):
                if feature_importance.shape[1] == 1:
                    feature_importance = feature_importance.iloc[:, 0]
            
            # Now process based on what we have
            if isinstance(feature_importance, pd.Series):
                # Sort the Series
                sorted_importance = feature_importance.sort_values(ascending=False)
                x_values = sorted_importance.index
                y_values = sorted_importance.values
            else:
                # Handle DataFrame case
                if 'feature' in feature_importance.columns and 'importance' in feature_importance.columns:
                    # Standard format with feature and importance columns
                    sorted_importance = feature_importance.sort_values(by='importance', ascending=False)
                    x_values = sorted_importance['feature']
                    y_values = sorted_importance['importance']
                else:
                    # Just use the first column as index and second as values
                    first_col = feature_importance.columns[0]
                    feature_importance = feature_importance.set_index(first_col)
                    sorted_importance = feature_importance.iloc[:, 0].sort_values(ascending=False)
                    x_values = sorted_importance.index
                    y_values = sorted_importance.values
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_values,
                y=y_values,
                marker_color=CHART_COLORS['bar_tuned']
            ))
            
            fig.update_layout(
                title=f'Feature Importance: {selected_model}',
                xaxis_title='Feature',
                yaxis_title='Importance',
                margin=dict(l=40, r=40, t=60, b=60),
                plot_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=18)
            )
            
            # Improve appearance with grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            
            return fig
            
        except Exception as e:
            # Return empty figure with error message
            fig = go.Figure()
            fig.update_layout(
                title=f'Error loading feature importance for {selected_model}',
                annotations=[{
                    'text': f'Error: {str(e)}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return fig
    
    # Model details callback
    @app.callback(
        Output('model-details', 'children'),
        [Input('model-dropdown', 'value')]
    )
    def update_model_details(selected_model):
        """Update the model hyperparameters display based on selected model."""
        try:
            if not selected_model:
                return html.P("Please select a model to view details.")
                
            model_details = load_model_details()
            
            if model_details is None:
                return html.P("No model details available.")
            
            # Look for a row corresponding to the selected model
            # The model name might be in the index or in a column called 'model'
            model_row = None
            
            if selected_model in model_details.index:
                model_row = model_details.loc[selected_model]
            elif 'model' in model_details.columns and selected_model in model_details['model'].values:
                model_row = model_details[model_details['model'] == selected_model].iloc[0]
            else:
                # Try first row as a fallback (might be applicable to all models)
                model_row = model_details.iloc[0]
            
            # Extract and format parameters
            details = []
            
            # First check for model__ prefixed parameters which is common in sklearn pipelines
            model_params = {}
            for column in model_details.columns:
                if column.startswith('model__'):
                    param_name = column.replace('model__', '')
                    if isinstance(model_row, pd.Series):
                        value = model_row[column]
                    else:
                        value = model_details.loc[0, column]
                    model_params[param_name] = value
            
            # If no model__ params found, use all columns except obvious non-parameter ones
            if not model_params:
                excluded_cols = ['model', 'score', 'time', 'rank']
                for column in model_details.columns:
                    if column.lower() not in excluded_cols:
                        if isinstance(model_row, pd.Series):
                            value = model_row[column]
                        else:
                            value = model_details.loc[0, column]
                        model_params[column] = value
            
            # Format parameters as a list of details
            for param, value in model_params.items():
                try:
                    # Try to parse JSON for complex values
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            parsed_value = json.loads(value.replace("'", '"'))
                            formatted_value = str(parsed_value)
                        except:
                            formatted_value = str(value)
                    else:
                        formatted_value = str(value)
                    
                    # Skip None or NaN values
                    if formatted_value.lower() == 'none' or formatted_value.lower() == 'nan':
                        continue
                        
                    details.append(html.Div([
                        html.Strong(f"{param}: "),
                        html.Span(formatted_value)
                    ], style={'marginBottom': '8px'}))
                except:
                    pass
            
            if not details:
                return html.P("No hyperparameter details available for this model.")
            
            return [
                html.H3(f"Hyperparameters for {selected_model}", style={'marginBottom': '15px'}),
                html.Div(details)
            ]
            
        except Exception as e:
            return html.P(f"Error loading model details: {str(e)}")