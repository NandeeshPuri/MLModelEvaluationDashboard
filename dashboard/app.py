"""
Main application file for the Machine Learning Model Comparison Dashboard.
This file initializes the Dash app and sets up the layout and callbacks.
"""
import dash
from components.layout import create_layout
from components.callbacks import register_callbacks
from config import DASHBOARD_TITLE

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # For deploying to production

# Set the app layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

# App title
app.title = DASHBOARD_TITLE

if __name__ == '__main__':
    app.run_server(debug=True)