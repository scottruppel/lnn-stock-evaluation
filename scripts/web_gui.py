#!/usr/bin/env python3
"""
Enhanced Web GUI for Liquid Neural Network Stock Analysis
Provides interface for both single stock and batch analysis with network access support.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
import socket
import subprocess

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your analysis components
try:
    from run_analysis import ComprehensiveAnalyzer
    from data.data_loader import StockDataLoader
    print("✓ Successfully imported analysis modules")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Some features may be limited")

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "LNN Stock Analysis Dashboard"

# Global variables for analysis state
analysis_status = {
    'running': False,
    'progress': 0,
    'current_phase': '',
    'results': None,
    'error': None
}

# Predefined asset class groups for batch analysis
ASSET_CLASSES = {
    'Large Cap Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
    'Market Indices': ['^GSPC', '^DJI', '^IXIC', '^RUT'],
    'ETFs - Broad Market': ['SPY', 'QQQ', 'IWM', 'VTI'],
    'ETFs - Sector': ['XLF', 'XLE', 'XLK', 'XLV', 'XLI'],
    'ETFs - Fixed Income': ['AGG', 'TLT', 'HYG', 'LQD'],
    'Commodities': ['GLD', 'SLV', 'USO', 'UNG'],
    'International': ['EFA', 'EEM', 'VEA', 'VWO'],
    'Custom': []  # User-defined list
}

# CSS styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🧠 Liquid Neural Network Stock Analysis", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.H3("AI-Powered Market Intelligence Dashboard", 
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': 0}),
        html.Hr()
    ]),
    
    # Network Status
    html.Div(id='network-status', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#e8f5e8', 'borderRadius': '5px'}),
    
    # Main Control Panel
    html.Div([
        html.H3("📊 Analysis Configuration"),
        
        # Analysis Type Selection
        html.Div([
            html.Label("Analysis Type:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='analysis-type',
                options=[
                    {'label': '🎯 Single Stock Analysis (with similar assets)', 'value': 'single'},
                    {'label': '📦 Batch Analysis (multiple stocks)', 'value': 'batch'}
                ],
                value='single',
                style={'margin': '10px 0'}
            )
        ], style={'marginBottom': '20px'}),
        
        # Single Stock Configuration
        html.Div(id='single-stock-config', children=[
            html.H4("Single Stock Configuration"),
            html.Div([
                html.Label("Target Stock Symbol:", style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='target-stock',
                    type='text',
                    placeholder='Enter stock symbol (e.g., AAPL)',
                    value='AAPL',
                    style={'width': '200px', 'margin': '5px'}
                )
            ]),
            html.Div([
                html.Label("Include Similar Assets:", style={'fontWeight': 'bold', 'display': 'block'}),
                dcc.Checklist(
                    id='similar-assets',
                    options=[
                        {'label': 'Market Index (^GSPC)', 'value': '^GSPC'},
                        {'label': 'Sector ETF (auto-detect)', 'value': 'sector_etf'},
                        {'label': 'Broad Market ETF (SPY)', 'value': 'SPY'},
                        {'label': 'Tech ETF (QQQ)', 'value': 'QQQ'},
                        {'label': 'Bonds (AGG)', 'value': 'AGG'}
                    ],
                    value=['^GSPC', 'SPY'],
                    style={'margin': '10px 0'}
                )
            ])
        ]),
        
        # Batch Analysis Configuration
        html.Div(id='batch-config', children=[
            html.H4("Batch Analysis Configuration"),
            html.Div([
                html.Label("Asset Class:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='asset-class-dropdown',
                    options=[{'label': k, 'value': k} for k in ASSET_CLASSES.keys()],
                    value='Large Cap Tech',
                    style={'margin': '10px 0'}
                )
            ]),
            html.Div([
                html.Label("Custom Stock List (comma-separated):", style={'fontWeight': 'bold'}),
                dcc.Textarea(
                    id='custom-stocks',
                    placeholder='Enter custom symbols: AAPL,MSFT,GOOGL...',
                    style={'width': '100%', 'height': '60px', 'margin': '5px 0'}
                )
            ])
        ], style={'display': 'none'}),
        
        # Analysis Parameters
        html.Div([
            html.H4("Analysis Parameters"),
            html.Div([
                html.Div([
                    html.Label("Date Range:", style={'fontWeight': 'bold'}),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=(datetime.now() - timedelta(days=365*2)).date(),
                        end_date=datetime.now().date(),
                        display_format='YYYY-MM-DD'
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Analysis Depth:", style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='analysis-depth',
                        options=[
                            {'label': '⚡ Quick (Basic features)', 'value': 'quick'},
                            {'label': '🔍 Standard (All features)', 'value': 'standard'},
                            {'label': '🎯 Deep (Full analysis + advanced)', 'value': 'deep'}
                        ],
                        value='standard',
                        style={'margin': '5px 0'}
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ])
        ], style={'marginBottom': '20px'}),
        
        # Control Buttons
        html.Div([
            html.Button('🚀 Start Analysis', id='start-btn', n_clicks=0, 
                       style={'backgroundColor': '#27ae60', 'color': 'white', 'padding': '10px 20px', 
                             'border': 'none', 'borderRadius': '5px', 'marginRight': '10px', 'fontSize': '16px'}),
            html.Button('⏹️ Stop Analysis', id='stop-btn', n_clicks=0,
                       style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '10px 20px',
                             'border': 'none', 'borderRadius': '5px', 'marginRight': '10px', 'fontSize': '16px'}),
            html.Button('📁 Load Previous Results', id='load-btn', n_clicks=0,
                       style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                             'border': 'none', 'borderRadius': '5px', 'fontSize': '16px'})
        ], style={'textAlign': 'center', 'margin': '20px 0'})
        
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Progress Section
    html.Div(id='progress-section', children=[
        html.H3("📈 Analysis Progress"),
        html.Div(id='progress-info'),
        dcc.Interval(id='progress-interval', interval=2000, n_intervals=0)  # Update every 2 seconds
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#fff3cd', 'borderRadius': '10px', 'display': 'none'}),
    
    # Results Section
    html.Div(id='results-section', children=[
        html.H3("📊 Analysis Results"),
        html.Div(id='results-content')
    ], style={'margin': '20px'})
])

# Callbacks for dynamic UI updates
@app.callback(
    [Output('single-stock-config', 'style'),
     Output('batch-config', 'style')],
    [Input('analysis-type', 'value')]
)
def toggle_config_sections(analysis_type):
    if analysis_type == 'single':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

@app.callback(
    Output('network-status', 'children'),
    [Input('progress-interval', 'n_intervals')]  # Update periodically
)
def update_network_status(n):
    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        return html.Div([
            html.Strong("🌐 Network Status: "),
            f"Running on {hostname} ({local_ip}:8050) | ",
            html.A("Local Access", href="http://localhost:8050", target="_blank"),
            " | ",
            html.A("Network Access", href=f"http://{local_ip}:8050", target="_blank")
        ])
    except:
        return html.Div([
            html.Strong("🌐 Network Status: "),
            "Local only | ",
            html.A("Local Access", href="http://localhost:8050", target="_blank")
        ])

@app.callback(
    [Output('progress-section', 'style'),
     Output('progress-info', 'children')],
    [Input('progress-interval', 'n_intervals')]
)
def update_progress(n):
    global analysis_status
    
    if analysis_status['running']:
        progress_bar = html.Div([
            html.Div([
                html.Div(style={
                    'width': f"{analysis_status['progress']}%",
                    'backgroundColor': '#28a745',
                    'height': '20px',
                    'borderRadius': '10px',
                    'transition': 'width 0.5s'
                })
            ], style={
                'width': '100%',
                'backgroundColor': '#e9ecef',
                'borderRadius': '10px',
                'marginBottom': '10px'
            }),
            html.P(f"Current Phase: {analysis_status['current_phase']}", style={'margin': '5px 0'}),
            html.P(f"Progress: {analysis_status['progress']:.1f}%", style={'margin': '5px 0'})
        ])
        
        return {'margin': '20px', 'padding': '20px', 'backgroundColor': '#fff3cd', 'borderRadius': '10px', 'display': 'block'}, progress_bar
    else:
        return {'display': 'none'}, ""

@app.callback(
    Output('results-content', 'children'),
    [Input('start-btn', 'n_clicks'),
     Input('load-btn', 'n_clicks')],
    [State('analysis-type', 'value'),
     State('target-stock', 'value'),
     State('similar-assets', 'value'),
     State('asset-class-dropdown', 'value'),
     State('custom-stocks', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('analysis-depth', 'value')]
)
def handle_analysis(start_clicks, load_clicks, analysis_type, target_stock, similar_assets, 
                   asset_class, custom_stocks, start_date, end_date, analysis_depth):
    global analysis_status
    
    ctx = callback_context
    if not ctx.triggered:
        return "Ready to analyze. Configure your parameters and click 'Start Analysis'."
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-btn' and start_clicks > 0:
        # Validate inputs
        if analysis_type == 'single' and not target_stock:
            return html.Div("❌ Please enter a target stock symbol.", style={'color': 'red'})
        
        if analysis_type == 'batch' and asset_class == 'Custom' and not custom_stocks:
            return html.Div("❌ Please enter custom stock symbols for batch analysis.", style={'color': 'red'})
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(
            target=run_analysis_background,
            args=(analysis_type, target_stock, similar_assets, asset_class, custom_stocks, 
                 start_date, end_date, analysis_depth)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return html.Div("🚀 Analysis started! Check progress above.", style={'color': 'green'})
    
    elif button_id == 'load-btn' and load_clicks > 0:
        return load_previous_results()
    
    return "Ready to analyze."

def run_analysis_background(analysis_type, target_stock, similar_assets, asset_class, 
                          custom_stocks, start_date, end_date, analysis_depth):
    """Run analysis in background thread with progress updates."""
    global analysis_status
    
    try:
        analysis_status['running'] = True
        analysis_status['progress'] = 0
        analysis_status['current_phase'] = 'Initializing...'
        analysis_status['error'] = None
        
        # Prepare configuration
        if analysis_type == 'single':
            tickers = [target_stock]
            if similar_assets:
                tickers.extend(similar_assets)
            tickers = list(set(tickers))  # Remove duplicates
        else:
            if asset_class == 'Custom' and custom_stocks:
                tickers = [s.strip().upper() for s in custom_stocks.split(',')]
            else:
                tickers = ASSET_CLASSES.get(asset_class, ['AAPL'])
        
        config = {
            'data': {
                'tickers': tickers,
                'target_ticker': target_stock if analysis_type == 'single' else tickers[0],
                'start_date': start_date,
                'end_date': end_date
            },
            'analysis': {
                'use_advanced_features': analysis_depth in ['standard', 'deep'],
                'pattern_analysis': True,
                'temporal_analysis': analysis_depth == 'deep',
                'dimensionality_reduction': analysis_depth in ['standard', 'deep']
            },
            'model': {
                'sequence_length': 30,
                'hidden_size': 50 if analysis_depth == 'quick' else 100,
                'num_epochs': 50 if analysis_depth == 'quick' else 100,
                'learning_rate': 0.001,
                'batch_size': 32,
                'patience': 10
            }
        }
        
        # Save temporary config
        temp_config_path = 'config/temp_gui_config.yaml'
        os.makedirs('config', exist_ok=True)
        
        import yaml
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run analysis with progress tracking
        experiment_name = f"gui_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        analysis_status['current_phase'] = 'Data Loading'
        analysis_status['progress'] = 10
        
        analyzer = ComprehensiveAnalyzer(
            config_path=temp_config_path,
            experiment_name=experiment_name
        )
        
        # Determine phases based on analysis depth
        if analysis_depth == 'quick':
            phases = ['data', 'training', 'evaluation']
        elif analysis_depth == 'standard':
            phases = ['data', 'features', 'training', 'evaluation', 'report']
        else:  # deep
            phases = ['data', 'features', 'training', 'evaluation', 'report']
        
        # Run analysis with progress updates
        phase_progress = {
            'data': 30,
            'features': 50,
            'training': 70,
            'evaluation': 90,
            'report': 100
        }
        
        for phase in phases:
            analysis_status['current_phase'] = f'Running {phase} analysis...'
            analysis_status['progress'] = phase_progress.get(phase, 50)
            
            # This is a simplified version - you might need to modify ComprehensiveAnalyzer
            # to support individual phase execution with progress callbacks
            time.sleep(2)  # Simulate work
        
        # For now, run the complete analysis
        analysis_status['current_phase'] = 'Running complete analysis...'
        results = analyzer.run_complete_analysis(phases=phases)
        
        analysis_status['results'] = results
        analysis_status['progress'] = 100
        analysis_status['current_phase'] = 'Complete!'
        analysis_status['running'] = False
        
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            
    except Exception as e:
        analysis_status['error'] = str(e)
        analysis_status['running'] = False
        analysis_status['current_phase'] = f'Error: {str(e)}'
        print(f"Analysis error: {e}")

def load_previous_results():
    """Load and display previous analysis results."""
    try:
        # Look for recent result files
        results_dir = 'results/reports'
        if not os.path.exists(results_dir):
            return "No previous results found."
        
        # Find recent JSON files
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            return "No previous results found."
        
        # Sort by modification time (most recent first)
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        
        recent_files = json_files[:5]  # Show 5 most recent
        
        return html.Div([
            html.H4("Recent Analysis Results:"),
            html.Ul([
                html.Li([
                    html.A(f, href=f"/results/reports/{f}", target="_blank"),
                    f" (Modified: {datetime.fromtimestamp(os.path.getmtime(os.path.join(results_dir, f))).strftime('%Y-%m-%d %H:%M')})"
                ]) for f in recent_files
            ])
        ])
        
    except Exception as e:
        return f"Error loading results: {str(e)}"

def check_matplotlib_backend():
    """Check and configure matplotlib for headless operation."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        print("✓ Matplotlib configured for headless operation")
        return True
    except Exception as e:
        print(f"⚠️  Matplotlib configuration issue: {e}")
        return False

def main():
    """Run the web GUI with enhanced network support and error handling."""
    print("=" * 60)  
    print("🚀 STOCK LNN ANALYSIS - WEB GUI")
    print("=" * 60)
    print("Starting web interface...")
    
    # Configure matplotlib for headless operation
    if not check_matplotlib_backend():
        print("⚠️  Warning: Matplotlib issues detected. Plots may not work correctly.")
    
    # Get network information
    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        print(f"Hostname: {hostname}")
        print(f"Local IP: {local_ip}")
        print()
        print("ACCESS URLS:")
        print(f"  Local:   http://localhost:8050")
        print(f"  Network: http://{local_ip}:8050")
        print()
        print("GUI FEATURES:")
        print("  ✓ Single stock analysis with similar assets")
        print("  ✓ Batch analysis across asset classes")
        print("  ✓ Configurable analysis depth (Quick/Standard/Deep)")
        print("  ✓ Real-time progress tracking")
        print("  ✓ Network access from other devices")
        print()
        print("TROUBLESHOOTING:")
        print("  1. Ensure all devices are on the same WiFi network")
        print("  2. Check firewall settings if network access fails")
        print("  3. Try restarting if matplotlib errors occur")
        print()
        
    except Exception as e:
        print(f"Network detection error: {e}")
        print("Local access only: http://localhost:8050")
    
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Ensure required directories exist
        os.makedirs('results/reports', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        os.makedirs('config', exist_ok=True)
        
        # Run the app
        app.run(
            host='0.0.0.0',           # Listen on all network interfaces
            port=8050,                # Standard port
            debug=False,              # Disable debug for stability
            threaded=True,            # Handle multiple connections
            dev_tools_hot_reload=False  # Disable hot reload for stability
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\nTROUBLESHOoting suggestions:")
        print("1. Check if another process is using port 8050:")
        print("   netstat -tulpn | grep 8050")
        print("2. Try killing any existing processes:")
        print("   pkill -f web_gui.py")
        print("3. Restart your Jetson Nano if issues persist")

if __name__ == '__main__':
    main()
