# scripts/web_gui.py
#!/usr/bin/env python3
"""
Web-based GUI for Stock LNN Analysis
Allows users to select stocks, configure parameters, and run analysis through a web interface.

Usage:
    python scripts/web_gui.py
    Then open browser to http://localhost:8050
"""

import os
import sys
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, date
import json
import threading
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.experiment_tracker import ExperimentTracker
from data.data_loader import StockDataLoader

# Initialize Dash app
app = dash.Dash(__name__, title="Stock LNN Analysis")
app.config.suppress_callback_exceptions = True

# Global variables
experiment_tracker = ExperimentTracker()
analysis_status = {"running": False, "progress": "", "results": None}

# Popular stock tickers for easy selection
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR",
    "SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "VIX"
]

MARKET_INDICES = [
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"
]

def create_layout():
    """Create the main GUI layout."""
    return html.Div([
        # Header
        html.Div([
            html.H1("🚀 Stock LNN Analysis Platform", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            html.P("Comprehensive AI-powered stock market analysis using Liquid Neural Networks",
                   style={'textAlign': 'center', 'fontSize': 18, 'color': '#7f8c8d'})
        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginBottom': '20px'}),
        
        # Main content
        html.Div([
            # Left panel - Configuration
            html.Div([
                html.H3("📊 Analysis Configuration", style={'color': '#34495e'}),
                
                # Stock Selection
                html.Div([
                    html.Label("Target Stock:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='target-stock',
                        options=[{'label': f"{ticker} - Popular", 'value': ticker} for ticker in POPULAR_TICKERS],
                        value='AAPL',
                        placeholder="Select or type a stock symbol",
                        searchable=True,
                        style={'marginBottom': '15px'}
                    ),
                    
                    html.Label("Market Context (for feature engineering):", 
                              style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='context-stocks',
                        options=[{'label': f"{ticker} - Index", 'value': ticker} for ticker in MARKET_INDICES] + 
                                [{'label': f"{ticker} - Popular", 'value': ticker} for ticker in POPULAR_TICKERS[:10]],
                        value=['^GSPC', 'QQQ', 'GLD'],
                        multi=True,
                        placeholder="Select market context",
                        style={'marginBottom': '20px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                # Date Range
                html.Div([
                    html.Label("Analysis Period:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=date(2020, 1, 1),
                        end_date=date.today(),
                        display_format='YYYY-MM-DD',
                        style={'marginBottom': '20px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                # Model Configuration
                html.Div([
                    html.H4("🧠 Model Parameters", style={'color': '#2980b9'}),
                    
                    html.Label("Hidden Layer Size:", style={'fontWeight': 'bold'}),
                    dcc.Slider(
                        id='hidden-size',
                        min=25, max=200, step=25, value=50,
                        marks={i: str(i) for i in range(25, 201, 25)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Sequence Length (days):", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                    dcc.Slider(
                        id='sequence-length',
                        min=10, max=60, step=5, value=30,
                        marks={i: str(i) for i in range(10, 61, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Learning Rate:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                    dcc.Dropdown(
                        id='learning-rate',
                        options=[
                            {'label': '0.0001 (Conservative)', 'value': 0.0001},
                            {'label': '0.0005 (Moderate)', 'value': 0.0005},
                            {'label': '0.001 (Standard)', 'value': 0.001},
                            {'label': '0.005 (Aggressive)', 'value': 0.005}
                        ],
                        value=0.001
                    )
                ], style={'marginBottom': '20px'}),
                
                # Analysis Options
                html.Div([
                    html.H4("🔍 Analysis Options", style={'color': '#27ae60'}),
                    
                    dcc.Checklist(
                        id='analysis-options',
                        options=[
                            {'label': ' Advanced Feature Engineering', 'value': 'advanced_features'},
                            {'label': ' Pattern Recognition', 'value': 'patterns'},
                            {'label': ' Temporal Analysis', 'value': 'temporal'},
                            {'label': ' Dimensionality Reduction', 'value': 'dim_reduction'},
                            {'label': ' Quick Mode (faster)', 'value': 'quick_mode'}
                        ],
                        value=['advanced_features', 'patterns'],
                        style={'marginBottom': '20px'}
                    )
                ], style={'marginBottom': '30px'}),
                
                # Run Analysis Button
                html.Div([
                    html.Button(
                        "🚀 Run Analysis", 
                        id='run-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#3498db',
                            'color': 'white',
                            'border': 'none',
                            'padding': '15px 30px',
                            'fontSize': '16px',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'width': '100%'
                        }
                    )
                ], style={'textAlign': 'center'})
                
            ], style={
                'width': '30%', 
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '10px',
                'marginRight': '20px',
                'height': 'fit-content'
            }),
            
            # Right panel - Results
            html.Div([
                html.H3("📈 Analysis Results", style={'color': '#34495e'}),
                
                # Status indicator
                html.Div(id='status-indicator', style={'marginBottom': '20px'}),
                
                # Progress bar
                html.Div(id='progress-bar', style={'marginBottom': '20px'}),
                
                # Results tabs
                dcc.Tabs(id='results-tabs', value='overview', children=[
                    dcc.Tab(label='📊 Overview', value='overview'),
                    dcc.Tab(label='💹 Performance', value='performance'), 
                    dcc.Tab(label='🔍 Patterns', value='patterns'),
                    dcc.Tab(label='📋 History', value='history')
                ]),
                
                # Results content
                html.Div(id='results-content', style={'marginTop': '20px'})
                
            ], style={
                'width': '65%',
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '10px',
                'border': '1px solid #bdc3c7'
            })
            
        ], style={'display': 'flex', 'padding': '20px'}),
        
        # Footer
        html.Div([
            html.P("Powered by Liquid Neural Networks on NVIDIA Jetson", 
                   style={'textAlign': 'center', 'color': '#95a5a6', 'marginTop': '30px'})
        ]),
        
        # Hidden div to store analysis results
        html.Div(id='analysis-results-store', style={'display': 'none'}),
        
        # Auto-refresh component
        dcc.Interval(
            id='interval-component',
            interval=2000,  # Update every 2 seconds
            n_intervals=0
        )
    ])

# Callback for running analysis
@app.callback(
    [Output('status-indicator', 'children'),
     Output('progress-bar', 'children'),
     Output('analysis-results-store', 'children')],
    [Input('run-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('target-stock', 'value'),
     State('context-stocks', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('hidden-size', 'value'),
     State('sequence-length', 'value'),
     State('learning-rate', 'value'),
     State('analysis-options', 'value')]
)
def update_analysis(n_clicks, n_intervals, target_stock, context_stocks, start_date, end_date,
                   hidden_size, sequence_length, learning_rate, analysis_options):
    """Handle analysis execution and status updates."""
    
    # Check if analysis button was clicked
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'run-button.n_clicks' and n_clicks > 0:
        # Start analysis in background thread
        if not analysis_status["running"]:
            config = {
                'target_stock': target_stock,
                'context_stocks': context_stocks or [],
                'start_date': start_date,
                'end_date': end_date,
                'hidden_size': hidden_size,
                'sequence_length': sequence_length,
                'learning_rate': learning_rate,
                'analysis_options': analysis_options or []
            }
            
            thread = threading.Thread(target=run_analysis_background, args=(config,))
            thread.daemon = True
            thread.start()
    
    # Update status display
    if analysis_status["running"]:
        status = html.Div([
            html.Span("🔄 ", style={'fontSize': '20px'}),
            html.Span("Analysis Running...", style={'color': '#f39c12', 'fontWeight': 'bold'})
        ])
        
        progress = html.Div([
            html.Div([
                html.Div(
                    analysis_status["progress"],
                    style={
                        'width': '100%',
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'textAlign': 'center',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'animation': 'pulse 2s infinite'
                    }
                )
            ])
        ])
        
        return status, progress, ""
    
    elif analysis_status["results"]:
        status = html.Div([
            html.Span("✅ ", style={'fontSize': '20px'}),
            html.Span("Analysis Complete!", style={'color': '#27ae60', 'fontWeight': 'bold'})
        ])
        
        progress = html.Div()

        return status, progress, json.dumps(analysis_status["results"], cls=NumpyEncoder)
    
    else:
        status = html.Div([
            html.Span("⚡ ", style={'fontSize': '20px'}),
            html.Span("Ready to Analyze", style={'color': '#7f8c8d'})
        ])
        
        progress = html.Div()
        
        return status, progress, ""

def run_analysis_background(config):
    """Run REAL analysis in background thread."""
    analysis_status["running"] = True
    analysis_status["progress"] = "Starting real analysis..."
    
    try:
        # Create temporary config file for this analysis
        import tempfile
        import yaml
        
        analysis_status["progress"] = "Preparing configuration..."
        
        # Create real config based on GUI selections
        real_config = {
            'data': {
                'tickers': config['context_stocks'] + [config['target_stock']],
                'start_date': config['start_date'],
                'end_date': config['end_date'], 
                'target_ticker': config['target_stock']
            },
            'model': {
                'sequence_length': config['sequence_length'],
                'hidden_size': config['hidden_size'],
                'learning_rate': config['learning_rate'],
                'batch_size': 32,
                'num_epochs': 50,  # Shorter for GUI
                'patience': 10
            },
            'analysis': {
                'use_advanced_features': 'advanced_features' in config['analysis_options'],
                'pattern_analysis': 'patterns' in config['analysis_options'],
                'temporal_analysis': 'temporal' in config['analysis_options']
            }
        }
        
        # Save temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(real_config, f)
            temp_config_path = f.name
        
        analysis_status["progress"] = "Running real analysis..."
        
        # Import and run real analysis
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from scripts.run_analysis import ComprehensiveAnalyzer
        
        # Run real analysis
        analyzer = ComprehensiveAnalyzer(
            config_path=temp_config_path,
            experiment_name=f"gui_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        results = analyzer.run_complete_analysis(phases=['data', 'training', 'evaluation'])
        
        # Extract real results
        if 'evaluation' in results and results['evaluation']:
            eval_metrics = results['evaluation'].get('metrics', {})
            
            trading_metrics = eval_metrics.get('trading_metrics', {})
            directional_metrics = eval_metrics.get('directional_metrics', {})
            
            analysis_status["results"] = {
                'target_stock': config['target_stock'],
                'total_return': trading_metrics.get('total_return', 0),
                'sharpe_ratio': trading_metrics.get('sharpe_ratio', 0),
                'max_drawdown': trading_metrics.get('max_drawdown', 0),
                'directional_accuracy': directional_metrics.get('directional_accuracy', 0),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Fallback if analysis fails
            analysis_status["results"] = {'error': 'Analysis failed to complete'}
        
        # Clean up temp file
        os.unlink(temp_config_path)
        
    except Exception as e:
        analysis_status["results"] = {'error': str(e)}
    
    finally:
        analysis_status["running"] = False
        analysis_status["progress"] = ""

# Callback for updating results display
@app.callback(
    Output('results-content', 'children'),
    [Input('results-tabs', 'value'),
     Input('analysis-results-store', 'children')]
)
def update_results_display(active_tab, results_json):
    """Update the results display based on selected tab."""
    
    if not results_json:
        return html.Div([
            html.P("No analysis results yet. Configure your analysis and click 'Run Analysis' to get started.",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontStyle': 'italic'})
        ])
    
    try:
        results = json.loads(results_json)
    except:
        return html.Div("Error loading results")
    
    if 'error' in results:
        return html.Div([
            html.H4("❌ Analysis Error", style={'color': '#e74c3c'}),
            html.P(f"Error: {results['error']}")
        ])
    
    if active_tab == 'overview':
        return create_overview_tab(results)
    elif active_tab == 'performance':
        return create_performance_tab(results)
    elif active_tab == 'patterns':
        return create_patterns_tab(results)
    elif active_tab == 'history':
        return create_history_tab()
    
    return html.Div("Tab content not implemented yet")

def create_overview_tab(results):
    """Create overview tab content."""
    return html.Div([
        html.H4(f"📊 Analysis Overview - {results['target_stock']}", style={'color': '#2c3e50'}),
        
        # Key metrics cards
        html.Div([
            html.Div([
                html.H3(f"{results['total_return']:.1%}", style={'color': '#27ae60', 'margin': '0'}),
                html.P("Total Return", style={'margin': '0', 'color': '#7f8c8d'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                     'borderRadius': '10px', 'width': '22%', 'margin': '1%'}),
            
            html.Div([
                html.H3(f"{results['sharpe_ratio']:.2f}", style={'color': '#3498db', 'margin': '0'}),
                html.P("Sharpe Ratio", style={'margin': '0', 'color': '#7f8c8d'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                     'borderRadius': '10px', 'width': '22%', 'margin': '1%'}),
            
            html.Div([
                html.H3(f"{results['max_drawdown']:.1%}", style={'color': '#e74c3c', 'margin': '0'}),
                html.P("Max Drawdown", style={'margin': '0', 'color': '#7f8c8d'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                     'borderRadius': '10px', 'width': '22%', 'margin': '1%'}),
            
            html.Div([
                html.H3(f"{results['directional_accuracy']:.1%}", style={'color': '#9b59b6', 'margin': '0'}),
                html.P("Directional Accuracy", style={'margin': '0', 'color': '#7f8c8d'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                     'borderRadius': '10px', 'width': '22%', 'margin': '1%'})
            
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
        # Analysis timestamp
        html.P(f"Analysis completed: {results['timestamp'][:19]}", 
               style={'color': '#95a5a6', 'fontStyle': 'italic'})
    ])

def create_performance_tab(results):
    """Create performance tab content."""
    return html.Div([
        html.H4("💹 Performance Analysis", style={'color': '#2c3e50'}),
        html.P("Detailed performance charts and metrics will be displayed here."),
        html.P("(Implementation in progress)")
    ])

def create_patterns_tab(results):
    """Create patterns tab content.""" 
    return html.Div([
        html.H4("🔍 Pattern Analysis", style={'color': '#2c3e50'}),
        html.P("Market patterns and technical analysis will be displayed here."),
        html.P("(Implementation in progress)")
    ])

def create_history_tab():
    """Create history tab content."""
    experiments = experiment_tracker.get_experiments_dataframe()
    
    if experiments.empty:
        return html.Div([
            html.P("No previous experiments found.",
                   style={'textAlign': 'center', 'color': '#7f8c8d'})
        ])
    
    return html.Div([
        html.H4("📋 Experiment History", style={'color': '#2c3e50'}),
        dash_table.DataTable(
            data=experiments.head(10).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in ['experiment_name', 'timestamp']],
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': '#3498db', 'color': 'white'}
        )
    ])

# Set app layout
app.layout = create_layout()

def main():
    """Run the web GUI."""
    print("=" * 60)
    print("🚀 STOCK LNN ANALYSIS - WEB GUI")
    print("=" * 60)
    print("Starting web interface...")
    print("Open your browser to: http://localhost:8050")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=8050)

if __name__ == "__main__":
    main()
