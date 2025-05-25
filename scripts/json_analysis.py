import json
import pandas as pd

# Load results
with open('results/multi_stock_analysis/comprehensive_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)

# Find best performers
best_sharpe = df.loc[df['metrics.sharpe_ratio'].idxmax()]
print(f"Best Sharpe: {best_sharpe['ticker']} with {best_sharpe['metrics.sharpe_ratio']:.3f}")
