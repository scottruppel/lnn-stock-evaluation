# Create a convenient wrapper script
cat > run_batch.sh << 'EOF'
#!/bin/bash
# Batch Analysis Wrapper - handles Python paths automatically

export PYTHONPATH=$PWD/src:$PWD/scripts:$PWD
export MPLBACKEND=Agg  # Prevent GUI backend issues

echo "🚀 Starting batch analysis with proper Python paths..."
echo "PYTHONPATH: $PYTHONPATH"
echo ""

python scripts/batch_analysis.py "$@"
EOF

chmod +x run_batch.sh
