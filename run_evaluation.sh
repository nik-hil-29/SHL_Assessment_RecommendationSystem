#!/bin/bash

# Create evaluation directory if it doesn't exist
mkdir -p evaluation

# Check if test_data.json exists, if not copy it from the sample
if [ ! -f "evaluation/test_data.json" ]; then
    echo "Copying test data to evaluation directory..."
    cp test_data.json evaluation/
fi

# Run the evaluation
echo "Running evaluation..."
python evaluate.py --api-url http://localhost:8000 --test-data evaluation/test_data.json --output evaluation/results.json --k-values 3 5 10

# Display results
echo "Evaluation complete. Results saved to evaluation/results.json"
echo "Summary of results:"
python -c "import json; data = json.load(open('evaluation/results.json')); print('\n'.join([f'{k}: {v:.4f}' for k, v in data['mean_metrics'].items()]))"#!/bin/bash

# Create evaluation directory if it doesn't exist
mkdir -p evaluation

# Check if test_data.json exists, if not copy it from the sample
if [ ! -f "evaluation/test_data.json" ]; then
    echo "Copying test data to evaluation directory..."
    cp test_data.json evaluation/
fi

# Run the debug evaluation script
echo "Running debug evaluation..."
python debug_evaluation.py

echo "Debug evaluation complete. Results saved to evaluation/debug_results.json"