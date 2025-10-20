#!/bin/bash

# Music Recommendation Engine - Start All Services
# This script starts both the API and Dashboard

echo "========================================="
echo "Music Recommendation Engine"
echo "========================================="
echo ""

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if models exist
if [ ! -f "models_saved/als_model.pkl" ]; then
    echo "❌ Models not found! Please train models first:"
    echo "   PYTHONPATH=$(pwd) venv/bin/python train_models.py"
    exit 1
fi

echo "Starting services..."
echo ""

# Export PYTHONPATH
export PYTHONPATH="$DIR"

# Start API in background
echo "🚀 Starting API on port 5001..."
venv/bin/python api/app.py > logs/api.log 2>&1 &
API_PID=$!
echo "   API PID: $API_PID"

# Wait a moment for API to start
sleep 2

# Start Dashboard in background
echo "📊 Starting Dashboard on port 8050..."
venv/bin/python dashboard/app.py > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "   Dashboard PID: $DASHBOARD_PID"

# Wait a moment for Dashboard to start
sleep 3

echo ""
echo "========================================="
echo "✅ All services started successfully!"
echo "========================================="
echo ""
echo "📡 API:        http://localhost:5001"
echo "📊 Dashboard:  http://localhost:8050"
echo ""
echo "Process IDs:"
echo "  API:       $API_PID"
echo "  Dashboard: $DASHBOARD_PID"
echo ""
echo "To stop services:"
echo "  kill $API_PID $DASHBOARD_PID"
echo ""
echo "Logs:"
echo "  API:       tail -f logs/api.log"
echo "  Dashboard: tail -f logs/dashboard.log"
echo ""
echo "Test the API:"
echo "  curl http://localhost:5001/health"
echo "  curl 'http://localhost:5001/recommend/0?model=ensemble&n=5'"
echo ""
