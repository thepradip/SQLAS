#!/bin/bash
# AriaSQL — quick local start (no Docker needed)
set -e

echo "=== AriaSQL Local Start ==="

# ── Backend ──────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Checking backend dependencies..."
cd backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created virtual environment."
fi
source venv/bin/activate
pip install -r requirements.txt -q

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "  .env created — fill in your Azure OpenAI credentials:"
    echo "  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME"
    echo ""
    read -p "  Press Enter after editing .env to continue..."
fi

echo "[2/4] Loading sample data..."
python ingest.py 2>/dev/null && echo "  health.db ready." || echo "  health.db already exists."

echo "[3/4] Starting backend on http://localhost:8000 ..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
sleep 3

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[4/4] Starting frontend on http://localhost:5173 ..."
cd ../frontend
npm install -q
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=============================="
echo "  AriaSQL running!"
echo "  UI  → http://localhost:5173"
echo "  API → http://localhost:8000"
echo "  Docs→ http://localhost:8000/docs"
echo "=============================="
echo "  Press Ctrl+C to stop."
echo ""

# Wait and clean up on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
