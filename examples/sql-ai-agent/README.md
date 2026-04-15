# SQL AI Agent

Full-stack application allowing you to map natural language queries to SQL execution using a dynamic LLM agent with MLflow observability.

## Prerequisites
- **Node.js**: v18+ (for frontend Vite server)
- **Python**: 3.9+ (for backend FastAPI server)
- **Azure OpenAI**: Endpoint details & API key

## Setup Instructions

### 1. Environment Configuration
Create a `.env` file in the root directory by copying the template.
```bash
cp .env.example .env
```
Open `.env` and configure your Azure OpenAI settings and Database URL.

### 2. Backend Setup
Navigate to the `backend` folder, create a Python virtual environment, install dependencies, and start the app:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
The FastAPI server will be available at `http://localhost:8000`. You can test its health endpoint at `http://localhost:8000/health`.

### 3. Frontend Setup
In a new terminal window, navigate to the `frontend` folder, install dependencies using npm, and start the dev server:

```bash
cd frontend
npm install
npm run dev
```
The React frontend will be available at `http://localhost:5173`.


cd /Users/pradip/Desktop/Learning/Claude/Infogain/SQL-AI-Agent-Pradip-Tivhale/backend
mlflow server --host 0.0.0.0 --port 5000
