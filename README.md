# Video Search System 🎬

![Demo Screenshot](./demo.png)
An AI-powered video keyframe search system that leverages advanced machine learning models to search across video datasets using text queries. It utilizes hybrid search retrieval combining vector embeddings and reranking mechanisms to provide highly relevant results.

## 🚀 Key Features

*   **Text-to-Image/Video Search:** Search through thousands of extracted video frames using natural language text queries.
*   **Vector Search & Reranking:** Powered by Milvus Vector Database for ultra-fast similarity search and a HuggingFace Reranker (BGE) for improved accuracy and relevance.
*   **Modern Interactive UI:** A high-performance, dark-themed responsive frontend built with React, Vite, and TailwindCSS.
*   **GPU Accelerated:** Leverages NVIDIA GPUs for both backend embedding generation, reranking, and Milvus vector computations.

## 🛠️ Tech Stack

### Web & Architecture
*   **Frontend:** React 18, Vite, TailwindCSS, Axios
*   **Backend:** FastAPI, Python, Uvicorn (ASGI)
*   **Infrastructure:** Docker Compose

### AI / Machine Learning
*   **Vector Database:** Milvus (GPU Enabled)
*   **Embeddings:** `open_clip_torch`
*   **Reranker:** `FlagEmbedding` (BAAI/bge-reranker)
*   **Deep Learning Framework:** PyTorch, Transformers, Peft

### Data Storage & Operations
*   **Object Storage:** MinIO (S3 compatible)
*   **Key-Value Store:** Etcd
*   **Image Processing:** OpenCV, Pillow

## 📂 Project Structure

```text
video_search_project/
├── backend/            # FastAPI python server, AI models and database interactions
├── frontend/           # React + Vite application
├── data/               # Local data storage for datasets and media
├── extract/            # Extraction storage locations/scripts
├── scripts/            # Utility scripts (e.g., migrations, processing)
├── docker-compose.yml  # Docker environment configurations for Milvus & services
└── README.md           # Project documentation
```

## ⚙️ Getting Started

### Prerequisites
*   Node.js & npm (for frontend)
*   Python 3.10+ (for backend)
*   Docker & Docker Compose (for infrastructure)
*   NVIDIA GPU with CUDA installed (highly recommended)

### 1. Start Infrastructure (Milvus Vector DB)
You can start up the required database and dependent services (MinIO, Etcd) using Docker Compose:
```bash
docker-compose up -d milvus-standalone etcd minio attu
```
*(Note: You can access the Attu dashboard at `http://localhost:8000` to inspect Milvus vectors)*

### 2. Backend Setup
Navigate to the `backend` directory and install the required dependencies:
```bash
cd backend
python -m venv .venv
# Activate your venv:
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```
Run the FastAPI backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Frontend Setup
Open a new terminal, **you MUST navigate to the `frontend` directory first**, then install packages and start the development server:
```bash
cd frontend
npm install
npm run dev
```
The React frontend should now be running (usually on `http://localhost:5173` or `http://localhost:3000`).

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
