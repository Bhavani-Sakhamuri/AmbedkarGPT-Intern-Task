# 🧠 RAG Pipeline with LangChain, ChromaDB, HuggingFace & Ollama (Mistral 7B)

This project demonstrates a fully local Retrieval-Augmented Generation (RAG) pipeline using:

- **LangChain** for orchestration
- **ChromaDB** as the vector store
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
- **Ollama** running **Mistral 7B** as the local LLM
- **No API keys, no accounts, no cloud dependencies**

---

## 🚀 Setup Instructions

### 1. Clone the Repository

git clone https://github.com/Bhavani-Sakhamuri/rag-ollama-mistral.git cd rag-ollama-mistral

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Install and Run Ollama

Install Ollama from https://ollama.com and pull the Mistral model:

ollama pull mistral

Make sure Ollama is running in the background.

### 4. Add Your Document

Place your speech.txt file in the root directory

### 5. Run the App

python main.py

Ask questions about the document interactively!



