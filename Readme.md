Here's comprehensive documentation in Markdown format following industry standards:

```markdown
# RAG Chat Agent with Local LLM

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama Required](https://img.shields.io/badge/Ollama-Required-important)](https://ollama.ai/)

A privacy-focused chat agent that combines document analysis with local AI capabilities using Ollama models.

## Features

- 📁 Document processing (PDF/TXT)
- 🧠 Local LLM integration (Ollama)
- 🔍 Hybrid question answering (Documents + General knowledge)
- 🖥️ Streamlit web interface
- 🛡️ 100% local execution (No data leaves your machine)

## Installation

### Prerequisites

1. Install [Python 3.10+](https://www.python.org/downloads/)
2. Install [Ollama](https://ollama.ai/download)
3. Download required models:
```bash
ollama pull nomic-embed-text
ollama pull deepseek-llm
```

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/rag-chat-agent.git
cd rag-chat-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create directory structure
mkdir -p data/pdf data/txt uploads/pdf uploads/txt
```

## Configuration

1. **File Storage**:
   - Place initial documents in:
     - `data/pdf/` for PDF files
     - `data/txt/` for text files
   - Uploaded files will be stored in `uploads/` directory

2. **Environment Setup**:
   ```bash
   # Optional: Set Ollama API endpoint if using remote instance
   # export OLLAMA_HOST=http://your-ollama-server:11434
   ```

## Usage

### Start Application
```bash
streamlit run app/gui.py
```

### Interface Guide
1. **Upload Documents**:
   - Use sidebar uploader for PDF/TXT files
   - Files are automatically processed

2. **Ask Questions**:
   - Type questions in main chat input
   - System prioritizes document content when relevant

3. **Controls**:
   - 🔄 Refresh Knowledge Base: Reprocess documents
   - 🗑️ Clear Chat: Start new conversation

![Interface Demo](assets/interface-demo.png) <!-- Add actual screenshot -->

## Deployment

### Local Deployment
```bash
# Run with production settings
streamlit run app/gui.py --server.headless true
```

### Docker Deployment
1. Build image:
   ```dockerfile
   # Dockerfile
   FROM python:3.10-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["streamlit", "run", "app/gui.py"]
   ```
2. Run container:
   ```bash
   docker build -t rag-chat .
   docker run -p 8501:8501 rag-chat
   ```

### Cloud Deployment (AWS/Heroku)
1. Add `requirements.txt` with:
   ```
   streamlit
   langchain-community
   langchain-ollama
   chromadb
   unstructured
   ```
2. Set startup command:
   ```bash
   streamlit run app/gui.py --server.port $PORT
   ```

## Customization

### Modify Settings
1. **Chunk Size** (`data_loader.py`):
   ```python
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,  # Adjust for document size
       chunk_overlap=200
   )
   ```

2. **LLM Model** (`rag_pipeline.py`):
   ```python
   llm = ChatOllama(
       model="deepseek-r1:1.5b",  # Change to preferred model
       temperature=0.5
   )
   ```

3. **UI Settings** (`gui.py`):
   ```python
   st.set_page_config(
       page_title="Custom Title",
       layout="wide"
   )
   ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Run `ollama pull model-name` |
| Document processing fails | Check file permissions in `uploads/` |
| Low quality answers | Adjust chunk size in `data_loader.py` |
| Memory issues | Use smaller LLM model |

## Contributing

1. Fork the repository
2. Create feature branch:
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit changes
4. Push to branch
5. Open pull request

## License

MIT License - See [LICENSE](LICENSE) for details

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM management
- [LangChain](https://www.langchain.com/) for RAG pipeline
- [ChromaDB](https://www.trychroma.com/) for vector storage
```

This documentation follows industry standards with:

1. Clear installation instructions
2. Configuration guidance
3. Usage examples
4. Deployment options
5. Customization paths
6. Troubleshooting section
7. Contribution guidelines

To use this documentation:
1. Save as `README.md` in project root
2. Create `requirements.txt` with:
```
streamlit
langchain-community
langchain-ollama
chromadb
unstructured
python-magic-bin
pypdf
```
3. Add license file
4. Create `assets/` folder for screenshots

Would you like me to explain any specific section in more detail or add additional sections?