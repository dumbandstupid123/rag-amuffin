# RAG Pipeline for PDF Documents

This is a Retrieval-Augmented Generation (RAG) pipeline that can process multiple PDF documents and answer questions about their content.

## Features

- Loads multiple PDF documents
- Splits documents into chunks for efficient processing
- Uses OpenAI embeddings for semantic search
- Generates answers using GPT-4o-mini
- Includes LangSmith tracing for debugging and monitoring

## Setup

1. Clone this repository
2. Install required packages:
   ```bash
   pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph python-dotenv langchain-openai
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Update the `file_paths` list in `rag.py` with your PDF documents
2. Modify the question at the bottom of `rag.py` to ask what you want to know
3. Run the pipeline:
   ```bash
   python rag.py
   ```

## Configuration

### PDF Documents
Update the `file_paths` list in `rag.py` to include your PDF documents:
```python
file_paths = [
    "/path/to/your/document1.pdf",
    "/path/to/your/document2.pdf"
]
```

### Questions
Modify the question at the bottom of `rag.py`:
```python
response = graph.invoke({"question": "Your question here"})
```

## Requirements

- Python 3.8+
- OpenAI API key
- LangSmith API key (optional, for tracing)
- PDF documents to process

## Architecture

The pipeline uses:
- **LangChain** for document processing and LLM integration
- **LangGraph** for workflow orchestration
- **OpenAI Embeddings** for semantic search
- **In-memory Vector Store** for document retrieval
- **GPT-4o-mini** for answer generation
