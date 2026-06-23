# RAG Complaint Chatbot for CrediTrust Financial

An intelligent complaint analysis system using Retrieval-Augmented Generation (RAG) to transform customer feedback into actionable insights for financial services.

## Project Overview

CrediTrust Financial processes thousands of customer complaints monthly across:
- Credit Cards
- Personal Loans  
- Savings Accounts
- Money Transfers

This RAG-powered chatbot enables stakeholders to ask natural language questions about complaint data and receive synthesized, evidence-backed answers.

## Project Structure

```
rag-complaint-chatbot/
├── data/
│   ├── raw/              # Raw CFPB complaint dataset
│   └── processed/        # Cleaned and filtered data
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code modules
├── tests/                # Unit tests
├── vector_store/         # Persisted vector database
├── app.py               # Main application interface
└── requirements.txt     # Python dependencies
```

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data**
   - Download CFPB complaint dataset
   - Place as `data/raw/complaints.csv`

3. **Run Task 1 - EDA and Preprocessing**
   ```bash
   cd src
   python task1_eda_preprocessing.py
   ```
   Or use the Jupyter notebook in `notebooks/`

4. **Run Task 2 - Chunking and Embedding Pipeline**
   ```bash
   cd src
   python task2_embedding_pipeline.py
   ```
   Creates vector store with embeddings for semantic search

## Tasks

- [x] **Task 1**: Exploratory Data Analysis and Data Preprocessing
- [x] **Task 2**: Build Chunking and Embedding Pipeline  
- [ ] **Task 3**: Develop RAG Pipeline
- [ ] **Task 4**: Create Interactive UI

## Key Features

- Semantic search over complaint narratives using vector embeddings
- Natural language querying of complaint data
- Multi-product filtering and comparison
- Evidence-based answer generation with source citations

## Data

This project uses the Consumer Financial Protection Bureau (CFPB) complaint dataset containing real customer complaints across financial products.

## License

This project is for educational and internal business use.