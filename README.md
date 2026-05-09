# Multi-City RAG Travel Assistant

Final Project for CSC 7644: Applied LLM Development

## Project Overview

The Multi-City Travel Assistant is a Retrieval-Augmented Generation (RAG) application that provides grounded travel recommendations for specific cities using a hybrid retrieval pipeline. The system combines BM25 lexical retrieval and vector similarity retrieval to improve contextual relevance and reduce hallucinations in generated responses.

The application was designed to address the issue of generic LLMs generating outdated or unsupported travel recommendations when relying only on pretrained knowledge. Instead of generating responses from general model memory, the system retrieves relevant travel passages from a curated document corpus and uses them to generate grounded travel responses.

The current implementation supports the following cities:

- New Orleans
- Atlanta
- New York City
- Houston
- Los Angeles

Users can interact with the system through either:

- A command-line interface
- A Streamlit web interface

The project also includes a custom evaluation pipeline for testing retrieval quality, city filtering accuracy, source diversity, and response latency.

---

# Key Features

- Hybrid retrieval using BM25 and vector similarity search
- ChromaDB vector storage for embedding retrieval
- Metadata-based city filtering to reduce hallucinations
- Grounded response generation using GPT-4o-mini
- Streamlit conversational web interface
- Command-line interaction support
- Automatic document chunking and embedding ingestion
- Retrieval evaluation using Precision@K and latency metrics
- Duplicate chunk removal before generation
- Structured itinerary generation through prompt engineering
- Unsupported city handling to prevent fabricated recommendations

---

# Tech Stack and Architecture

## Core Technologies

- Python
- OpenAI API
- ChromaDB
- rank-bm25
- Streamlit
- python-dotenv

## LLM and Embedding Models

### Chat Model

- GPT-4o-mini

### Embedding Model

- text-embedding-3-small

## Main Components

### `travel_rag.py`

Main backend RAG pipeline implementation.

Responsibilities:

- Document loading
- Text chunking
- Embedding generation
- ChromaDB operations
- BM25 retrieval
- Vector retrieval
- Hybrid fusion
- Prompt generation
- Grounded answer generation

### `app.py`

Streamlit frontend interface.

Responsibilities:

- User interaction
- Query submission
- Response display

### `evaluate_rag.py`

Evaluation and benchmarking script.

Responsibilities:

- Precision@K evaluation
- City detection testing
- Retrieved chunk city accuracy testing
- Source diversity testing
- Retrieval latency testing
- Response latency testing
- CSV export of evaluation results

### `data/corpus/`

Contains the travel documents used for ingestion.

---

# Setup Instructions

## Prerequisites

The project was developed using:

- Python 3.12
- Windows 11

The project should also work on macOS or Linux with Python 3.12 installed.

---

## Clone the Repository

```bash
git clone https://github.com/sescob4/csc7644-final-project-Escobar-Mesa.git
cd csc7644-final-project-Escobar-Mesa
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the root project directory.

Example:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The OpenAI API key is required for:

- Embedding generation
- GPT-4o-mini response generation

Do not upload your `.env` file to GitHub.

---

# Running the Application

## Step 1: Ingest Documents

Before querying the system, the travel documents must be chunked, embedded, and stored inside ChromaDB.

Run:

```bash
python travel_rag.py ingest --data_dir ./data/corpus --db_path ./kb --collection travel --embed_model text-embedding-3-small --size 800 --stride 300
```

This command:

- Loads all `.txt` travel documents
- Splits them into overlapping chunks
- Generates embeddings using OpenAI
- Stores embeddings and metadata inside ChromaDB

---

## Step 2: Run the Streamlit Frontend

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

Example:

```text
http://localhost:8501
```

Users can then ask travel-related questions through the web interface.

Example queries:

- What are good nightlife options in New Orleans?
- Plan a weekend trip to New York City.
- What are family friendly activities in Atlanta?
- What outdoor things can I do in Los Angeles?

---

# Command-Line Usage

The project can also be used directly from the terminal.

## Example Query

```bash
python travel_rag.py answer --query "What are good restaurants in Houston?"
```

Optional parameters:

```bash
--top_k 8
```

---

# Running Evaluation

To evaluate retrieval performance and response quality:

```bash
python evaluate_rag.py
```

The evaluation pipeline measures:

- Precision@K
- Query city detection accuracy
- Retrieved chunk city accuracy
- Source diversity
- Retrieval latency
- Response generation latency

Evaluation results are exported to:

```text
rag_evaluation_results.csv
```

---

# Repository Organization

```text
project-root/
│
├── app.py
├── travel_rag.py
├── evaluate_rag.py
├── rag_evaluation_results.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── corpus/
│       └── *.txt travel documents

```

## File Descriptions

### `app.py`

Streamlit frontend interface for user interaction.

### `travel_rag.py`

Core backend RAG pipeline implementation.

### `evaluate_rag.py`

Evaluation and benchmarking script.

### `rag_evaluation_results.csv`

Saved evaluation output containing retrieval metrics and test query results.

### `requirements.txt`

Project dependency list.

### `.gitignore`

Specifies files and directories excluded from GitHub repository

### `data/corpus/`

Travel documents used for ingestion.

---

# High-Level Workflow

1. User submits a travel query.
2. The system detects whether the query references a supported city.
3. Documents are filtered using city metadata.
4. BM25 retrieval and vector retrieval are performed.
5. Hybrid score fusion combines retrieval scores.
6. Top retrieved chunks are formatted into context.
7. GPT-4o-mini generates a grounded response using only retrieved passages.
8. The response is returned to the user.

---

# Known Limitations

- Only five cities are currently supported.
- The travel corpus is static and may become outdated over time.
- Precision@K evaluation is keyword-based and may miss semantic relevance.
- Broader travel coverage requires a larger document corpus.

---

# Future Improvements

- Expand support to additional cities
- Add international travel support
- Integrate live travel APIs or web retrieval
- Add multilingual support
- Add conversational memory and itinerary refinement

---

# Attributions and Citations

The main backend pipeline in `travel_rag.py` was adapted from the RAG assignment template provided in Module 4 of CSC 7644: Applied LLM Development.

External libraries and frameworks used:

- OpenAI API
- ChromaDB
- rank-bm25
- Streamlit
- python-dotenv
