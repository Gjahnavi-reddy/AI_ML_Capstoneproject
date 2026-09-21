# Zepto Support Assistant

## Overview

This module implements an offline GenAI-style customer support assistant for Zepto policies.

The service uses document ingestion, sentence-transformer embeddings, ChromaDB vector retrieval, LangGraph orchestration, Pydantic structured output, and a FastAPI API.

The required graded baseline runs fully offline using `MOCK_LLM=1`.

## Architecture

```text
Zepto Policy Documents
        ↓
Document Ingestion
(app/ingestion.py)
        ↓
all-MiniLM-L6-v2 Embeddings
        ↓
ChromaDB
(zepto_policies collection)
        ↓
User Query
        ↓
LangGraph StateGraph
        ↓
classify_intent
     ↙       ↘
policy       general
 ↓             ↓
retrieve_      direct_
and_answer     answer
 ↓             ↓
Structured Answer
(answer, sources, confidence)
        ↓
FastAPI POST /ask

## Components

### 1. Document Corpus

The `documents/` folder contains the eight required Zepto policy documents:

- Delivery Policy
- Returns & Refunds
- Membership Tiers
- Order Tracking
- Order Cancellation Policy
- Damaged or Missing Items
- Gift Cards
- Customer Support Hours

### 2. Embeddings and Vector Store

`app/ingestion.py`:

- Loads all eight documents.
- Uses `all-MiniLM-L6-v2` from Sentence Transformers.
- Generates normalized embeddings.
- Stores them in ChromaDB.
- Uses the `zepto_policies` collection.
- Persists the database in `chroma_db/`.

### 3. LangGraph

`app/graph.py` contains a `StateGraph` with three nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

Policy queries are routed to retrieval, while general queries receive the fixed mock response.

Policy queries retrieve the top three most similar chunks from ChromaDB using cosine similarity.

### 4. Structured Output

Responses use the Pydantic model:

```text
answer: str
sources: list[str]
confidence: float

### 5. MOCK_LLM

The application defaults to:

```text
MOCK_LLM=1

### 6. FastAPI

`app/main.py` provides:

```text
POST /ask

### request example
{
  "query": "What is the delivery fee for orders below INR 149?"
}

### response example
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01", "doc_05", "doc_03"],
  "confidence": 1.0
}

### general question
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

### 7. Prompt Template

The structured prompt in `app/graph.py` contains:

- Role
- Context
- Task
- Format
- Length
- Negative constraint
- Few-shot example

The negative constraint explicitly prevents using information that is not present in the retrieved context.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt

python app/ingestion.py

python -m uvicorn app.main:app --reload --port 8000

## Docker

Build the Docker image:

```bash
docker build -t zepto-support-assistant .

docker run -p 7860:7860 zepto-support-assistant

http://127.0.0.1:7860


## Project Structure

```text
support_assistant/
├── documents/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── app/
│   ├── ingestion.py
│   ├── graph.py
│   └── main.py
├── chroma_db/
├── Dockerfile
├── requirements.txt
└── README.md

## Data Flow

1. The eight Zepto policy documents are loaded from the `documents/` folder.
2. `app/ingestion.py` reads and chunks the documents.
3. Each chunk is converted into an embedding using `all-MiniLM-L6-v2`.
4. The embeddings and document information are stored in the ChromaDB `zepto_policies` collection.
5. A user sends a query to the FastAPI `/ask` endpoint.
6. `classify_intent` in `app/graph.py` determines whether the query is a policy question or a general question.
7. Policy questions are sent to `retrieve_and_answer`, which retrieves the top 3 similar chunks from ChromaDB.
8. In mock mode, the answer is generated from the most similar retrieved chunk.
9. General questions are handled by `direct_answer` and receive the fixed mock response.
10. The final answer is validated using the Pydantic response schema containing `answer`, `sources`, and `confidence`.
11. FastAPI returns the validated JSON response to the user.

### MOCK_LLM Behavior

The application uses `MOCK_LLM=1` by default.

In mock mode:

- Intent classification uses a deterministic keyword heuristic.
- Retrieval uses the real Sentence Transformer embeddings and ChromaDB.
- Policy answers are generated from the retrieved context.
- General questions receive a fixed response.
- No LLM API call is made.

When `MOCK_LLM=0` is explicitly enabled, the optional real-LLM path can be used for the generation steps. Retrieval still uses the local embeddings and ChromaDB.

## Verification

The Support Assistant was tested with `MOCK_LLM` left at its default state.

### Policy Question

Example query:

```json
{
  "query": "How fast does Zepto deliver?"
}

## Prompt Template

The structured prompt used by the optional real-LLM path follows the required role-context-task-format-length structure.

It contains the following components:

- **Role:** Defines the assistant as a Zepto policy support assistant.
- **Context:** Provides the retrieved Zepto policy chunks.
- **Task:** Instructs the assistant to answer the user's question using only the provided context.
- **Format:** Requires the response to follow the structured answer format.
- **Length:** Instructs the assistant to provide a concise answer.
- **Negative constraint:** The assistant must not use information that is not present in the provided context.
- **Few-shot example:** The prompt includes an example showing how a policy question should be answered using retrieved context.

The mock mode does not call an LLM. Therefore, the prompt is used only by the optional `MOCK_LLM=0` real-LLM path.

## Module Summary

This module implements a complete offline GenAI-style Support Assistant for Zepto.

The required graded pipeline includes:

- 8 Zepto policy documents
- Local document ingestion and chunking
- `all-MiniLM-L6-v2` embeddings
- ChromaDB vector storage and retrieval
- LangGraph `StateGraph`
- Intent classification and conditional routing
- Mock LLM mode using `MOCK_LLM=1`
- Structured Pydantic JSON output
- FastAPI `/ask` endpoint
- Local Docker configuration



