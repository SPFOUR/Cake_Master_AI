# Cake_Master_AI
A modular RAG search engine that allows users to ask questions on tips and recipes relating to cakes.

## System's Procedures & SetUp
The project was done via Python ver. 3.12

- **Ingestion** (`rag/ingest.py`) - the system loads documents, tags them as `.txt` or `.pdf` from data/sample_docs. Pdf documents get extracted via pypdf. The documents are arranged into two categories, doc_type = "recipes" and doc_type = "article". Chunking begins at this stage. 
- **Embedding/Indexing** (`rag/embed_store.py`) - the code utilizes real dense embeddings via sentence-transformers, using the model "all-MiniLM-L6-v2".
- **Generation** (`rag/generate.py`) - consists of the extractive mode (no LLM involved) and llm mode (via Google Gemini 2.5 Flash) to retrieve document chunks.
- **Orchestration** (`app.py`) - Uses Streamlit as frontend and handles Gemini rate limit errors with 3 backoff attempts. It also displays results and sources to the user.

## Project Structure
```
final_project/
├── app.py                  # Streamlit interface.
├── .env                    # Stores Gemini API Key.
├── requirements.txt
├── data/       
|   ├── sample_docs/         # Where 35 documents on cakes (both pdf and txt) are stored.      
|   └── sources.csv          # Where documents' citations & URLs are stored.
└── rag/
    ├── ingest.py            # Loads and checks documents (pdf or txt) before chunking.
    ├── embed_store.py       # Vectorize chunks using sentence-transforms tool.
    └── generate.py          # Turn retrieved chunks into an answer and utilizes Google Gemini LLM.
    └── sources.py           # Load sources from the data/sources.csv folder.
``` 

## Project Architecture
 !![My Screenshot](.png)
              

The project's architecture is a classic modular RAG pipeline with four independent stages.

Data Source (data/sample_docs + sources.csv)
    |
Load and Chunk documents (ingest.py + sources.py for citations)
    |
Embedding documents (embed_store.py)
    |
Generation (extractive mode or using LLM)
    |
Frondend display (app.py using Streamlit)



## Design Decisions

- 35 documents were chosen for this agent.
- Citations and URLs get stored outside of sample_docs for ease of loading via rag/sources.py
- Documents get categorize into recipes in a txt format and articles in a pdf format for ease of chunking as most recipes can be retrieved through html prints. Pdf files are trickier and requires pypdf tool.
- When Gemini reaches its rate limits, the system will initiate a retry procedure for 3 attempts before displaying an error message.
- If the user asked a question that is not cake related, the agent will tell the user that it has no information about the query.
