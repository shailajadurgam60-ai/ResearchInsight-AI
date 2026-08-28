# ResearchInsight AI

A RAG-powered research assistant that lets you upload PDF research papers and ask questions about them using natural language.

## Features

- Upload multiple PDF research papers
- Extracts and chunks text from PDFs
- Generates semantic embeddings using `all-MiniLM-L6-v2`
- Stores and searches vectors with FAISS
- Answers questions using Groq LLM with source citations

## Tech Stack

- **Frontend**: Streamlit
- **PDF Parsing**: PyMuPDF
- **Embeddings**: Sentence Transformers
- **Vector Search**: FAISS
- **LLM**: Groq API

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ResearchInsight-AI.git
   cd ResearchInsight-AI
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Add your Groq API key to the `.env` file. Get one free at https://console.groq.com

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud

1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud and connect your GitHub repo
3. In the app settings, add your secret:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy
