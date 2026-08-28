# ResearchInsight AI

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A RAG-powered research assistant that lets you upload PDF research papers and ask questions about them using natural language.

## Live Demo

**[researchinsight-ai-tbpk.onrender.com](https://researchinsight-ai-tbpk.onrender.com/)**

> Note: The app is hosted on Render's free tier — it may take ~60 seconds to load after a period of inactivity.

## Screenshots

![ResearchInsight AI — Chat Interface](screenshots/Screenshot%20from%202026-08-28%2011-10-42.png)

## Features

- Upload multiple PDF research papers
- Extracts and chunks text from PDFs
- Generates semantic embeddings using `all-MiniLM-L6-v2`
- Stores and searches vectors with FAISS
- Answers questions using Groq LLM with source citations
- Live step-by-step progress during PDF processing
- Session analytics (queries, response times, pages indexed)

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS |
| LLM | Groq API |
| Deployment | Render |

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

3. Create a `.env` file and add your Groq API key:
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
   Get a free key at [console.groq.com](https://console.groq.com)

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service** → connect your repo
3. Fill in:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Under **Environment**, add:
   ```
   GROQ_API_KEY = your_groq_api_key_here
   ```
5. Click **Deploy**

## License

This project is licensed under the MIT License.
