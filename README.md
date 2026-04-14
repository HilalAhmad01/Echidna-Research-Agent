# 🌌 Echidna AI Research Station

A powerful, privacy-focused, hybrid AI research suite. Echidna combines the speed and privacy of local Large Language Models (via Ollama) with the vast knowledge of cloud APIs (Google Gemini) to create a comprehensive research, writing, and verification environment.

![Dashboard Screenshot](https://github.com/HilalAhmad01/Echidna-Research-Agent/blob/main/gemma%20research.png)

![Dashboard Screenshot](https://github.com/HilalAhmad01/Echidna-Research-Agent/blob/main/gemini_factchecker.png)

![Dashboard Screenshot](https://github.com/HilalAhmad01/Echidna-Research-Agent/blob/main/Paraphraser.png)


---

## 🚀 Features

### 1. 🔍 Deep Research Agent (Local RAG)
* **How it works:** Searches the web using Tavily, chunks and embeds the data locally using `nomic-embed-text` and FAISS, and answers questions using the local `gemma2:2b` model.
* **Benefit:** 100% token-free generation and complete local privacy for your reading, synthesis, and Q&A.

### 2. ⚖️ Fact Checker (Cloud)
* **How it works:** Uses Google's **Gemini 3 Flash** API to cross-reference text, verify claims, and provide an accuracy score.
* **Benefit:** Leverages massive, up-to-date cloud knowledge to catch complex myths, historical inaccuracies, and nuanced claims.

### 3. ✍️ Paraphraser (Local)
* **How it works:** Uses your local `gemma2:2b` model to rewrite text, fix grammar, and detect tone.
* **Benefit:** Lightning-fast, private text refinement without rate limits.

---

## 🛠️ Tech Stack

* **UI / Interface:** [Gradio 6.0+](https://gradio.app/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Local Inference:** [Ollama](https://ollama.com/) (`gemma2:2b`, `nomic-embed-text`)
* **Cloud Inference:** Google GenAI (`gemini-3-flash-preview`)
* **Vector Store:** FAISS
* **Search API:** Tavily

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/HilalAhmad01/Echidna-Research-Agent
cd echidna-ai-research
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
gradio>=5.0.0
python-dotenv
langchain
langchain-ollama
langchain-community
langchain-google-genai
langchain-text-splitters
tavily-python
faiss-cpu
pydantic
```

---

