# 📚 Multi-Source Research Agent with FAISS & Transformers  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)  
![FAISS](https://img.shields.io/badge/FAISS-Indexing-orange)  
![Transformers](https://img.shields.io/badge/Transformers-NLP-green)  
![License](https://img.shields.io/badge/License-MIT-brightgreen)  

An **AI-powered multi-source research assistant** that retrieves, synthesizes, and summarizes knowledge from multiple documents.  
It integrates **FAISS** for fast similarity search, **Sentence Transformers** for embeddings, **HuggingFace Transformers** for summarization, and an **interactive reasoning loop** for clarifying ambiguous queries.  

It can also **export research reports in Markdown and PDF**.  

---

## 🚀 Features  

- 📂 Load `.txt` and `.pdf` research documents  
- 🔍 Semantic retrieval with **FAISS**  
- 🧠 Multi-source summarization & synthesis  
- 🤖 Interactive clarifying questions for low-confidence queries  
- 📝 Export results as **Markdown + PDF**  

---

## 📂 Project Structure  
├── research_docs/ # Folder containing input PDFs/TXT files

├── multi_source_agent.py # Main script

├── README.md # Project documentation (this file)


---

## ⚙️ Quick Setup & Run  

Copy the following into a file named **`setup.sh`**:  

```bash
#!/bin/bash

# ==============================
# Multi-Source Research Agent Setup Script
# ==============================

echo "📚 Setting up Multi-Source Research Agent..."

# Step 1: Create virtual environment
echo "⚙️ Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 2: Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Step 3: Install dependencies
echo "📦 Installing dependencies..."
pip install sentence-transformers faiss-cpu transformers PyPDF2 markdown2 weasyprint torch

# Step 4: Prepare research_docs folder if not exists
if [ ! -d "research_docs" ]; then
  echo "📂 Creating research_docs folder..."
  mkdir research_docs
  echo "ℹ️ Place your PDF/TXT files in the 'research_docs' folder."
fi

# Step 5: Run the agent
echo "🚀 Running Multi-Source Research Agent..."
python multi_source_agent.py

echo "✅ Setup complete! Reports will be generated as .md and .pdf"

