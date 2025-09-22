import os
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from transformers import pipeline
import markdown2
from weasyprint import HTML

# -----------------------------
# 1. Load documents
# -----------------------------
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        text = ""
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + " "
        if text:
            documents.append((filename, text))
    return documents

# -----------------------------
# 2. Chunking
# -----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# -----------------------------
# 3. Generate embeddings
# -----------------------------
def generate_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    all_chunks = []
    metadata = []
    for doc_name, text in documents:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([doc_name]*len(chunks))
    embeddings = model.encode(all_chunks)
    return model, embeddings, all_chunks, metadata

# -----------------------------
# 4. Build FAISS index
# -----------------------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# -----------------------------
# 5. Retrieve with confidence
# -----------------------------
def retrieve(query, model, index, all_chunks, metadata, top_k=5):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    retrieved_docs = [metadata[i] for i in indices[0]]
    confidence = 1 - (distances[0].mean() / 4.0)
    return retrieved_chunks, retrieved_docs, confidence

# -----------------------------
# 6. Summarization & Multi-Source Synthesis
# -----------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def multi_source_summarize(chunks, chunk_doc_map):
    combined_text = ""
    docs_grouped = {}
    
    for chunk, doc in zip(chunks, chunk_doc_map):
        if doc not in docs_grouped:
            docs_grouped[doc] = []
        docs_grouped[doc].append(chunk)
    
    doc_summaries = []
    for doc, chunks_list in docs_grouped.items():
        text = " ".join(chunks_list)[:4000]  # HuggingFace token limit
        summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        doc_summaries.append(f"{doc}: {summary}")
    
    combined_text = " ".join(doc_summaries)[:4000]
    final_summary = summarizer(combined_text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
    return final_summary, doc_summaries

# -----------------------------
# 7. Generate clarifying question
# -----------------------------
def clarifying_question(query, retrieved_docs):
    return (f"The query '{query}' may be too broad. "
            f"Could you specify which aspect of the following documents you want more details on: "
            f"{', '.join(list(set(retrieved_docs))) }?")

# -----------------------------
# 8. Interactive multi-source reasoning (Python 3.13 compatible)
# -----------------------------
def interactive_multi_source_agent(query, model, index, all_chunks, metadata, top_k=5, confidence_threshold=0.7):
    final_summary = ""
    history = []
    current_query = query
    
    while True:
        retrieved_chunks, retrieved_docs, confidence = retrieve(current_query, model, index, all_chunks, metadata, top_k)
        final_summary, doc_summaries = multi_source_summarize(retrieved_chunks, retrieved_docs)
        
        history.append({
            "query": current_query,
            "retrieved_docs": list(set(retrieved_docs)),
            "retrieved_chunks_preview": [chunk[:200]+"..." for chunk in retrieved_chunks],
            "doc_summaries": doc_summaries,
            "final_summary": final_summary,
            "confidence": confidence
        })
        
        print(f"\nQuery: {current_query}")
        print(f"Retrieved Docs: {list(set(retrieved_docs))}")
        print(f"Confidence: {confidence:.2f}")
        
        if confidence >= confidence_threshold:
            break
        else:
            # Clarifying question for user
            question = clarifying_question(current_query, retrieved_docs)
            print("\nClarifying Question:", question)
            user_input = input("Refine your query (or press Enter to accept): ").strip()
            if not user_input:
                break
            current_query = user_input
    
    return final_summary, history

# -----------------------------
# 9. Export Report
# -----------------------------
def export_report(history, final_summary, filename="multi_source_research_report"):
    md_content = f"# Multi-Source Interactive Research Report\n\n"
    for step in history:
        md_content += f"## Query Step: {step['query']}\n"
        md_content += f"**Retrieved Docs:** {', '.join(step['retrieved_docs'])}\n"
        md_content += f"**Confidence:** {step['confidence']:.2f}\n"
        md_content += f"**Chunk Previews:**\n"
        for i, chunk in enumerate(step['retrieved_chunks_preview']):
            md_content += f"{i+1}. {chunk}\n"
        md_content += f"**Document Summaries:**\n"
        for ds in step['doc_summaries']:
            md_content += f"- {ds}\n"
        md_content += f"**Step Summary:** {step['final_summary']}\n\n"
    md_content += f"# Final Synthesized Summary\n{final_summary}\n"
    
    with open(f"{filename}.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    
    HTML(string=markdown2.markdown(md_content)).write_pdf(f"{filename}.pdf")
    print(f"Report exported as {filename}.md and {filename}.pdf")

# -----------------------------
# 10. Run the agent
# -----------------------------
if __name__ == "__main__":
    documents = load_documents("research_docs")
    model, embeddings, all_chunks, metadata = generate_embeddings(documents)
    index = build_index(embeddings)
    
    query = "Applications of FAISS in AI research"
    final_summary, history = interactive_multi_source_agent(query, model, index, all_chunks, metadata)
    
    export_report(history, final_summary, "multi_source_interactive_report")
