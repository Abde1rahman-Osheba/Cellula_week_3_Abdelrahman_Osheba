import os
import sys
import numpy as np

# ──────────────────────────────────────────────
# Stage 1: Document Ingestion
# ──────────────────────────────────────────────

def load_document(filepath: str) -> str:
    """Load and validate the knowledge base text file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Knowledge base not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        raise ValueError("Knowledge base file is empty.")

    print(f"[✓] Document loaded: {filepath} ({len(content)} characters)")
    return content


# ──────────────────────────────────────────────
# Stage 2: Text Segmentation (Paragraph-based)
# ──────────────────────────────────────────────

def chunk_text(text: str, min_chunk_length: int = 50) -> list[str]:
    """
    Split text into semantic chunks based on paragraphs (double newlines).
    Filters out chunks that are too short to be meaningful.
    """
    raw_chunks = text.split("\n\n")
    chunks = [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) >= min_chunk_length]

    if not chunks:
        raise ValueError("No valid chunks produced from the document.")

    print(f"[✓] Text segmented into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: {len(chunk)} chars — \"{chunk[:60]}...\"")
    return chunks


# ──────────────────────────────────────────────
# Stage 3: Embedding Generation
# ──────────────────────────────────────────────

def generate_embeddings(chunks: list[str]) -> tuple:
    """
    Generate vector embeddings for each chunk using sentence-transformers.
    Returns the model and the numpy array of embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model_name = "all-MiniLM-L6-v2"
    print(f"[…] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    print(f"[✓] Generated embeddings: shape {embeddings.shape}")
    return model, embeddings


# ──────────────────────────────────────────────
# Stage 4: Vector Storage (FAISS)
# ──────────────────────────────────────────────

def build_vector_store(embeddings: np.ndarray):
    """Build a FAISS index from the embeddings for similarity search."""
    import faiss

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"[✓] FAISS index built: {index.ntotal} vectors, dimension {dimension}")
    return index


# ──────────────────────────────────────────────
# Stage 5: Retrieval Mechanism
# ──────────────────────────────────────────────

def retrieve(query: str, model, index, chunks: list[str], top_k: int = 3) -> list[str]:
    """
    Embed the query and retrieve the Top-K most relevant chunks from FAISS.
    """
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    retrieved = []
    print(f"\n[🔍] Query: \"{query}\"")
    print(f"[🔍] Top-{top_k} retrieved chunks:")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        chunk = chunks[idx]
        retrieved.append(chunk)
        print(f"    {rank+1}. (distance={dist:.4f}) \"{chunk[:80]}...\"")

    return retrieved


# ──────────────────────────────────────────────
# Stage 6: Response Generation (Ollama LLM)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise question-answering assistant. You must answer the user's question using ONLY the provided context below. Follow these rules strictly:

1. Base your answer EXCLUSIVELY on the provided context.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the knowledge base to answer this question."
3. Do NOT make assumptions or add external knowledge.
4. Keep your answer concise, informative, and directly relevant.
5. Do NOT mention that you are using a context or retrieved chunks — just answer naturally."""


def generate_response(query: str, retrieved_chunks: list[str]) -> str:
    """
    Generate a grounded response using Ollama (local LLM).
    The LLM is instructed to use ONLY the retrieved context.
    """
    from langchain_ollama import OllamaLLM

    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    llm = OllamaLLM(model="llama3")

    try:
        response = llm.invoke(
            prompt,
            system=SYSTEM_PROMPT
        )
        return response.strip()
    except Exception as e:
        return f"[Error generating response: {e}]"


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PERSONALIZED RAG PIPELINE")
    print("  TEXT → CHUNKS → EMBEDDINGS → VECTOR DB → RETRIEVAL → LLM")
    print("=" * 60)
    print()

    # Determine knowledge base path (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "knowledge_base.txt")

    # Stage 1: Ingest
    text = load_document(kb_path)

    # Stage 2: Chunk
    chunks = chunk_text(text)

    # Stage 3: Embed
    model, embeddings = generate_embeddings(chunks)

    # Stage 4: Store
    index = build_vector_store(embeddings)

    # Pipeline ready
    print()
    print("=" * 60)
    print("  Pipeline ready! Ask questions about the knowledge base.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    # Interactive query loop
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Stage 5: Retrieve
        retrieved = retrieve(query, model, index, chunks, top_k=3)

        # Stage 6: Generate
        print("\n[⏳] Generating response...\n")
        answer = generate_response(query, retrieved)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
