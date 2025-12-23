# examples/compare_with_vanilla_rag.py
"""
Compare REFRAG vs Vanilla RAG performance

Both implementations use the SAME embedding model (all-MiniLM-L6-v2).
This isolates the REFRAG technique: adding LLM-generated representations.

Also demonstrates REFRAG's caching: representations are generated once
and reused on subsequent indexing operations.
"""

from refrag import REFRAGRetriever
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from dotenv import load_dotenv 
import os

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Sample documents
documents = [
    "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "JavaScript is primarily used for web development and runs in browsers. It was created by Brendan Eich in 1995.",
    "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
    "Rust is a systems programming language focused on safety and performance, created by Mozilla Research.",
]

class VanillaRAG:
    """
    Vanilla RAG implementation using SentenceTransformer.
    Uses the SAME embedding model as REFRAG for comparison.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = None
        self.embeddings = None
    
    def index(self, documents):
        """Index documents by embedding them directly (no LLM representations)."""
        self.documents = documents
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)
    
    def retrieve(self, query, top_k=3):
        """Retrieve top-k most similar documents."""
        query_emb = self.model.encode(query, normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [{"text": self.documents[idx], "score": float(similarities[idx])} 
                for idx in top_indices]

def main():
    query = "What programming languages are good for AI development?"
    
    print("\n" + "=" * 80)
    print("REFRAG vs VANILLA RAG - BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (same for both)")
    print(f"LLM Provider: OpenAI GPT-4o-mini (REFRAG only)")
    print(f"Documents: {len(documents)}")
    print(f"Query: {query}")
    print("=" * 80)
    
    # Run benchmarks
    print("\n🔄 Running benchmarks...\n")
    
    # Vanilla RAG
    print("[1/3] Vanilla RAG...")
    vanilla = VanillaRAG(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    start = time.time()
    vanilla.index(documents)
    index_time_vanilla = time.time() - start
    
    start = time.time()
    vanilla_results = vanilla.retrieve(query, top_k=3)
    retrieve_time_vanilla = time.time() - start
    
    # REFRAG first run
    print("[2/3] REFRAG (first run - generating representations)...")
    refrag = REFRAGRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    start = time.time()
    refrag.index(documents, show_progress=False)
    index_time_refrag = time.time() - start
    
    start = time.time()
    refrag_results = refrag.retrieve(query, top_k=3, return_scores=True)
    retrieve_time_refrag = time.time() - start
    
    # REFRAG cached run
    print("[3/3] REFRAG (cached run - reusing representations)...")
    start = time.time()
    refrag.index(documents, show_progress=False)
    index_time_refrag_cached = time.time() - start
    
    start = time.time()
    refrag_cached_results = refrag.retrieve(query, top_k=3, return_scores=True)
    retrieve_time_refrag_cached = time.time() - start
    
    # Calculate metrics
    speedup = index_time_refrag / index_time_refrag_cached
    first_run_overhead = index_time_refrag / index_time_vanilla
    cached_vs_vanilla = index_time_refrag_cached / index_time_vanilla
    
    # Display results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Indexing performance table with visual bars
    print("\n📊 INDEXING PERFORMANCE")
    print("-" * 80)
    print(f"{'Method':<30} {'Time (s)':<15} {'vs Vanilla':<15} {'Visual':<25}")
    print("-" * 80)
    
    # Visual bar for vanilla (baseline)
    vanilla_bar = "█" * 20
    print(f"{'Vanilla RAG':<30} {index_time_vanilla:<15.3f} {'1.00x':<15} {vanilla_bar:<25}")
    
    # Visual bar for REFRAG first run
    first_run_bar_length = min(int(first_run_overhead * 20), 80)
    first_run_bar = "█" * first_run_bar_length
    print(f"{'REFRAG (first run)':<30} {index_time_refrag:<15.3f} {f'{first_run_overhead:.2f}x':<15} {first_run_bar:<25}")
    
    # Visual bar for REFRAG cached (much shorter)
    cached_bar_length = max(1, int(cached_vs_vanilla * 20))
    cached_bar = "█" * cached_bar_length
    print(f"{'REFRAG (cached)':<30} {index_time_refrag_cached:<15.3f} {f'{cached_vs_vanilla:.2f}x':<15} {cached_bar:<25} ⚡")
    print("-" * 80)
    
    # Speedup visualization
    print(f"\n💡 CACHING IMPACT:")
    print(f"   First run:  {index_time_refrag:.3f}s  {'█' * 40}")
    print(f"   Cached:     {index_time_refrag_cached:.3f}s  █")
    print(f"   Speedup:    {speedup:.1f}x FASTER! 🚀")
    
    # Retrieval performance table with visual bars
    print("\n🔍 RETRIEVAL PERFORMANCE")
    print("-" * 80)
    print(f"{'Method':<30} {'Time (s)':<15} {'vs Vanilla':<15} {'Visual':<25}")
    print("-" * 80)
    
    # Retrieval bars
    retrieval_vanilla_bar = "█" * 20
    print(f"{'Vanilla RAG':<30} {retrieve_time_vanilla:<15.3f} {'1.00x':<15} {retrieval_vanilla_bar:<25}")
    
    retrieval_refrag_ratio = retrieve_time_refrag / retrieve_time_vanilla
    retrieval_refrag_bar_length = max(1, int(retrieval_refrag_ratio * 20))
    retrieval_refrag_bar = "█" * retrieval_refrag_bar_length
    print(f"{'REFRAG (first run)':<30} {retrieve_time_refrag:<15.3f} {f'{retrieval_refrag_ratio:.2f}x':<15} {retrieval_refrag_bar:<25} ⚡")
    
    retrieval_cached_ratio = retrieve_time_refrag_cached / retrieve_time_vanilla
    retrieval_cached_bar_length = max(1, int(retrieval_cached_ratio * 20))
    retrieval_cached_bar = "█" * retrieval_cached_bar_length
    print(f"{'REFRAG (cached)':<30} {retrieve_time_refrag_cached:<15.3f} {f'{retrieval_cached_ratio:.2f}x':<15} {retrieval_cached_bar:<25} 🚀")
    print("-" * 80)
    
    # Quality comparison
    print("\n📈 RETRIEVAL QUALITY (Top 3 Results)")
    print("-" * 80)
    
    print("\nVanilla RAG Results:")
    for i, result in enumerate(vanilla_results[:3], 1):
        print(f"  {i}. Score: {result['score']:.4f}")
        print(f"     {result['text'][:80]}...")
    
    print("\nREFRAG Results (with representations):")
    for i, result in enumerate(refrag_results[:3], 1):
        print(f"  {i}. Score: {result['score']:.4f}")
        print(f"     Original: {result['text'][:60]}...")
        print(f"     Rep: {result['representation'][:60]}...")
    
    # Speedup summary table
    print("\n⚡ SPEEDUP SUMMARY")
    print("-" * 80)
    print(f"{'Metric':<40} {'Improvement':<20} {'Impact':<20}")
    print("-" * 80)
    
    # Indexing speedups
    if cached_vs_vanilla < 1:
        indexing_improvement = f"{1/cached_vs_vanilla:.1f}x FASTER"
        indexing_impact = "🚀 Huge win"
    else:
        indexing_improvement = f"{cached_vs_vanilla:.2f}x slower"
        indexing_impact = "⚠️ Trade-off"
    print(f"{'REFRAG Cached vs Vanilla (indexing)':<40} {indexing_improvement:<20} {indexing_impact:<20}")
    
    # Retrieval speedups
    if retrieval_refrag_ratio < 1:
        retrieval_improvement = f"{1/retrieval_refrag_ratio:.1f}x FASTER"
        retrieval_impact = "⚡ Faster"
    else:
        retrieval_improvement = f"{retrieval_refrag_ratio:.2f}x slower"
        retrieval_impact = "≈ Similar"
    print(f"{'REFRAG vs Vanilla (retrieval)':<40} {retrieval_improvement:<20} {retrieval_impact:<20}")
    
    # Cache benefit
    print(f"{'REFRAG: First run → Cached':<40} {f'{speedup:.1f}x FASTER':<20} {'🔥 Caching':<20}")

if __name__ == "__main__":
    main()