# examples/basic_usage.py
"""
Basic REFRAG usage example
"""

from dotenv import load_dotenv
load_dotenv()

from refrag import REFRAGRetriever

# Sample documents
documents = [
    "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose.",
    "The Pacific Ocean is the largest ocean on Earth, covering about 46% of the water surface."
]

def main():
    # Initialize REFRAG retriever
    print("Initializing REFRAG retriever...")
    retriever = REFRAGRetriever(
        llm_provider="openai",  # or "anthropic"
        llm_model="gpt-4o-mini",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Index documents
    print("\nIndexing documents...")
    retriever.index(documents, show_progress=True)
    
    # Show stats
    print("\nRetriever stats:")
    print(retriever.get_stats())
    
    # Retrieve relevant documents
    query = "Tell me about programming languages"
    print(f"\nQuery: {query}")
    print("\nTop 3 results:")
    
    results = retriever.retrieve(query, top_k=3, return_scores=True)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
        print(f"Original: {result['text']}")
        print(f"Representation: {result['representation']}")

if __name__ == "__main__":
    main()