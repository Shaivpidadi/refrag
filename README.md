# REFRAG: Representation-Focused Retrieval Augmented Generation

Open-source implementation of Meta's REFRAG technique for improved RAG systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.01092-b31b1b.svg)](https://arxiv.org/abs/2509.01092)

## 🚀 What is REFRAG?

Traditional RAG systems embed and retrieve raw document chunks. **REFRAG** uses an LLM to generate optimized representations of each chunk before embedding, resulting in:

- **Better retrieval quality**: LLM-powered representations focus on key information
- **Smaller context windows**: Condensed representations = fewer tokens to LLM
- **Improved relevance**: Semantic understanding > pure vector similarity

Based on [Meta AI's REFRAG paper (arXiv:2509.01092)](https://arxiv.org/abs/2509.01092). This is an independent open-source implementation for the community.

## 📊 Benchmarks

**Fair Comparison:** Both methods use `sentence-transformers/all-MiniLM-L6-v2` for embeddings. This isolates the REFRAG technique (LLM-generated representations) rather than just comparing embedding quality.

### Performance Results

| Method | Index Time (5 docs) | Retrieval Time | Top Result Quality |
|--------|---------------------|----------------|-------------------|
| **Vanilla RAG** | 0.405s | 0.095s | Python, Rust, JavaScript |
| **REFRAG (first run)** | 5.263s | **0.023s** ⚡ | Python, Rust, **Machine Learning** ✓ |
| **REFRAG (cached)** | **0.012s** 🚀 | **0.012s** 🚀 | Python, Rust, **Machine Learning** ✓ |

### Key Takeaways

- **First run:** 13x slower indexing (LLM generates representations)
- **Cached runs:** **34x faster** indexing than vanilla (representations reused)
- **Retrieval:** **4-8x faster** than vanilla RAG
- **Quality:** Better semantic matching (finds "Machine Learning" instead of "JavaScript")
- **Cache benefit:** **424x speedup** between first and subsequent indexes

[See full comparison](examples/compare_with_vanilla_rag.py)

...

## 📚 Citation

This implementation is based on the following paper:

```bibtex
@misc{lin2025refragrethinkingragbased,
      title={REFRAG: Rethinking RAG based Decoding}, 
      author={Xiaoqiang Lin and Aritra Ghosh and Bryan Kian Hsiang Low and Anshumali Shrivastava and Vijai Mohan},
      year={2025},
      eprint={2509.01092},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.01092}, 
}
```

If you use this implementation in your research, please cite both the original paper and this repository.

## 🙏 Acknowledgments

Based on [REFRAG research by Meta AI](https://arxiv.org/abs/2509.01092). This is an independent implementation for the open-source community.

**Disclaimer:** This is not an official Meta product. For the official implementation, please refer to Meta's repositories.
