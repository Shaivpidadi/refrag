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

| Method      | Top Result                         | Retrieval Time | Index Time (5 docs) |
| ----------- | ---------------------------------- | -------------- | ------------------- |
| Vanilla RAG | Python, Rust, **JavaScript**       | 0.168s         | 0.33s               |
| REFRAG      | Python, Rust, **Machine Learning** | **0.029s**     | 7.4s                |

Query: "What programming languages are good for AI development?"

**REFRAG correctly ranked Machine Learning content over JavaScript** - better semantic understanding through LLM representations.

[See full comparison](examples/compare_with_vanilla_rag.py)

...

## 📚 Citation

This implementation is based on the following paper:

```bibtex
@article{refrag2024,
  title={REFRAG: Representation-Focused Retrieval Augmented Generation},
  author={Meta AI Research},
  journal={arXiv preprint arXiv:2509.01092},
  year={2024},
  url={https://arxiv.org/abs/2509.01092}
}
```

If you use this implementation in your research, please cite both the original paper and this repository.

## 🙏 Acknowledgments

Based on [REFRAG research by Meta AI](https://arxiv.org/abs/2509.01092). This is an independent implementation for the open-source community.

**Disclaimer:** This is not an official Meta product. For the official implementation, please refer to Meta's repositories.
