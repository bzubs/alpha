v0.1.0 - Initialization and Setup of L1 and L2 Semantic Cache

Features:
-->Hierarchial Semantic Memory
-->Dynamic Cluster Formation
-->Custom Cluster eviction logic from L1


Primary Experiment Results:

Experiment Result

{'retrieval_accuracy': 0.8, 'avg_tokens': 11.06}

Brief Overview:

The system organizes conversation history into topic clusters represented by centroid embeddings and summarized text. These clusters are stored across a multi-level memory hierarchy:

L1 — Active Memory: clusters injected directly into the model context

L2 — Warm Memory: recently used clusters not currently in context

Vector Store — Long-term Memory: full semantic archive

When a new prompt arrives, the system evaluates how well the current L1 memory explains the prompt. If the alignment score is weak, relevant clusters are retrieved from L2 or the vector store and promoted into L1 if they improve overall semantic alignment.

This enables LLM systems to maintain a dynamic semantic working set, improving context relevance while controlling token usage.

Detailed Evaluations and Explanation will follow soon
