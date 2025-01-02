# RAG for Search: HELPFUL
A work in progress for plain-text based search method using Retrieval-Augmented Generation 
(RAG) that aims to preserve semantic relationships within structured data entries. The focus 
so far is to maintain data coherence and enhance retrieval accuracy through use-case specific 
splitting and processing, and the aim now is to enhance the RAG approaches through Agentic and Knowledge Graph enhanced strategies. 

The current implementation achieves the following: 
- **Semantic Document Splitting**: Preserve meaningful relationships in structured data. 
- **Coherent Information Retrieva**: Related information is kept together during processing, 
for improved context. Document metatada is also utilizeed to further improve context. 
- **Open Source Stack**: The current implementation is built entirely with open-source 
components, making it more easily replicable and deployable. 

This branch is dedicated to develop the following features:
- **Agentic RAG through CrewAI**: Specialized agents for different aspects of the RAG pipeline.

    - Query Analysis and refinement.
    - KG exploration and traversal agent. 
    - Document retrieval and ranking agent. 
    - Response synthesis and fact-checking agent. 

- **KG Enhanced Retrieval**: Integration of graph-based and vector-based retrieval methods, with dynamic subgraph extraction based on query context. 
- **Improved Query Understanding**: Structured query decomposition into capabilities, resources, and constraints. Context-aware query refinement. 

### Stack
- CrewAI for multi-agent orchestration. 
- LangChain for RAG components. 
- HuggingFace for embeddings and language models. 
- Networkx for KG implementation. 
- Chroma for vector storage. 
 

