# RAG for Search: HELPFUL
A work in progress for plain-text based search method using Retrieval-Augmented Generation 
(RAG) that aims to preserve semantic relationships within structured data entries. The focus 
so far is to maintain data coherence and enhance retrieval accuracy through use-case specific 
splitting and processing. 

The current implementation achieves the following: 
- **Semantic Document Splitting**: Preserve meaningful relationships in structured data. 
- **Coherent Information Retrieva**: Related information is kept together during processing, 
for improved context. Document metatada is also utilizeed to further improve context. 
- **Open Source Stack**: The current implementation is built entirely with open-source 
components, making it more easily replicable and deployable. 

**Ignore all below this. Need to update** 

## Setup (will work towards making this more practical if possible)
On the cloned directory:
```bash
# Linux or macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```
then install the dependencies

```bash
pip install -r requirements.txt
```
## Ongoing Work
- Integration with Knowledge Graphs (Supply trees) for enhanced relationship mapping, 
alternative paths computation, etc. 
- Integration of specialized agents for highly domain-specific tasks. 
- Auto-RAG? To automatically detect appropriate methods of query search. 

## Usage

Now you can place your JSON data files in the OKWs directory, and then import and use the 
semantic RAG implementation: 
```python
from semantic_rag import load_and_process_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

#load and process docs with semantic splitting
documents = load_and_process_documents('./OKWs/')

#init embeddings and vector store
embeddings = HuggingFaceEmbeddings(
	model_name='sentence_transformers/all-MiniLM-L6-v2'
)
vectorstore = Chroma.from_documents(
	documents=documents,
	embeddings=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
```

Now you can run the cells in the notebook sequentially to execute the retrieval and reranking workflow. 

## Notes
- Do not share your API key and do not commit in to version control. Use env variables instead. 
- You can find your API key on jina ai's website landing page, no need to create an account. 
- We are working towards a fully open-sourced version of multi-faceted search based on reranking, perspectives and agentic strategies. Perspectives (generating different interpretations of the user's search) and agents, we hope, will allow us to bypass potential vagueness and ambiguity in user's searches, and make search capabilites far more comprehensive.
- We will also work towards building a more easily deployable and replicable version.
