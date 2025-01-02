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
 

- **kgsandagents branch**:

    - A first, admittedly basic and naive implementation of Knowledge Graphs enhancements and agents to the already demo'd code.
    - Still working on improving it. In any case it is just a mock version of a KG and agents (the newest version now uses CrewAI), to more or less show how i think about these tools in the context of the project.
### Setup (will add more on Windows instructions ASAP)
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
from preprocessing import load_and_process_documents
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

