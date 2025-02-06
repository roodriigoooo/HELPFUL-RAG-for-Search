# RAG for Search: HELPFUL
A work in progress for plain-text based search method using Retrieval-Augmented Generation 
(RAG) that aims to preserve semantic relationships within structured data entries. The focus 
so far is to maintain data coherence and enhance retrieval accuracy through use-case specific 
splitting and processing. 

The current implementation achieves the following: 
- **Semantic Document Splitting**: Preserve meaningful relationships in structured data. 
- **Coherent Information Retrieval**: Related information is kept together during processing, 
for improved context. Document metatada is also utilized to further improve context. 
 

- **kgsandagents branch**:

    - A first, admittedly basic and naive implementation of Knowledge Graphs enhancements and agents to the already demo'd code. It is just a mock version of a KG and agents, to more or less show how i think about these tools in the context of the project.
 
  
### Setup 
On the cloned directory:
```bash
# Linux or macOS
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (Powershell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```
then install the dependencies

```bash
pip install -r requirements.txt
```
#### Windows Specific Issues I ran into
- If you get ```'python' is not recognized```: Ensure Python is added to your PATH.
- If PowerShell execution policy blocks activation: Run ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser```

## Usage
Now you can place your JSON data files in the OKWs directory, and then import and use the 
semantic RAG implementation. I assume each 'document' to follow the following structure:
```json
{
    "title": "Your Title",
    "description": "Your Description",
    "keywords": ["keyword1", "keyword2"],
    "inventory-atoms": [...],
    "product-atoms": [...],
    "tool-list-atoms": [...]
}
```

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

# to perform a search
question = 'your search query here, whether just keywords or complete sentences'
results = retriever.get_relevant_documents(question)

# Process results
for doc in results:
    print(f"Title: {doc.metadata.get('title', 'No Title')}")
    print(f"Content: {doc.page_content}\n")
```

## Ongoing Work
- Integration with Knowledge Graphs (Supply trees) for enhanced relationship mapping, 
alternative paths computation, memory concerns, etc. 
- Integration of specialized agents for highly domain-specific tasks. 
- Auto-RAG? To automatically detect appropriate methods of query search. 

