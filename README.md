# RAG for Search: HELPFUL
A work in progress for plain-text based search method using reranking, perspectives and eventually agents. The document retrieval and reranking pipeline were done using LangChain and Jina ai. 

## Setup (will work towards making this more practical if possible)
On the cloned directory:
```bash
python3 -m venv venv
```
or if you are using windows:
```bash
python -m venv venv
venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
**Setting up the key:**
```bash
export JINA_API_KEY=your_key_here
```
or, on windows:
```bash
set JINA_API_KEY=your_key_here
```
then, in your local notebook:
```python
import os
jina_api_key = os.getenv('JINA_API_KEY')

if not jina_api_key:
    raise ValueError('Set the key as an env variable.')
```
Now you can place your JSON data files in the OKWs directory. 
```bash
jupyter notebook
```
Now you can run the cells in the notebook sequentially to execute the retrieval and reranking workflow. 

## Notes
- Do not share your API key and do not commit in to version control. Use env variables instead. 
- You can find your API key on jina ai's website landing page, no need to create an account. 
- We are working towards a fully open-sourced version of multi-faceted search based on reranking, perspectives and agentic strategies. Perspectives (generating different interpretations of the user's search) and agents, we hope, will allow us to bypass potential vagueness and ambiguity in user's searches, and make search capabilites far more comprehensive.
- We will also work towards building a more easily deployable and replicable version.
