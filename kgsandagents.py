from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from networkx import DiGraph
import networkx as nx
from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import json
from langchain.docstore.document import Document
import glob
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from semantic_rag import SemanticJSONSplitter, load_and_process_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import numpy as np


class QueryType(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"


class MakerspaceKG:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.graph = DiGraph()
        self.node_embeddings = {}
        self.embeddings = embeddings  # i use the same original embeddings model

    def get_subgraph_for_query(self, query: str) -> DiGraph:
        # get query embedding using our embedding model instance
        query_embedding = self.embeddings.embed_query(query)

        # Find relevant nodes
        relevant_nodes = []
        for node in self.graph.nodes:
            if node in self.node_embeddings:
                similarity = self._calculate_similarity(
                    query_embedding,
                    self.node_embeddings[node]
                )
                if similarity > 0.5:  # Threshold can be adjusted
                    relevant_nodes.append(node)

        return nx.subgraph(self.graph, relevant_nodes)

    def _calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def add_node_with_embedding(self, name: str, description: str, **attrs):
        """Add a node and compute its embedding"""
        self.graph.add_node(name, **attrs)
        self.node_embeddings[name] = self.embeddings.embed_query(description)


class AutoRAGRouter:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def determine_query_type(self, query: str) -> QueryType:
        # Analyze query complexity and structure
        query_features = self._extract_query_features(query)

        if query_features['requires_graph_traversal']:
            return QueryType.GRAPH
        elif query_features['is_semantic']:
            return QueryType.SEMANTIC
        elif query_features['is_keyword_based']:
            return QueryType.KEYWORD
        else:
            return QueryType.HYBRID

    def _extract_query_features(self, query: str) -> Dict[str, bool]:
        # Implement query analysis logic
        # This is a simplified version
        words = query.lower().split()

        return {
            'requires_graph_traversal': any(w in words for w in ['related', 'connected', 'similar']),
            'is_semantic': len(words) > 5 and ' '.join(words).find(' with ') != -1,
            'is_keyword_based': len(words) <= 3
        }


class SearchAgent:
    def __init__(self, llm, knowledge_graph: MakerspaceKG, vector_store):
        self.llm = llm
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.router = AutoRAGRouter()

    def search(self, query: str) -> List[Dict[str, Any]]:
        # Determine search strategy
        query_type = self.router.determine_query_type(query)

        # Execute appropriate search strategy
        if query_type == QueryType.GRAPH:
            results = self._graph_search(query)
        elif query_type == QueryType.SEMANTIC:
            results = self._semantic_search(query)
        elif query_type == QueryType.KEYWORD:
            results = self._keyword_search(query)
        else:
            results = self._hybrid_search(query)

        if results is None:
            results = []

        return self._format_results(results)

    def _graph_search(self, query: str) -> List[Any]:
        try:
            subgraph = self.kg.get_subgraph_for_query(query, self.model)
            if not subgraph:
                return []

            results = []
            for node in subgraph.nodes:
                node_data = subgraph.nodes[node]
                results.append({
                    'name': node,
                    'type': node_data.get('type', 'unknown'),
                    'capabilities': node_data.get('capabilities', []),
                    'resources': node_data.get('resources', [])
                })
            return results
        except Exception as e:
            print(f"Graph search error: {e}")
            return []

    def _semantic_search(self, query: str) -> List[Document]:
        try:
            results = self.vector_store.similarity_search(query)
            return results if results else []
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def _keyword_search(self, query: str) -> List[Document]:
        # Implement basic keyword search using vector store
        try:
            # Split query into keywords
            keywords = query.lower().split()
            results = self.vector_store.similarity_search(
                ' OR '.join(keywords),
                k=5
            )
            return results if results else []
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    def _hybrid_search(self, query: str) -> List[Any]:
        # Combine multiple search strategies
        try:
            graph_results = self._graph_search(query)
            semantic_results = self._semantic_search(query)
            return self._merge_results(graph_results, semantic_results)
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return []

    def _format_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        formatted_results = []

        for result in results:
            if isinstance(result, Document):
                # Handle document objects (from langchain documents object type)
                formatted_results.append({
                    'title': result.metadata.get('title', 'No Title'),
                    'content': result.page_content,
                    'type': 'document',
                    'metadata': result.metadata
                })
            elif isinstance(result, dict):
                # Handle graph search results
                formatted_results.append({
                    'title': result.get('name', 'No Name'),
                    'capabilities': result.get('capabilities', []),
                    'resources': result.get('resources', []),
                    'type': 'graph_node',
                    'metadata': result
                })

        return formatted_results

    def _merge_results(self, graph_results: List[Any], semantic_results: List[Document]) -> List[Any]:
        """Merge results from different search strategies"""
        all_results = []

        # Add graph results
        all_results.extend(graph_results)

        # Add semantic results, avoiding duplicates
        seen_titles = {r.get('title') for r in all_results}
        for doc in semantic_results:
            if doc.metadata.get('title') not in seen_titles:
                all_results.append(doc)
                seen_titles.add(doc.metadata.get('title'))

        return all_results


class MakerspaceMatchingAgent:
    def __init__(self, search_agent: SearchAgent):
        self.search_agent = search_agent

    def find_optimal_makerspaces(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Analyze requirements
        required_capabilities = requirements.get('capabilities', [])
        required_resources = requirements.get('resources', [])

        # Search for potential matches
        matches = []
        for capability in required_capabilities:
            results = self.search_agent.search(capability)
            matches.extend(results)

        # here, we would ideally score and rank matches, we can use an open-source reranking model, or develop the algorithm by scratch
        # ranked_matches = self._rank_matches(matches, requirements)
        # return ranked_matches
        return matches

    def _rank_matches(self, matches: List[Dict[str, Any]],
                      requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Implement ranking logic
        pass


from semantic_rag import SemanticJSONSplitter, load_and_process_documents


def setup_enhanced_rag():
    # Initialize components
    documents = load_and_process_documents('./OKWs/')
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    kg = MakerspaceKG(embeddings)

    # Add some test data to the knowledge graph
    kg.add_node_with_embedding(
        name="Woodworking Shop",
        description="Specializes in sustainable wood processing and furniture making",
        capabilities=["woodworking", "furniture making"],
        resources=["sustainable wood"]
    )

    kg.add_node_with_embedding(
        name="Textile Studio",
        description="Focuses on organic fabric processing and sustainable textile work",
        capabilities=["textile processing"],
        resources=["organic fabrics"]
    )

    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
    llm = HuggingFacePipeline(pipeline=pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    ))

    search_agent = SearchAgent(llm, kg, vector_store)
    matching_agent = MakerspaceMatchingAgent(search_agent)

    return matching_agent


matching_agent = setup_enhanced_rag()
requirements = {
    "capabilities": ["woodworking", "textile processing"],
    "resources": ["sustainable wood", "organic fabrics"],
    "preferences": {
        "location": "local",
        "sustainability": "high"
    }
}

matches = matching_agent.find_optimal_makerspaces(requirements)

print("Found matches:")
for match in matches:
    print("\nMatch:")
    print(f"Title: {match.get('title', 'No title')}")
    if match.get('type') == 'graph_node':
        print(f"Capabilities: {match.get('capabilities', [])}")
        print(f"Resources: {match.get('resources', [])}")
    elif match.get('type') == 'document':
        print(f"Content: {match.get('content', '')}")
    print(f"Type: {match.get('type', 'unknown')}")
