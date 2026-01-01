"""Utility tools for searching, downloading, and processing papers."""
import requests
import fitz  # PyMuPDF
import arxiv
import networkx as nx
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from models import ArxivPaper, KnowledgeGraph
from config import llm, embeddings


def search_arxiv(queries: List[str], max_results: int = 3) -> List[ArxivPaper]:
    """
    Search ArXiv for papers based on queries.
    
    Args:
        queries: List of search query strings
        max_results: Maximum number of results per query
        
    Returns:
        List of unique ArxivPaper objects
    """
    client = arxiv.Client()
    found_papers = []
    
    for query in queries:
        print(f"   -> Searching ArXiv for: {query}")
        search = arxiv.Search(
            query=query, 
            max_results=max_results, 
            sort_by=arxiv.SortCriterion.Relevance
        )
        for result in client.results(search):
            paper = ArxivPaper(
                title=result.title, 
                summary=result.summary, 
                url=result.pdf_url, 
                published_date=str(result.published)
            )
            found_papers.append(paper)
    
    # Deduplicate by URL
    unique_papers = {p.url: p for p in found_papers}.values()
    return list(unique_papers)


def download_and_read_pdf(url: str) -> str:
    """
    Download and extract text from a PDF URL.
    
    Args:
        url: URL to the PDF
        
    Returns:
        Extracted text as string, empty string on error
    """
    print(f"   -> Downloading: {url}")
    try:
        response = requests.get(url.replace("abs", "pdf"), allow_redirects=True)
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"   [ERROR] {e}")
        return ""


def process_papers(papers: List[ArxivPaper]) -> List:
    """
    Download and chunk papers into documents for vector storage.
    
    Args:
        papers: List of ArxivPaper objects
        
    Returns:
        List of document chunks with metadata
    """
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for paper in papers:
        text = download_and_read_pdf(paper.url)
        if text:
            # Add metadata so we know where chunks come from
            new_docs = splitter.create_documents(
                [text], 
                metadatas=[{"source": paper.title}]
            )
            docs.extend(new_docs)
    
    return docs


def create_vectorstore(documents: List):
    """
    Create a Chroma vector store from documents.
    
    Args:
        documents: List of document chunks
        
    Returns:
        Retriever for semantic search
    """
    if not documents:
        return None
    
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        collection_name="temp_papers"
    )
    return vectorstore.as_retriever()


def build_knowledge_graph(docs, topic: str) -> nx.DiGraph:
    """
    Build a knowledge graph from document chunks.
    
    Args:
        docs: List of document chunks
        topic: Research topic
        
    Returns:
        NetworkX directed graph of concepts and relationships
    """
    print("   -> Building Knowledge Graph from text...")
    
    # Use first 2000 characters of first 3 docs to save tokens
    combined_text = "\n".join([d.page_content[:2000] for d in docs[:3]]) 
    
    structured_llm = llm.with_structured_output(KnowledgeGraph)
    
    system_msg = SystemMessage(
        content="You are a Knowledge Graph extractor. Extract key academic concepts and their relationships."
    )
    user_msg = HumanMessage(
        content=f"Topic: {topic}\n\nText: {combined_text}\n\nExtract 15 distinct triples (Subject, Predicate, Object)."
    )
    
    result = structured_llm.invoke([system_msg, user_msg])
    
    # Create NetworkX Graph
    G = nx.DiGraph()
    for triple in result.triples:
        G.add_edge(triple.subject, triple.object, relation=triple.predicate)
    
    return G


def analyze_graph_centrality(G: nx.DiGraph) -> str:
    """
    Analyze graph centrality to find key concepts.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        String describing top concepts by centrality
    """
    if len(G.nodes) == 0:
        return "No graph created."
    
    # Degree Centrality: Who has the most connections?
    centrality = nx.degree_centrality(G)
    # Sort by importance
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    
    top_concepts = [node for node, score in sorted_nodes[:5]]
    return f"Key Concepts (by Centrality): {', '.join(top_concepts)}"


def format_graph_edges(G: nx.DiGraph) -> str:
    """
    Convert graph edges to a string of relationships.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        String representation of top relationships
    """
    if len(G.edges) == 0:
        return "No relationships found."
    
    edge_strings = []
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'related_to')
        edge_strings.append(f"- {u} is {relation} {v}")
    
    # Return the top 20 relationships to save tokens
    return "\n".join(edge_strings[:20])

