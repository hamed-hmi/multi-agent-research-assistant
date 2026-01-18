"""Utility functions for external interactions and internal helpers."""
import os
import re
import json
from typing import List, Optional, Dict
import arxiv
import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import embeddings
from models import Paper


def search_arxiv(query: str, max_results: int = 6) -> List[Paper]:
    """
    Search ArXiv for papers matching the query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of Paper objects from ArXiv
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for result in search.results():
            paper = Paper(
                title=result.title,
                summary=result.summary,
                url=result.entry_id,
                published_date=result.published.strftime("%Y-%m-%d") if result.published else "",
                source="arxiv"
            )
            papers.append(paper)
        return papers
    except Exception as e:
        print(f"   [ERROR] ArXiv search failed: {e}")
        return []


def search_wos(query: str, api_key: str, max_results: int = 6) -> List[Paper]:
    """
    Search Web of Science for papers matching the query.
    Note: This is a placeholder - implement actual WOS API integration.
    
    Args:
        query: Search query string
        api_key: WOS API key
        max_results: Maximum number of results to return
        
    Returns:
        List of Paper objects from WOS
    """
    # TODO: Implement actual WOS API integration
    print(f"   [WARNING] WOS search not implemented, skipping query: {query}")
    return []


def process_papers(papers: List[Paper], pdf_folder: str = "") -> List[Paper]:
    """
    Process papers, optionally extracting text from PDFs if available.
    
    Args:
        papers: List of Paper objects
        pdf_folder: Optional folder containing PDF files
        
    Returns:
        List of processed Paper objects
    """
    processed = []
    for paper in papers:
        if pdf_folder and paper.pdf_path:
            pdf_path = os.path.join(pdf_folder, paper.pdf_path)
            if os.path.exists(pdf_path):
                try:
                    doc = fitz.open(pdf_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    paper.summary = full_text[:5000]  # Limit summary length
                except Exception as e:
                    print(f"   [WARNING] Failed to process PDF {pdf_path}: {e}")
        processed.append(paper)
    return processed


def create_vectorstore(documents: List[Document]) -> object:
    """
    Create a Chroma vector store from documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Chroma retriever object with compatibility wrapper
    """
    if not documents:
        return None
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="paper_chunks"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create a compatibility wrapper that supports both old and new LangChain APIs
    class RetrieverWrapper:
        def __init__(self, retriever):
            self.retriever = retriever
        
        def invoke(self, query: str):
            """Invoke method for newer LangChain versions."""
            return self.retriever.invoke(query)
        
        def get_relevant_documents(self, query: str):
            """Legacy method for older LangChain versions."""
            if hasattr(self.retriever, 'get_relevant_documents'):
                return self.retriever.get_relevant_documents(query)
            else:
                return self.retriever.invoke(query)
    
    return RetrieverWrapper(retriever)


def build_knowledge_graph(triples: List[dict]) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from knowledge triples.
    
    Args:
        triples: List of dicts with 'subject', 'predicate', 'object' keys
        
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    for triple in triples:
        G.add_edge(triple['subject'], triple['object'], label=triple['predicate'])
    return G


def analyze_graph_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """
    Analyze graph centrality to find important concepts.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping nodes to centrality scores
    """
    if len(G) == 0:
        return {}
    centrality = nx.degree_centrality(G)
    return centrality


def format_graph_edges(G: nx.DiGraph) -> str:
    """
    Format graph edges as a string representation.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Formatted string of edges
    """
    edges = []
    for u, v, data in G.edges(data=True):
        label = data.get('label', '')
        edges.append(f"{u} --[{label}]--> {v}")
    return "\n".join(edges)


def extract_year_from_date(date_str: str) -> str:
    """
    Extract year from a date string (handles various formats).
    
    Args:
        date_str: Date string in various formats (YYYY-MM-DD, YYYY, etc.)
        
    Returns:
        Year as string (YYYY)
    """
    if not date_str:
        return "Unknown"
    
    # Try to extract year (4 digits)
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return year_match.group(0)
    
    return "Unknown"


def format_paper_title_with_year(paper: Paper) -> str:
    """
    Format paper title with publication year.
    
    Args:
        paper: Paper object
        
    Returns:
        Formatted string: "Title (Year)"
    """
    year = extract_year_from_date(paper.published_date)
    return f"{paper.title} ({year})"


def get_or_assign_citation(citation_map: dict, paper_title: str) -> int:
    """
    Get existing citation number or assign a new one.
    
    Args:
        citation_map: Dictionary mapping paper titles to citation numbers
        paper_title: Title of the paper
        
    Returns:
        Citation number (int)
    """
    if paper_title not in citation_map:
        citation_map[paper_title] = len(citation_map) + 1
    return citation_map[paper_title]


def format_citation(citation_number: int) -> str:
    """
    Format citation number as [N].
    
    Args:
        citation_number: Citation number
        
    Returns:
        Formatted citation string
    """
    return f"[{citation_number}]"


def visualize_categorization_graph(state: Dict, output_file: str = "categorization_graph.png") -> Optional[str]:
    """
    Create a NetworkX graph visualization of the taxonomy and paper categorization.
    
    Args:
        state: AgentState dictionary
        output_file: Output filename for the graph image
        
    Returns:
        Path to saved file, or None if failed
    """
    try:
        G = nx.DiGraph()
        taxonomy_structure = state.get('taxonomy_structure', {})
        target_papers = state.get('target_papers', [])
        citation_map = state.get('citation_map', {})
        
        # Add main categories as nodes
        main_categories = taxonomy_structure.get('main_categories', [])
        for cat in main_categories:
            cat_name = cat.get('name', '')
            if cat_name:
                G.add_node(cat_name, node_type='category', size=1000)
                
                # Add subcategories
                subcategories = cat.get('subcategories', [])
                for subcat in subcategories:
                    if subcat:
                        G.add_node(subcat, node_type='subcategory', size=500)
                        G.add_edge(cat_name, subcat, edge_type='contains')
        
        # Add papers and connect to subcategories
        for paper in target_papers:
            paper_title = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
            citation_num = citation_map.get(paper.title, 0)
            node_label = f"[{citation_num}] {paper_title}"
            G.add_node(node_label, node_type='paper', size=300)
            
            # Connect papers to relevant subcategories (simplified - in reality, use RAG results)
            # For now, we'll just show the structure without paper connections
            # You could enhance this by tracking which papers were used in which subsections
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes by type
        categories = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'category']
        subcategories = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'subcategory']
        papers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'paper']
        
        nx.draw_networkx_nodes(G, pos, nodelist=categories, node_color='lightblue', 
                               node_size=1000, alpha=0.9, label='Category')
        nx.draw_networkx_nodes(G, pos, nodelist=subcategories, node_color='lightgreen', 
                               node_size=500, alpha=0.8, label='Subcategory')
        if papers:
            nx.draw_networkx_nodes(G, pos, nodelist=papers, node_color='lightcoral', 
                                   node_size=300, alpha=0.7, label='Paper')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowsize=20, edge_color='gray')
        
        # Draw labels
        labels = {n: n[:30] + "..." if len(n) > 30 else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Taxonomy and Paper Categorization Graph", fontsize=16, fontweight='bold')
        plt.legend(loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   [INFO] Graph visualization saved to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"   [WARNING] Failed to create graph visualization: {e}")
        return None
