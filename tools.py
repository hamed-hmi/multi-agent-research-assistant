"""Utility tools for searching, downloading, and processing papers."""
import os
import time
import re
import requests
import fitz  # PyMuPDF
import arxiv
import networkx as nx
from typing import List, Optional, Dict
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from models import Paper, KnowledgeGraph
from config import llm, embeddings
from sk import wos_sk

# WOS API Configuration
WOS_API_KEY = wos_sk
WOS_BASE_URL = "https://api.clarivate.com/api/wos"

# Rate limiting: Track last WOS API call time to enforce 1 query per second limit
_last_wos_call_time = 0.0


def search_wos(queries: List[str], max_results: int = 6) -> List[Paper]:
    """
    Search Web of Science for papers using the WOS Starter API.
    Includes rate limiting to respect the 1 query per second API limit.
    
    Args:
        queries: List of search query strings
        max_results: Maximum number of results per query
        
    Returns:
        List of unique Paper objects
    """
    global _last_wos_call_time
    
    found_papers = []
    # Standard endpoint for the structure you provided (WOS Starter API)
    search_url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    
    headers = {
        "X-ApiKey": WOS_API_KEY,
        "Accept": "application/json"
    }
    
    for i, query in enumerate(queries):
        # Rate limiting: ensure at least 1.1 seconds between any WOS API calls
        current_time = time.time()
        time_since_last_call = current_time - _last_wos_call_time
        
        if time_since_last_call < 1.1:
            wait_time = 1.1 - time_since_last_call
            print(f"   [RATE LIMIT] Waiting {wait_time:.2f} seconds before WOS query...")
            time.sleep(wait_time)
        
        print(f"   -> Searching Web of Science for: {query}")
        
        try:
            # WOS query format: TS=(query) for topic search
            params = {
                "q": f"TS=({query})",
                "limit": max_results,
                "page": 1
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=30)
            
            # Update last call time after making the API request
            _last_wos_call_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                # Iterate strictly over the 'hits' list
                for hit in data.get("hits", []):
                    
                    # 1. Extract Title
                    title = hit.get("title", "Untitled")
                    
                    # 2. Extract DOI (Safe access: identifiers -> doi)
                    # Your first example had no DOI, so we must handle None safely
                    identifiers = hit.get("identifiers", {})
                    doi = identifiers.get("doi", "")
                    
                    # 3. Extract Date (Safe access: source -> publishYear)
                    source_info = hit.get("source", {})
                    pub_year = source_info.get("publishYear", "Unknown")
                    
                    # 4. Extract Abstract
                    # Note: The JSON structure you provided does NOT contain an abstract.
                    # The WOS Starter API typically excludes it. 
                    abstract = "No abstract available via WOS API"

                    paper = Paper(
                        title=title,
                        summary=abstract,
                        doi=doi,
                        published_date=str(pub_year),
                        source="wos"
                    )
                    found_papers.append(paper)
            
            else:
                print(f"   [ERROR] WOS API returned {response.status_code}: {response.text}")

        except Exception as e:
            print(f"   [ERROR] WOS search failed for query '{query}': {e}")
            continue

    # Deduplicate by DOI (if present) or Title
    unique_papers = {}
    for paper in found_papers:
        # Use DOI as primary key, fallback to lowercased title
        key = paper.doi if paper.doi else paper.title.lower().strip()
        if key not in unique_papers:
            unique_papers[key] = paper
    
    return list(unique_papers.values())


def search_arxiv(queries: List[str], max_results: int = 6) -> List[Paper]:
    """
    Search ArXiv for papers based on queries.
    
    Args:
        queries: List of search query strings
        max_results: Maximum number of results per query
        
    Returns:
        List of unique Paper objects from ArXiv
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
            paper = Paper(
                title=result.title, 
                summary=result.summary, 
                url=result.pdf_url, 
                published_date=str(result.published),
                source="arxiv"
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


def read_local_pdf(pdf_path: str) -> str:
    """
    Read text from a local PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as string, empty string on error
    """
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"   [ERROR] Could not read {pdf_path}: {e}")
        return ""


def match_pdfs_to_papers(papers: List[Paper], pdf_folder: str) -> None:
    """
    Match PDF files in folder to papers based on title/DOI.
    Updates paper.pdf_path for matched papers.
    
    Args:
        papers: List of Paper objects
        pdf_folder: Path to folder containing PDFs
    """
    if not pdf_folder or not os.path.exists(pdf_folder):
        return
    
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    print(f"   -> Found {len(pdf_files)} PDF files in {pdf_folder}")
    
    for paper in papers:
        if paper.pdf_path:  # Already matched
            continue
            
        # Try to match by title (simplified matching)
        paper_title_clean = paper.title.lower().replace(" ", "_")[:30]
        
        for pdf_file in pdf_files:
            pdf_name_clean = pdf_file.stem.lower().replace(" ", "_")
            
            # Simple matching: check if paper title keywords appear in PDF name
            if paper_title_clean in pdf_name_clean or any(
                word in pdf_name_clean for word in paper_title_clean.split("_")[:3] if len(word) > 4
            ):
                paper.pdf_path = str(pdf_file)
                print(f"   [MATCHED] {paper.title[:40]}... -> {pdf_file.name}")
                break


def process_papers(papers: List[Paper], pdf_folder: Optional[str] = None) -> List:
    """
    Download/chunk papers into documents for vector storage.
    Handles both ArXiv (downloads) and WOS (uses local PDFs).
    
    Args:
        papers: List of Paper objects
        pdf_folder: Optional folder path for manually provided PDFs
        
    Returns:
        List of document chunks with metadata
    """
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Match PDFs to papers if folder provided
    if pdf_folder:
        match_pdfs_to_papers(papers, pdf_folder)
    
    for paper in papers:
        text = ""
        
        # Try to get PDF text
        if paper.pdf_path and os.path.exists(paper.pdf_path):
            # Use local PDF
            print(f"   -> Reading local PDF: {paper.pdf_path}")
            text = read_local_pdf(paper.pdf_path)
        elif paper.url and paper.source == "arxiv":
            # Download from ArXiv
            text = download_and_read_pdf(paper.url)
        else:
            print(f"   [SKIP] No PDF available for: {paper.title[:40]}...")
            continue
        
        if text:
            # Add metadata so we know where chunks come from
            new_docs = splitter.create_documents(
                [text], 
                metadatas=[{"source": paper.title, "paper_source": paper.source}]
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


def extract_year_from_date(date_str: str) -> str:
    """
    Extract year from a date string.
    
    Args:
        date_str: Date string (could be full date or just year)
        
    Returns:
        Year as string, or "Unknown" if not found
    """
    if not date_str:
        return "Unknown"
    year_match = re.search(r'\d{4}', date_str)
    if year_match:
        return year_match.group(0)
    return "Unknown"


def format_paper_title_with_year(paper: Paper) -> str:
    """
    Format paper title with publication year.
    
    Args:
        paper: Paper object
        
    Returns:
        Formatted string: "Title (Year)" or "Title" if year not available
    """
    year = extract_year_from_date(paper.published_date)
    if year != "Unknown":
        return f"{paper.title} ({year})"
    return paper.title


def get_or_assign_citation(citation_map: dict, paper_title: str) -> int:
    """
    Get existing citation number for a paper or assign a new one.
    
    Args:
        citation_map: Dictionary mapping paper titles to citation numbers
        paper_title: Title of the paper
        
    Returns:
        Citation number (int)
    """
    if paper_title not in citation_map:
        # Assign next available number
        next_num = len(citation_map) + 1
        citation_map[paper_title] = next_num
    return citation_map[paper_title]


def format_citation(citation_number: int) -> str:
    """
    Format citation number as [N].
    
    Args:
        citation_number: Citation number
        
    Returns:
        Formatted citation string like "[1]"
    """
    return f"[{citation_number}]"


def visualize_categorization_graph(state: Dict, output_file: str = "categorization_graph.png") -> Optional[str]:
    """
    Create a visual NetworkX graph showing categories, subcategories, and papers.
    
    Args:
        state: AgentState dictionary
        output_file: Path to save the visualization image
        
    Returns:
        Path to saved file if successful, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("   [WARNING] matplotlib not available, skipping graph visualization")
        return None
    
    taxonomy_structure = state.get('taxonomy_structure', {})
    subsections = state.get('subsections', {})
    citation_map = state.get('citation_map', {})
    target_papers = state.get('target_papers', [])
    
    if not taxonomy_structure or not subsections:
        print("   [WARNING] No taxonomy or subsections found for visualization")
        return None
    
    # Build reverse mapping: citation number -> paper
    citation_to_paper = {}
    for paper_title, cit_num in citation_map.items():
        paper = next((p for p in target_papers if p.title == paper_title), None)
        if paper:
            citation_to_paper[cit_num] = paper
    
    # Extract papers from each subsection by parsing citations
    subcategory_papers = {}
    for subcat, content in subsections.items():
        # Find all citations like [1], [2], etc.
        citations = re.findall(r'\[(\d+)\]', content)
        paper_nums = [int(c) for c in citations]
        papers = [citation_to_paper[num] for num in paper_nums if num in citation_to_paper]
        subcategory_papers[subcat] = papers
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add topic as root node
    topic = state.get('topic', 'Survey Topic')
    G.add_node(topic, node_type='topic', size=2000)
    
    # Add categories, subcategories, and papers
    for main_cat in taxonomy_structure.get('main_categories', []):
        cat_name = main_cat.get('name', '')
        subcategories = main_cat.get('subcategories', [])
        
        # Only include category if it has written subsections
        written_subsections = [sc for sc in subcategories if sc in subsections]
        if not written_subsections:
            continue
        
        # Add category node
        G.add_node(cat_name, node_type='category', size=1500)
        G.add_edge(topic, cat_name)
        
        # Add subcategories and papers
        for subcat in written_subsections:
            papers = subcategory_papers.get(subcat, [])
            
            # Add subcategory node
            G.add_node(subcat, node_type='subcategory', size=1000, paper_count=len(papers))
            G.add_edge(cat_name, subcat)
            
            # Add paper nodes
            for paper in papers:
                # Truncate long titles for display
                display_title = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
                year = extract_year_from_date(paper.published_date)
                cit_num = citation_map.get(paper.title, '?')
                paper_label = f"[{cit_num}] {display_title}"
                
                G.add_node(paper_label, node_type='paper', size=500, year=year)
                G.add_edge(subcat, paper_label)
    
    if len(G.nodes) == 0:
        print("   [WARNING] No nodes to visualize")
        return None
    
    # Create visualization
    plt.figure(figsize=(24, 16))
    
    # Use hierarchical layout - try different layouts for better visualization
    try:
        # Try to use a hierarchical layout if possible
        # For tree-like structures, spring layout with good parameters works well
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    except:
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.spring_layout(G, seed=42)
    
    # Separate nodes by type
    topic_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'topic']
    category_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'category']
    subcategory_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'subcategory']
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'paper']
    
    # Draw nodes with different colors and sizes
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes, node_color='#FF6B6B', 
                           node_size=2000, alpha=0.9, node_shape='s')
    nx.draw_networkx_nodes(G, pos, nodelist=category_nodes, node_color='#4ECDC4', 
                           node_size=1500, alpha=0.9, node_shape='o')
    nx.draw_networkx_nodes(G, pos, nodelist=subcategory_nodes, node_color='#95E1D3', 
                           node_size=1000, alpha=0.8, node_shape='^')
    nx.draw_networkx_nodes(G, pos, nodelist=paper_nodes, node_color='#FFE66D', 
                           node_size=500, alpha=0.7, node_shape='s')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, arrows=True, 
                           arrowsize=15, arrowstyle='->', width=1.5)
    
    # Draw labels with smaller font
    labels = {}
    for node in G.nodes():
        # Truncate long labels
        if len(node) > 40:
            labels[node] = node[:37] + "..."
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='Topic'),
        mpatches.Patch(color='#4ECDC4', label='Category'),
        mpatches.Patch(color='#95E1D3', label='Subcategory'),
        mpatches.Patch(color='#FFE66D', label='Paper')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.title(f"Taxonomy Categorization: {topic}", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   [SUCCESS] Categorization graph saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"   [ERROR] Failed to save graph: {e}")
        plt.close()
        return None

