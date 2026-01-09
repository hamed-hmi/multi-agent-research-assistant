"""Utility tools for searching, downloading, and processing papers."""
import os
import time
import requests
import fitz  # PyMuPDF
import arxiv
import networkx as nx
from typing import List, Optional
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from models import Paper, KnowledgeGraph
from config import llm, embeddings
from sk import wos_sk

# WOS API Configuration
WOS_API_KEY = wos_sk
WOS_BASE_URL = "https://api.clarivate.com/api/wos"

# Rate limiting: Track last WOS API call time to enforce 1 query per second limit
_last_wos_call_time = 0.0


def search_wos(queries: List[str], max_results: int = 3) -> List[Paper]:
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


def search_arxiv(queries: List[str], max_results: int = 3) -> List[Paper]:
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

