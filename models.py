"""Data models for the Survey Agent system."""
from typing import TypedDict, List
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Represents a paper from any source."""
    title: str
    summary: str
    url: str = ""  # May be empty for WOS papers
    doi: str = ""  # DOI for WOS papers
    published_date: str
    status: str = "unchecked"
    source: str = "arxiv"  # "arxiv", "wos", or "both"
    pdf_path: str = ""  # Local path to PDF if provided manually


class ArxivPaper(Paper):
    """Represents a paper from ArXiv (backward compatibility)."""
    pass


class SearchQueries(BaseModel):
    """Structured output for search queries."""
    queries: List[str] = Field(description="List of up to 3 distinct search queries")


class DualSearchQueries(BaseModel):
    """Structured output for survey and research queries."""
    survey_queries: List[str] = Field(description="List of queries to find survey/review papers")
    research_queries: List[str] = Field(description="List of queries to find research papers")


class ReviewDecision(BaseModel):
    """Review decision with feedback."""
    decision: str = Field(description="Strictly 'PASS' or 'FAIL'")
    feedback: str = Field(description="If FAIL, provide specific instructions on what to fix.")


class KnowledgeTriple(BaseModel):
    """A knowledge triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str


class KnowledgeGraph(BaseModel):
    """Collection of knowledge triples."""
    triples: List[KnowledgeTriple]


class AgentState(TypedDict):
    """State shared across all agent nodes."""
    topic: str
    search_sources: List[str]  # ["arxiv"], ["wos"], or ["arxiv", "wos"]
    search_queries: List[str]  # Legacy field, kept for compatibility
    survey_queries: List[str]  # Queries for finding survey papers
    research_queries: List[str]  # Queries for finding research papers
    papers: List[Paper]  # Legacy field, kept for compatibility
    survey_papers: List[Paper]  # Papers identified as surveys
    target_papers: List[Paper]  # Validated research papers
    pdf_folder: str  # Folder path for manually provided PDFs
    taxonomy: str  # Legacy field, kept for compatibility
    extracted_taxonomies: List[dict]  # Extracted taxonomies from survey papers: [{"source_paper": str, "taxonomy": str}]
    taxonomy_json: str  # Final taxonomy in JSON format
    organized_papers: dict  # Papers organized by taxonomy category: {category: [paper1, paper2, ...]}
    future_directions: str
    final_report: str
    revision_number: int
    reviewer_comments: str
    review_status: str

