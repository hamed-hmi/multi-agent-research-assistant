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


class AgentState(TypedDict, total=False):
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
    taxonomy_structure: dict  # Parsed taxonomy structure: {"main_categories": [{"name": str, "subcategories": [str]}]}
    paper_retriever: object  # Vector store retriever for RAG (from create_vectorstore)
    citation_map: dict  # Paper title → citation number: {paper_title: int}
    subsections: dict  # Subcategory → markdown content: {subcategory: str}
    sections: dict  # Category → markdown content: {category: str}
    future_directions: str
    final_report: str
    revision_number: int
    reviewer_comments: str
    review_status: str
    # Feedback mechanism fields
    survey_query_feedback: str  # Feedback for regenerating survey queries
    survey_query_retry_count: int  # Number of retries for survey queries
    min_survey_papers: int  # Minimum number of survey papers needed before validation (default: 5)
    min_validated_surveys: int  # Minimum number of validated survey papers needed after validation (default: 3)
    max_survey_query_retries: int  # Maximum retries for survey queries (default: 2)
    termination_message: str  # Message to display if workflow terminates early
    workflow_terminated: bool  # Flag indicating if workflow should terminate

