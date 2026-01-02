"""Data models for the Survey Agent system."""
from typing import TypedDict, List
from pydantic import BaseModel, Field


class ArxivPaper(BaseModel):
    """Represents a paper from ArXiv."""
    title: str
    summary: str
    url: str
    published_date: str
    status: str = "unchecked"


class SearchQueries(BaseModel):
    """Structured output for search queries."""
    queries: List[str] = Field(description="List of up to 3 distinct search queries")


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
    inclusion_criteria: str  # Optional: what papers should include
    exclusion_criteria: str  # Optional: what papers should exclude
    search_queries: List[str]
    papers: List[ArxivPaper]
    taxonomy: str
    future_directions: str
    final_report: str
    revision_number: int
    reviewer_comments: str
    review_status: str

