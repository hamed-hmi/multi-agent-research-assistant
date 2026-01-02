"""Graph nodes for the Survey Agent workflow."""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END

from models import AgentState, SearchQueries, ReviewDecision
from config import llm
from tools import (
    search_arxiv, 
    process_papers, 
    create_vectorstore,
    build_knowledge_graph,
    analyze_graph_centrality,
    format_graph_edges
)


def planner_node(state: AgentState) -> dict:
    """
    Generate sophisticated search queries with AND/OR logic based on topic and criteria.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with search_queries
    """
    print(f"\n--- PLANNER: Processing '{state['topic']}' ---")
    
    # Build context for query generation
    context_parts = [f"Topic: {state['topic']}"]
    
    if state.get('inclusion_criteria') and state['inclusion_criteria'].strip():
        context_parts.append(f"Inclusion Criteria (MUST include): {state['inclusion_criteria']}")
    
    if state.get('exclusion_criteria') and state['exclusion_criteria'].strip():
        context_parts.append(f"Exclusion Criteria (MUST NOT include): {state['exclusion_criteria']}")
    
    context = "\n".join(context_parts)
    
    system_msg = SystemMessage(content="""You are an expert at creating precise ArXiv search queries. 
Generate a sophisticated search query that use boolean logic to precisely target the research focus.

ArXiv Query Syntax Rules:
- Space-separated terms are implicitly ANDed (e.g., "machine learning neural networks")
- Use explicit "OR" for alternatives (e.g., "(transformer OR attention) language model")
- Use "ANDNOT" to exclude terms (e.g., "transformer ANDNOT CNN")
- Use parentheses to group logical operations
- Terms can be quoted for exact phrases: "exact phrase"
- For inclusion criteria: incorporate them with AND to ensure they're present
- For exclusion criteria: use ANDNOT to explicitly exclude those topics
- Make queries specific and focused, not generic
- Each query should be a complete, valid ArXiv search string

Example: If topic is "federated learning for channel estimation", inclusion is "RIS systems", exclusion is "Internet of Things":
Query: "(channel estimation OR CSI estimation) AND (federated OR distributed OR FL) AND (RIS OR RIS-aided OR reconfigurable intelligent surface ) ANDNOT (IoT or Internet of Things)"

""")
    
    structured_llm = llm.with_structured_output(SearchQueries)
    result = structured_llm.invoke([system_msg, HumanMessage(content=context)])
    return {"search_queries": result.queries}


def search_node(state: AgentState) -> dict:
    """
    Search ArXiv for papers using the generated queries.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with papers
    """
    print("\n--- SEARCHER: Fetching metadata ---")
    papers = search_arxiv(state['search_queries'])
    return {"papers": papers}


def filter_node(state: AgentState) -> dict:
    """
    Filter papers based on relevance, inclusion/exclusion criteria, and exclude survey papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with filtered papers
    """
    print("\n--- FILTER: Screening papers ---")
    filtered = []
    
    # Build criteria context
    criteria_parts = [f"Topic: {state['topic']}"]
    
    if state.get('inclusion_criteria') and state['inclusion_criteria'].strip():
        criteria_parts.append(f"MUST include: {state['inclusion_criteria']}")
    
    if state.get('exclusion_criteria') and state['exclusion_criteria'].strip():
        criteria_parts.append(f"MUST NOT include: {state['exclusion_criteria']}")
    
    criteria_context = "\n".join(criteria_parts)
    
    for paper in state['papers']:
        # First check: Is this a survey/review paper? (We want research papers, not reviews)
        survey_check = f"""Is this paper a survey, review, or literature review paper?
        
Title: {paper.title}
Abstract: {paper.summary}

Look for keywords like: survey, review, literature review, systematic review, meta-analysis, state-of-the-art, comprehensive review.

Respond with ONLY 'YES' if it's a survey/review paper, or 'NO' if it's a research paper."""
        
        survey_res = llm.invoke([HumanMessage(content=survey_check)])
        is_survey = "YES" in survey_res.content.upper()
        
        if is_survey:
            print(f"   [SKIP - SURVEY] {paper.title[:40]}...")
            continue
        
        # Second check: Does it meet inclusion/exclusion criteria?
        relevance_prompt = f"""{criteria_context}

Paper Title: {paper.title}
Paper Abstract: {paper.summary}

Evaluate if this paper:
1. Is relevant to the topic
2. Meets the inclusion criteria (if specified)
3. Does NOT violate the exclusion criteria (if specified)
4. Is a research paper (not a survey/review)

Respond with ONLY 'YES' if the paper should be included, or 'NO' if it should be excluded."""
        
        res = llm.invoke([HumanMessage(content=relevance_prompt)])
        if "YES" in res.content.upper():
            paper.status = "kept"
            filtered.append(paper)
            print(f"   [KEPT] {paper.title[:40]}...")
        else:
            print(f"   [SKIP] {paper.title[:40]}...")
    
    return {"papers": filtered}


def analyst_node(state: AgentState) -> dict:
    """
    Analyze papers using RAG and Knowledge Graph extraction.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with taxonomy and future_directions
    """
    print("\n--- ANALYST: RAG & Knowledge Graph Extraction ---")
    papers = state['papers']
    
    # Return early if no papers
    if not papers: 
        return {"taxonomy": "None", "future_directions": "None"}
    
    # 1. Download & Chunk
    docs = process_papers(papers)
    
    if not docs: 
        return {"taxonomy": "None", "future_directions": "None"}
    
    # 2. Index (Vector Store)
    retriever = create_vectorstore(docs)
    if not retriever:
        return {"taxonomy": "None", "future_directions": "None"}

    # 3. Build & Analyze Knowledge Graph (Graph RAG)
    G = build_knowledge_graph(docs, state['topic'])
    centrality_insight = analyze_graph_centrality(G)
    structural_insight = format_graph_edges(G)
    
    print(f"   [GRAPH] Central: {centrality_insight}")
    print(f"   [GRAPH] Edges: {len(G.edges)} relationships found.")

    # 4. Generate Answers using Hybrid Context (Vector + Graph)
    # Prompt for TAXONOMY: Uses the Structural Edges
    taxonomy_prompt = ChatPromptTemplate.from_template("""
    You are a researcher. Create a hierarchical taxonomy of the methods found.
    
    Use the Knowledge Graph Relationships to define the hierarchy (e.g. if 'A is a type of B', put A under B).
    
    GRAPH RELATIONSHIPS:
    {edges}
    
    TEXT CONTEXT:
    {context}
    
    Topic: {topic}
    Answer:
    """)
    
    taxonomy_chain = (
        {
            "context": retriever | (lambda d: "\n\n".join(x.page_content for x in d)), 
            "topic": RunnablePassthrough(),
            "edges": lambda x: structural_insight
        }
        | taxonomy_prompt
        | llm
        | StrOutputParser()
    )
    
    # Prompt for FUTURE DIRECTIONS: Uses Centrality (what is popular?)
    future_prompt = ChatPromptTemplate.from_template("""
    Identify future research directions. Focus on the 'Central Concepts' identified in the graph.
    
    CENTRAL CONCEPTS:
    {centrality}
    
    TEXT CONTEXT:
    {context}
    
    Topic: {topic}
    Answer:
    """)

    future_chain = (
        {
            "context": retriever | (lambda d: "\n\n".join(x.page_content for x in d)), 
            "topic": RunnablePassthrough(),
            "centrality": lambda x: centrality_insight
        }
        | future_prompt
        | llm
        | StrOutputParser()
    )

    # Execute
    return {
        "taxonomy": taxonomy_chain.invoke(state['topic']),
        "future_directions": future_chain.invoke(state['topic'])
    }


def writer_node(state: AgentState) -> dict:
    """
    Generate the survey report draft.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final_report and revision_number
    """
    rev = state.get('revision_number', 0) + 1
    print(f"\n--- WRITER: Drafting (Rev {rev}) ---")
    prompt = f"""
    Topic: {state['topic']}
    Taxonomy: {state['taxonomy']}
    Future: {state['future_directions']}
    Feedback: {state.get('reviewer_comments', 'None')}
    
    Write a survey report in Markdown. Include Title, Intro, Taxonomy, Future Directions.
    """
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": res.content, "revision_number": rev}


def reviewer_node(state: AgentState) -> dict:
    """
    Review the generated report and provide feedback.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with reviewer_comments and review_status
    """
    print("\n--- REVIEWER: Checking draft ---")
    structured_llm = llm.with_structured_output(ReviewDecision)
    prompt = f"Review this report. It must be >100 words and have 'Taxonomy' and 'Future' headers.\n\n{state['final_report']}"
    res = structured_llm.invoke([HumanMessage(content=prompt)])
    print(f"   Decision: {res.decision}")
    return {"reviewer_comments": res.feedback, "review_status": res.decision}


def router(state: AgentState) -> str:
    """
    Route based on review status.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name or END
    """
    if state['review_status'] == "PASS" or state['revision_number'] > 2:
        return END
    return "writer"

