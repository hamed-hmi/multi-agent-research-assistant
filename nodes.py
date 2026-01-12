"""Graph nodes for the Survey Agent workflow."""
import os
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END

from models import AgentState, SearchQueries, DualSearchQueries, ReviewDecision
from config import llm
from tools import (
    search_arxiv,
    search_wos, 
    process_papers, 
    create_vectorstore,
    build_knowledge_graph,
    analyze_graph_centrality,
    format_graph_edges,
    format_paper_title_with_year,
    extract_year_from_date,
    get_or_assign_citation,
    format_citation
)


def initial_split_node(state: AgentState) -> dict:
    """
    Initial pass-through node that allows both query planners to start.
    This node does nothing but allows the graph to split into parallel paths.
    
    Args:
        state: Current agent state
        
    Returns:
        Empty dict (pass-through)
    """
    print("\n--- INITIAL SPLIT: Starting parallel query generation paths ---")
    return {}


def survey_query_planner_node(state: AgentState) -> dict:
    """
    Generate search queries specifically for survey papers.
    Can accept feedback to regenerate queries with broader scope.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_queries
    """
    print(f"\n--- SURVEY QUERY PLANNER: Generating survey queries for '{state['topic']}' ---")
    
    # Check if there's feedback for survey queries
    survey_feedback = state.get('survey_query_feedback', '')
    is_retry = bool(survey_feedback)
    previous_queries = state.get('survey_queries', [])
    
    if is_retry:
        print("   [RETRY] Regenerating survey queries with broader scope based on feedback")
    
    system_msg_content = """You are an expert at creating precise ArXiv and Web of Science search queries for finding survey, review, and literature review papers.

ArXiv/WOS Query Syntax Rules:
- Space-separated terms are implicitly ANDed (e.g., "machine learning neural networks")
- Use explicit "OR" for alternatives (e.g., "(transformer OR attention) language model")
- Use "ANDNOT" to exclude terms (e.g., "transformer ANDNOT CNN")
- Use parentheses to group logical operations
- Terms can be quoted for exact phrases: "exact phrase"
- Make queries specific and focused, not generic
- Each query should be a complete, valid search string

For Survey Queries: Include terms like "survey", "review", "systematic review", "literature review", "state-of-the-art", "comprehensive review"
Generate 3 queries, and try to target survey papers that could be helpful to look at to write a new survey paper on the topic.
Example: If topic is "federated learning for channel estimation":
Query 1: "(federated learning OR distributed learning) AND (channel estimation OR CSI estimation) AND (survey OR review OR literature review)"
Query 2: "(federated learning OR distributed learning) AND (telecomunication OR wireless systems) AND (survey OR review OR literature review)"
Query 3: "(deep learning OR neural networks) AND (channel estimation OR CSI estimation) AND (survey OR review OR literature review)"


"""
    
    # Add feedback instructions if this is a retry
    if is_retry:
        previous_queries_text = "\n".join([f"- {q}" for q in previous_queries]) if previous_queries else "None"
        system_msg_content += f"""
IMPORTANT: You are regenerating survey queries because the previous queries returned too few results.

Previous queries that didn't work well:
{previous_queries_text}

Previous feedback: {survey_feedback}

The goal is to find helpful survey papers that cover related topics, even if not exactly matching the specific topic.
Generate NEW queries that are broader than the previous ones - consider related domains, more general terms, and parent domains.
"""
    
    system_msg = SystemMessage(content=system_msg_content)
    
    user_content = f"Topic: {state['topic']}"
    if is_retry:
        user_content += f"\n\nPrevious survey queries returned insufficient results. Please generate broader survey queries."
    
    structured_llm = llm.with_structured_output(SearchQueries)
    result = structured_llm.invoke([system_msg, HumanMessage(content=user_content)])
    
    # Reset feedback after using it
    return {
        "survey_queries": result.queries,
        "survey_query_feedback": ""  # Clear feedback after use
    }


def research_query_planner_node(state: AgentState) -> dict:
    """
    Generate search queries specifically for research papers (target papers).
    This runs once and doesn't need feedback since scope is user-defined.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with research_queries
    """
    print(f"\n--- RESEARCH QUERY PLANNER: Generating research queries for '{state['topic']}' ---")
    
    system_msg_content = """You are an expert at creating precise ArXiv and Web of Science search queries for finding original research papers (NOT surveys or reviews).

ArXiv/WOS Query Syntax Rules:
- Space-separated terms are implicitly ANDed (e.g., "machine learning neural networks")
- Use explicit "OR" for alternatives (e.g., "(transformer OR attention) language model")
- Use "ANDNOT" to exclude terms (e.g., "transformer ANDNOT CNN")
- Use parentheses to group logical operations
- Terms can be quoted for exact phrases: "exact phrase"
- Make queries specific and focused, not generic
- Each query should be a complete, valid search string

For Research Queries: Use the topic directly, but EXCLUDE survey/review terms

Example: If topic is "federated learning for channel estimation":
Research Query: "(channel estimation OR CSI estimation) AND (federated OR distributed OR FL) ANDNOT (survey OR review OR literature review)"

"""
    
    system_msg = SystemMessage(content=system_msg_content)
    user_content = f"Topic: {state['topic']}"
    
    structured_llm = llm.with_structured_output(SearchQueries)
    result = structured_llm.invoke([system_msg, HumanMessage(content=user_content)])
    
    return {
        "research_queries": result.queries,
        "search_queries": result.queries  # Legacy compatibility
    }


def survey_feedback_check_node(state: AgentState) -> dict:
    """
    Check if enough survey papers were found and provide feedback if needed.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_query_feedback if needed, or empty dict to continue
    """
    print("\n--- SURVEY FEEDBACK CHECK: Evaluating survey paper count ---")
    survey_papers = state.get('survey_papers', [])
    min_survey_papers = state.get('min_survey_papers', 5)  # Default threshold
    survey_query_retry_count = state.get('survey_query_retry_count', 0)
    max_retries = state.get('max_survey_query_retries', 2)  # Max retries to avoid infinite loops
    
    print(f"   [INFO] Found {len(survey_papers)} survey papers (minimum needed: {min_survey_papers})")
    print(f"   [INFO] Retry count: {survey_query_retry_count}/{max_retries}")
    
    if len(survey_papers) >= min_survey_papers:
        print("   [SUCCESS] Enough survey papers found, continuing...")
        return {}  # Continue normally
    
    if survey_query_retry_count >= max_retries:
        print(f"   [WARNING] Max retries reached ({max_retries}). Continuing with {len(survey_papers)} survey papers.")
        return {}  # Continue anyway to avoid infinite loop
    
    # Generate simple feedback message - the planner's LLM will figure out how to broaden
    feedback = f"""Previous queries returned only {len(survey_papers)} survey papers, but we need at least {min_survey_papers}. Please broaden the search scope significantly - include related domains, use more general terms, and consider parent domains that might contain relevant surveys."""
    
    print(f"   [FEEDBACK] Setting feedback for query refinement")
    print(f"   [ACTION] Will retry with broader queries (attempt {survey_query_retry_count + 1})")
    
    return {
        "survey_query_feedback": feedback,
        "survey_query_retry_count": survey_query_retry_count + 1,
        "survey_papers": []  # Clear current survey papers to retry search
    }


def survey_feedback_router(state: AgentState) -> str:
    """
    Route based on survey feedback check.
    
    Args:
        state: Current agent state
        
    Returns:
        "survey_query_planner" if retry needed, "survey_selector" if enough papers found
    """
    survey_query_feedback = state.get('survey_query_feedback', '')
    
    if survey_query_feedback:
        return "survey_query_planner"  # Retry with feedback
    else:
        return "survey_selector"  # Continue to selection (with validation)


def target_paper_termination_check_node(state: AgentState) -> dict:
    """
    Check if enough target papers were found. If less than 2, terminate with user message.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with termination message if needed
    """
    print("\n--- TARGET PAPER TERMINATION CHECK: Evaluating target paper count ---")
    target_papers = state.get('target_papers', [])
    min_target_papers = 2
    
    print(f"   [INFO] Found {len(target_papers)} target papers (minimum needed: {min_target_papers})")
    
    if len(target_papers) >= min_target_papers:
        print("   [SUCCESS] Enough target papers found, continuing...")
        return {}  # Continue normally
    
    # Generate alternative topic suggestion
    topic = state.get('topic', '')
    suggestion_prompt = f"""The search for research papers on the topic "{topic}" returned only {len(target_papers)} relevant papers, which is insufficient (need at least {min_target_papers}).

Suggest 2-3 alternative, broader research topics that are related but might have more available papers. The suggestions should be:
- Related to the original topic
- Broader in scope to increase paper availability
- Still relevant and useful for research

Provide the suggestions as a concise list."""
    
    res = llm.invoke([HumanMessage(content=suggestion_prompt)])
    suggestions = res.content.strip()
    
    termination_message = f"""
{'='*80}
INSUFFICIENT PAPERS FOUND
{'='*80}

Topic: {topic}
Found Papers: {len(target_papers)}
Required Minimum: {min_target_papers}

Unfortunately, there are not enough research papers available for the specific topic "{topic}".

SUGGESTED ALTERNATIVE TOPICS:
{suggestions}

Please try running the survey agent with one of these alternative topics, or provide a broader topic that might have more available papers.

{'='*80}
"""
    
    print(termination_message)
    
    return {
        "termination_message": termination_message,
        "workflow_terminated": True
    }


def survey_search_node(state: AgentState) -> dict:
    """
    Search for survey papers only.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_papers
    """
    print("\n--- SURVEY SEARCHER: Fetching survey paper metadata ---")
    sources = state.get('search_sources', ['arxiv'])
    survey_papers = []
    
    survey_queries = state.get('survey_queries', [])
    if not survey_queries:
        print("   [WARNING] No survey queries found")
        return {"survey_papers": []}
    
    if 'arxiv' in sources:
        print("   -> Searching ArXiv for surveys...")
        arxiv_surveys = search_arxiv(survey_queries, max_results=10)
        survey_papers.extend(arxiv_surveys)
    
    if 'wos' in sources:
        print("   -> Searching Web of Science for surveys...")
        wos_surveys = search_wos(survey_queries, max_results=10)
        survey_papers.extend(wos_surveys)
        print(f"   [INFO] Found {len(wos_surveys)} WOS survey papers. PDFs must be provided manually.")
    
    # Deduplicate by title
    unique_surveys = {}
    for paper in survey_papers:
        title_key = paper.title.lower().strip()
        if title_key not in unique_surveys:
            unique_surveys[title_key] = paper
    
    print(f"\n   [SUMMARY] Found {len(unique_surveys)} unique survey papers")
    
    return {
        "survey_papers": list(unique_surveys.values())
    }


def research_search_node(state: AgentState) -> dict:
    """
    Search for research papers only.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with target_papers (initial list before validation)
    """
    print("\n--- RESEARCH SEARCHER: Fetching research paper metadata ---")
    sources = state.get('search_sources', ['arxiv'])
    research_papers = []
    
    research_queries = state.get('research_queries', [])
    if not research_queries:
        print("   [WARNING] No research queries found")
        return {"target_papers": [], "papers": []}
    
    if 'arxiv' in sources:
        print("   -> Searching ArXiv for research papers...")
        arxiv_research = search_arxiv(research_queries, max_results=20)
        research_papers.extend(arxiv_research)
    
    if 'wos' in sources:
        print("   -> Searching Web of Science for research papers...")
        wos_research = search_wos(research_queries, max_results=20)
        research_papers.extend(wos_research)
        print(f"   [INFO] Found {len(wos_research)} WOS research papers. PDFs must be provided manually.")
    
    # Deduplicate by title
    unique_research = {}
    for paper in research_papers:
        title_key = paper.title.lower().strip()
        if title_key not in unique_research:
            unique_research[title_key] = paper
    
    print(f"\n   [SUMMARY] Found {len(unique_research)} unique research papers")
    
    return {
        "target_papers": list(unique_research.values()),
        "papers": list(unique_research.values())  # Legacy compatibility
    }


# ========== SURVEY TRACK NODES ==========

def survey_validator_node(state: AgentState) -> dict:
    """
    Filter survey papers to keep only high-quality, relevant surveys.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with validated survey_papers
    """
    print("\n--- SURVEY VALIDATOR: Filtering survey papers ---")
    validated = []
    survey_papers = state.get('survey_papers', [])
    
    if not survey_papers:
        print("   [INFO] No survey papers to validate")
        return {"survey_papers": []}
    
    for paper in survey_papers:
        validation_prompt = f"""Evaluate this survey paper for quality and relevance.

Topic: {state['topic']}
Title: {paper.title}
Abstract: {paper.summary}

Check:
1. Is it actually a survey/review paper? 
2. Is it relevant to the topic?

Respond with ONLY 'YES' if it should be kept, or 'NO' if it should be excluded."""
        
        res = llm.invoke([HumanMessage(content=validation_prompt)])
        if "YES" in res.content.upper():
            validated.append(paper)
            print(f"   [KEPT] {format_paper_title_with_year(paper)}")
        else:
            print(f"   [SKIP] {format_paper_title_with_year(paper)}")
    
    print(f"\n   [SUMMARY] Validated {len(validated)} out of {len(survey_papers)} survey papers")
    return {"survey_papers": validated}


def survey_validation_feedback_check_node(state: AgentState) -> dict:
    """
    Check if enough validated survey papers were found after selection and validation.
    If less than 3 validated surveys, provide feedback to retry with broader queries.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_query_feedback if needed, or empty dict to continue
    """
    print("\n--- SURVEY VALIDATION FEEDBACK CHECK: Evaluating selected and validated survey count ---")
    validated_surveys = state.get('survey_papers', [])
    min_validated_surveys = state.get('min_validated_surveys', 3)  # Default threshold: 3
    survey_query_retry_count = state.get('survey_query_retry_count', 0)
    max_retries = state.get('max_survey_query_retries', 2)  # Max retries to avoid infinite loops
    
    print(f"   [INFO] Selected and validated {len(validated_surveys)} survey papers (minimum needed: {min_validated_surveys})")
    print(f"   [INFO] Retry count: {survey_query_retry_count}/{max_retries}")
    
    if len(validated_surveys) >= min_validated_surveys:
        print("   [SUCCESS] Enough validated survey papers found, continuing...")
        return {}  # Continue normally
    
    if survey_query_retry_count >= max_retries:
        print(f"   [WARNING] Max retries reached ({max_retries}). Continuing with {len(validated_surveys)} validated survey papers.")
        return {}  # Continue anyway to avoid infinite loop
    
    # Generate simple feedback message - the planner's LLM will figure out how to broaden
    previous_queries = state.get('survey_queries', [])
    feedback = f"""Previous queries returned only {len(validated_surveys)} valid survey/review papers, but we need at least {min_validated_surveys}. The selected papers must be: (1) actually survey/review papers, (2) relevant to the topic, and (3) of sufficient quality. Please broaden the search scope - include related domains, use more general terms, and consider parent domains that might contain relevant surveys."""
    
    print(f"   [FEEDBACK] Setting feedback for query refinement")
    print(f"   [ACTION] Will retry with broader queries (attempt {survey_query_retry_count + 1})")
    
    return {
        "survey_query_feedback": feedback,
        "survey_query_retry_count": survey_query_retry_count + 1,
        "survey_papers": []  # Clear current survey papers to retry search
    }


def survey_validation_feedback_router(state: AgentState) -> str:
    """
    Route based on survey validation feedback check (after selector).
    
    Args:
        state: Current agent state
        
    Returns:
        "survey_query_planner" if retry needed, "taxonomy_extractor" if enough validated surveys
    """
    survey_query_feedback = state.get('survey_query_feedback', '')
    
    if survey_query_feedback:
        return "survey_query_planner"  # Retry with feedback
    else:
        return "taxonomy_extractor"  # Continue to taxonomy extraction


def survey_selector_node(state: AgentState) -> dict:
    """
    Select and validate the top 3 survey papers based on:
    1. Usefulness for writing a survey paper on the topic
    2. Ability to help find/extract taxonomy
    3. Recency (more recent papers are preferred)
    4. Complementarity (papers should work together)
    
    During selection, validates that papers are actually review/survey papers.
    If fewer than 3 valid review papers are found, sets feedback flag for retry.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with top 3 selected and validated survey_papers
    """
    print("\n--- SURVEY SELECTOR: Selecting and validating top 3 surveys ---")
    survey_papers = state.get('survey_papers', [])
    topic = state.get('topic', '')
    
    if not survey_papers:
        print("   [WARNING] No survey papers to select from")
        return {"survey_papers": []}
    
    print(f"   [INFO] Selecting top 3 from {len(survey_papers)} survey papers")
    
    # Prepare paper information with publication year
    papers_info = []
    for i, paper in enumerate(survey_papers):
        # Extract year from published_date (could be full date or just year)
        year = extract_year_from_date(paper.published_date)
        
        papers_info.append({
            "index": i,
            "title": paper.title,
            "abstract": paper.summary[:500] if paper.summary else "No abstract",
            "year": year,
            "source": paper.source
        })
    
    papers_list = "\n\n".join([
        f"[{p['index']}] Title: {p['title']}\n"
        f"    Abstract: {p['abstract']}\n"
        f"    Published: {p['year']} ({p['source']})"
        for p in papers_info
    ])
    
    selection_prompt = f"""You need to select 3 survey/review papers from the following papers. These papers will be used to:
1. Write a new survey paper on the topic: "{topic}"
2. Extract and learn from their taxonomy structures to design a taxonomy for the new survey

CRITICAL REQUIREMENTS - DO NOT SELECT PAPERS THAT:
- Are NOT survey/review papers 
- Are NOT relevant to the topic



Topic: {topic}

Survey Papers:
{papers_list}


Think about:
- Which papers are actually surveys/reviews AND relevant to the topic?
- Which 3 papers together would give the best overall coverage?
- Which combination avoids redundancy and maximizes diversity?
- Which set of taxonomies would be most useful to learn from and combine?

If you cannot find 3 valid survey/review papers that meet ALL the requirements above, select the best available ones that are at least survey/review papers and relevant to the topic.


Respond with ONLY the indices (numbers) of the 3 selected papers (if exists and all valid), separated by commas.
Example: "0, 3, 7"
"""
    
    res = llm.invoke([HumanMessage(content=selection_prompt)])
    selected_indices_str = res.content.strip()
    
    # Parse the selected indices
    try:
        # Extract numbers from the response
        indices = [int(x.strip()) for x in re.findall(r'\d+', selected_indices_str)]
        # Take first 3 unique indices
        selected_indices = []
        for idx in indices:
            if idx not in selected_indices and 0 <= idx < len(survey_papers):
                selected_indices.append(idx)
                if len(selected_indices) >= 3:
                    break
        
        # If we don't have 3, fill with remaining papers
        if len(selected_indices) < 3:
            for i in range(len(survey_papers)):
                if i not in selected_indices:
                    selected_indices.append(i)
                    if len(selected_indices) >= 3:
                        break
        
        selected_papers = [survey_papers[i] for i in selected_indices[:3]]
        
        # Display selected papers
        if selected_papers:
            print(f"\n   [SELECTED] Top {len(selected_papers)} surveys:")
            for i, paper in enumerate(selected_papers, 1):
                print(f"   [{i}] {format_paper_title_with_year(paper)}")
        
        return {"survey_papers": selected_papers}
        
    except Exception as e:
        print(f"   [ERROR] Failed to parse selection: {e}")
        print(f"   [FALLBACK] Using first 3 papers")
        # Fallback: use first 3 papers
        return {"survey_papers": survey_papers[:3]}


def taxonomy_extractor_node(state: AgentState) -> dict:
    """
    Extract taxonomy structures from validated survey papers using RAG.
    Uses semantic search to find taxonomy-related content in papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with extracted_taxonomies (list of taxonomy strings)
    """
    print("\n--- TAXONOMY EXTRACTOR: Extracting taxonomies using RAG ---")
    survey_papers = state.get('survey_papers', [])
    extracted_taxonomies = []
    topic = state.get('topic', '')
    
    if not survey_papers:
        print("   [INFO] No survey papers to extract taxonomies from")
        return {"extracted_taxonomies": []}
    
    # Process papers (download/read PDFs)
    pdf_folder = state.get('pdf_folder', '')
    all_docs = process_papers(survey_papers, pdf_folder)
    
    if not all_docs:
        print("   [WARNING] Could not process survey papers for taxonomy extraction")
        return {"extracted_taxonomies": []}
    
    # Group docs by paper title
    papers_docs = {}
    for doc in all_docs:
        paper_title = doc.metadata.get('source', 'Unknown')
        if paper_title not in papers_docs:
            papers_docs[paper_title] = []
        papers_docs[paper_title].append(doc)
    
    # Extract taxonomy from each survey paper using RAG
    for paper in survey_papers:
        paper_docs = papers_docs.get(paper.title, [])
        if not paper_docs:
            print(f"   [SKIP] No content extracted for: {format_paper_title_with_year(paper)}")
            continue
        
        print(f"   [PROCESSING] {format_paper_title_with_year(paper)}")
        
        # Create vector store for this paper's chunks
        retriever = create_vectorstore(paper_docs)
        if not retriever:
            print(f"   [SKIP] Could not create vector store for: {format_paper_title_with_year(paper)}")
            continue
        
        # Use RAG to find taxonomy-related content
        # Query for taxonomy-related sections
        taxonomy_queries = [
            f"paper structure organization",
            f"hierarchical categories subcategories",
            f"taxonomy structure organization",
            f"categorizing the studies"
        ]
        
        # Retrieve relevant chunks for taxonomy extraction
        retrieved_chunks = []
        seen_chunks = set()
        
        for query in taxonomy_queries:
            try:
                chunks = retriever.invoke(query)
                for chunk in chunks:
                    # Deduplicate by content
                    chunk_key = chunk.page_content[:100]  # Use first 100 chars as key
                    if chunk_key not in seen_chunks:
                        retrieved_chunks.append(chunk)
                        seen_chunks.add(chunk_key)
            except Exception as e:
                print(f"   [WARNING] Error retrieving chunks for query '{query}': {e}")
                continue
        
        # If no chunks retrieved, fallback to first few chunks
        if not retrieved_chunks:
            print(f"   [FALLBACK] No taxonomy chunks found, using first chunks")
            retrieved_chunks = paper_docs[:5]
        
        # Limit to top 10 most relevant chunks to avoid token limits
        retrieved_chunks = retrieved_chunks[:10]
        
        # Combine retrieved chunks
        paper_text = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        paper_text = paper_text[:8000]  # Limit total length 
        
        extraction_prompt = f"""Extract the taxonomy, categorization, or organizational structure from this survey paper. How do they categorize their target studies?

Paper Title: {paper.title}

Paper Content (retrieved sections):
{paper_text}


If you find a taxonomy, describe it in a structured format (hierarchical categories and subcategories).
If no clear taxonomy is found, respond with "NO_TAXONOMY".

Respond with the taxonomy structure or "NO_TAXONOMY":"""
        
        res = llm.invoke([HumanMessage(content=extraction_prompt)])
        taxonomy_text = res.content.strip()
        
        if taxonomy_text and "NO_TAXONOMY" not in taxonomy_text.upper():
            extracted_taxonomies.append({
                "source_paper": paper.title,
                "taxonomy": taxonomy_text
            })
            print(f"   [EXTRACTED] Taxonomy from: {format_paper_title_with_year(paper)}")
        else:
            print(f"   [NO_TAXONOMY] {format_paper_title_with_year(paper)}")
    
    return {"extracted_taxonomies": extracted_taxonomies}


def taxonomy_designer_node(state: AgentState) -> dict:
    """
    Design unified taxonomy JSON structure for the specific topic by looking at the taxonomies extracted from related survey papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with taxonomy_json
    """
    print("\n--- TAXONOMY DESIGNER: Creating unified taxonomy ---")
    extracted_taxonomies = state.get('extracted_taxonomies', [])
    
    if not extracted_taxonomies:
        print("   [WARNING] No taxonomies extracted, creating from topic")
        # Create a basic taxonomy from the topic
        design_prompt = f"""You are a research assistant. You want to help specifying the main sections and subsections of a survey paper that categorize the target studies in a specific topic.

Topic: {state['topic']}
    
Design a taxonomy with main categories and subcategories that would organize research papers in the specific topic.
Return a JSON structure with this format:
{{
  "main_categories": [
    {{
      "name": "Category Name",
      "subcategories": ["Subcategory 1", "Subcategory 2"]
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""
    else:
        # Combine extracted taxonomies
        taxonomy_texts = "\n\n---\n\n".join([
            f"From: {t['source_paper']}\n{t['taxonomy']}" 
            for t in extracted_taxonomies
        ])
        
        design_prompt = f"""You are a research assistant. You want to help specifying the main sections and subsections of a survey paper that categorize the target studies in the specific topic.

Topic: {state['topic']}
    
Design a taxonomy with main categories and subcategories that would organize research papers in the specific topic by looking at the taxonomies extracted from some related survey papers. 
If the related survey papers are not that relevant or only consider one aspect, the priority is the specific topic. Remember we want to categorize the studies that exactly go under the specific topic.


Extracted Taxonomies:
{taxonomy_texts}

Create a unified taxonomy that:
1. Ensures coverage of the specific topic and only that.
2. Has a clear hierarchical structure
3. Is suitable for organizing research papers in the specific topic

Return a JSON structure with this format:
{{
  "main_categories": [
    {{
      "name": "Category Name",
      "subcategories": ["Subcategory 1", "Subcategory 2"]
    }}
  ]
}}

Return ONLY valid JSON, no additional text."""
    
    res = llm.invoke([HumanMessage(content=design_prompt)])
    taxonomy_json = res.content.strip()
    
    # Try to extract JSON if wrapped in markdown code blocks
    if "```json" in taxonomy_json:
        taxonomy_json = taxonomy_json.split("```json")[1].split("```")[0].strip()
    elif "```" in taxonomy_json:
        taxonomy_json = taxonomy_json.split("```")[1].split("```")[0].strip()
    
    print(f"   [SUCCESS] Taxonomy designed ({len(taxonomy_json)} characters)")
    return {"taxonomy_json": taxonomy_json}


# ========== PAPER TRACK NODES ==========

def relevance_judge_node(state: AgentState) -> dict:
    """
    Evaluate research papers for relevance to topic.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with validated target_papers
    """
    print("\n--- RELEVANCE JUDGE: Evaluating research papers ---")
    validated = []
    target_papers = state.get('target_papers', [])
    
    if not target_papers:
        print("   [INFO] No research papers to evaluate")
        return {"target_papers": []}
    
    for paper in target_papers:
        # First check: Is this actually a survey? (should go to survey track)
        survey_check = f"""Is this paper a survey, review, or literature review paper?

Title: {paper.title}
Abstract: {paper.summary}

Look for keywords like: survey, review, literature review, systematic review, meta-analysis, state-of-the-art, comprehensive review.

Respond with ONLY 'YES' if it's a survey/review paper, or 'NO' if it's a research paper."""
        
        survey_res = llm.invoke([HumanMessage(content=survey_check)])
        is_survey = "YES" in survey_res.content.upper()
        
        if is_survey:
            print(f"   [SKIP - SURVEY] {format_paper_title_with_year(paper)}")
            continue
        
        # Second check: Is it relevant to the topic?
        relevance_prompt = f"""Topic: {state['topic']}

Paper Title: {paper.title}
Paper Abstract: {paper.summary}

Evaluate if this research paper is relevant to the topic.

Respond with ONLY 'YES' if the paper should be included, or 'NO' if it should be excluded."""
        
        res = llm.invoke([HumanMessage(content=relevance_prompt)])
        if "YES" in res.content.upper():
            paper.status = "validated"
            validated.append(paper)
            print(f"   [KEPT] {format_paper_title_with_year(paper)}")
        else:
            print(f"   [SKIP] {format_paper_title_with_year(paper)}")
    
    print(f"\n   [SUMMARY] Validated {len(validated)} out of {len(target_papers)} research papers")
    return {"target_papers": validated}


def paper_validator_node(state: AgentState) -> dict:
    """
    Final validation step for research papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final target_papers list
    """
    print("\n--- PAPER VALIDATOR: Final validation ---")
    target_papers = state.get('target_papers', [])
    
    # Additional validation: ensure papers are research papers and not duplicates
    final_papers = []
    seen_titles = set()
    
    for paper in target_papers:
        title_key = paper.title.lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        
        # Quick quality check
        if paper.title and len(paper.title) > 10:  # Basic sanity check
            final_papers.append(paper)
    
    print(f"   [SUMMARY] Final validated papers: {len(final_papers)}")
    return {"target_papers": final_papers}


# ========== SUBSECTION WRITING NODES ==========

def taxonomy_parser_node(state: AgentState) -> dict:
    """
    Parse taxonomy JSON to extract structure with all subcategories.
    Waits for both Survey Track and Paper Track to complete.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with taxonomy_structure
    """
    taxonomy_json = state.get('taxonomy_json', '')
    target_papers = state.get('target_papers', [])
    
    # Check if both tracks have completed
    if not taxonomy_json or not taxonomy_json.strip():
        print("\n--- TAXONOMY PARSER: Waiting for Survey Track (taxonomy not ready) ---")
        return {}  # Return empty update to wait
    
    if not target_papers or len(target_papers) == 0:
        print("\n--- TAXONOMY PARSER: Waiting for Paper Track (papers not ready) ---")
        return {}  # Return empty update to wait
    
    print("\n--- TAXONOMY PARSER: Extracting taxonomy structure ---")
    print("   [READY] Both tracks completed, parsing taxonomy...")
    
    try:
        taxonomy_data = json.loads(taxonomy_json)
        # Store the parsed structure
        taxonomy_structure = {
            "main_categories": taxonomy_data.get('main_categories', [])
        }
        
        # Count subcategories
        total_subcategories = sum(
            len(cat.get('subcategories', [])) 
            for cat in taxonomy_structure['main_categories']
        )
        
        print(f"   [SUCCESS] Parsed {len(taxonomy_structure['main_categories'])} main categories with {total_subcategories} subcategories")
        return {"taxonomy_structure": taxonomy_structure}
    except json.JSONDecodeError as e:
        print(f"   [ERROR] Failed to parse taxonomy JSON: {e}")
        return {"taxonomy_structure": {}}


def paper_indexer_node(state: AgentState) -> dict:
    """
    Index all target papers into a vector store for RAG.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with paper_retriever and citation_map initialized
    """
    print("\n--- PAPER INDEXER: Building vector store from all papers ---")
    target_papers = state.get('target_papers', [])
    pdf_folder = state.get('pdf_folder', '')
    
    if not target_papers:
        print("   [WARNING] No target papers to index")
        return {"paper_retriever": None, "citation_map": {}}
    
    print(f"   -> Processing {len(target_papers)} papers...")
    
    # Process all papers into document chunks
    all_docs = process_papers(target_papers, pdf_folder)
    
    if not all_docs:
        print("   [WARNING] No documents extracted from papers")
        return {"paper_retriever": None, "citation_map": {}}
    
    print(f"   -> Created {len(all_docs)} document chunks")
    
    # Create vector store
    retriever = create_vectorstore(all_docs)
    
    # Initialize citation map (empty, will be populated as papers are cited)
    citation_map = {}
    
    print(f"   [SUCCESS] Indexed {len(target_papers)} papers into vector store")
    return {
        "paper_retriever": retriever,
        "citation_map": citation_map
    }


def subsection_writer_node(state: AgentState) -> dict:
    """
    Write a subsection for a specific subcategory using RAG to find relevant papers.
    This node processes one subcategory at a time.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with subsection content added to subsections dict
    """
    print("\n--- SUBSECTION WRITER: Writing subsection content ---")
    
    topic = state.get('topic', '')
    taxonomy_structure = state.get('taxonomy_structure', {})
    paper_retriever = state.get('paper_retriever')
    citation_map = state.get('citation_map', {})
    subsections = state.get('subsections', {})
    target_papers = state.get('target_papers', [])
    
    # Get current subcategory to process (stored in state)
    current_subcategory = state.get('current_subcategory', '')
    current_category = state.get('current_category', '')
    
    if not current_subcategory:
        print("   [WARNING] No current subcategory specified")
        return {}
    
    if not paper_retriever:
        print("   [WARNING] No paper retriever available")
        return {}
    
    print(f"   -> Writing subsection for: {current_subcategory}")
    
    # Use RAG to find relevant papers for this subcategory
    queries = [
        f"{current_subcategory} {topic}",
        f"{current_subcategory} approach method",
        f"{current_subcategory} technique",
        f"{current_subcategory} implementation",
        f"{current_subcategory} application"
    ]
    
    # Retrieve relevant chunks
    all_relevant_chunks = []
    seen_sources = set()
    
    for query in queries:
        try:
            retrieved = paper_retriever.invoke(query)
            for chunk in retrieved:
                source = chunk.metadata.get('source', '')
                if source and source not in seen_sources:
                    all_relevant_chunks.append(chunk)
                    seen_sources.add(source)
        except Exception as e:
            print(f"   [WARNING] Error retrieving for query '{query}': {e}")
            continue
    
    # Limit total chunks to avoid token overflow (keep within 8000 chars)
    combined_text = ""
    selected_chunks = []
    for chunk in all_relevant_chunks[:20]:  # Limit to 20 chunks
        if len(combined_text) + len(chunk.page_content) < 8000:
            combined_text += chunk.page_content + "\n\n"
            selected_chunks.append(chunk)
        else:
            break
    
    # Extract unique paper titles from chunks
    relevant_paper_titles = list(set(
        chunk.metadata.get('source', '') 
        for chunk in selected_chunks 
        if chunk.metadata.get('source', '')
    ))
    
    # Validation: Check if any relevant papers were found
    if not relevant_paper_titles or not selected_chunks:
        print(f"   [SKIP] No relevant papers found for subcategory '{current_subcategory}' - skipping subsection")
        return {}
    
    # Validation: Check if subcategory is appropriate (not metadata)
    metadata_keywords = ['year', 'publication year', 'author', 'journal', 'conference', 'venue', 
                         'affiliation', 'institution', 'geographic', 'region', 'country',
                         'publication type', 'paper type', 'first author', 'last author']
    subcategory_lower = current_subcategory.lower()
    if any(keyword in subcategory_lower for keyword in metadata_keywords):
        print(f"   [SKIP] Subcategory '{current_subcategory}' appears to be metadata - skipping subsection")
        return {}
    
    # Get or assign citations for relevant papers
    cited_papers = []
    citation_refs = []
    for paper_title in relevant_paper_titles:
        # Find the paper object
        paper_obj = next((p for p in target_papers if p.title == paper_title), None)
        if paper_obj:
            cit_num = get_or_assign_citation(citation_map, paper_title)
            cited_papers.append((paper_obj, cit_num))
            citation_refs.append(f"[{cit_num}]")
    
    # Generate subsection content
    papers_list = "\n".join([
        f"- {format_paper_title_with_year(paper)} {format_citation(cit_num)}"
        for paper, cit_num in cited_papers
    ]) if cited_papers else "No specific papers found."
    
    subsection_prompt = f"""Write a detailed subsection for a survey paper about "{topic}".

Category: {current_category}
Subcategory: {current_subcategory}

Relevant content from papers:
{combined_text[:6000] if combined_text else "No specific content found."}

Relevant papers that use this approach:
{papers_list}

Write a comprehensive subsection that:
1. Defines/explains what "{current_subcategory}" means in the context of {topic}
2. Describes how papers use this approach/method
3. Provides specific details from the relevant papers
4. Uses citations in the format [1], [2], etc. for each paper mentioned
5. Is written in academic style suitable for a survey paper

Format the output as markdown with a level 3 heading (###) for the subcategory name, followed by the content.

Return ONLY the markdown content, starting with the heading."""
    
    res = llm.invoke([HumanMessage(content=subsection_prompt)])
    subsection_content = res.content.strip()
    
    # Store subsection
    subsections[current_subcategory] = subsection_content
    
    print(f"   [SUCCESS] Wrote subsection for {current_subcategory} ({len(subsection_content)} chars, {len(cited_papers)} papers cited)")
    
    return {
        "subsections": subsections,
        "citation_map": citation_map
    }


def section_writer_node(state: AgentState) -> dict:
    """
    Write a section introduction for a main category.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with section content added to sections dict
    """
    print("\n--- SECTION WRITER: Writing section introduction ---")
    
    topic = state.get('topic', '')
    taxonomy_structure = state.get('taxonomy_structure', {})
    sections = state.get('sections', {})
    
    # Get current category to process
    current_category = state.get('current_category', '')
    
    if not current_category:
        print("   [WARNING] No current category specified")
        return {}
    
    # Find subcategories for this category
    subcategories = []
    for main_cat in taxonomy_structure.get('main_categories', []):
        if main_cat.get('name') == current_category:
            subcategories = main_cat.get('subcategories', [])
            break
    
    # Validation: Check if category is appropriate (not metadata)
    metadata_keywords = ['year', 'publication year', 'author', 'journal', 'conference', 'venue',
                         'affiliation', 'institution', 'geographic', 'region', 'country',
                         'publication type', 'paper type', 'first author', 'last author']
    category_lower = current_category.lower()
    if any(keyword in category_lower for keyword in metadata_keywords):
        print(f"   [SKIP] Category '{current_category}' appears to be metadata - skipping section")
        return {}
    
    # Validation: Check if there are written subsections for this category
    subsections = state.get('subsections', {})
    written_subsections = [subcat for subcat in subcategories if subcat in subsections]
    if not written_subsections:
        print(f"   [SKIP] No written subsections found for category '{current_category}' - skipping section")
        return {}
    
    print(f"   -> Writing section introduction for: {current_category}")
    
    # Only list subsections that were actually written
    subcategories_list = "\n".join([f"- {subcat}" for subcat in written_subsections])
    
    section_prompt = f"""Write a section introduction for a survey paper about "{topic}".

Category: {current_category}

Subcategories covered in this section:
{subcategories_list}

Write a comprehensive section introduction that:
1. Provides an overview of the {current_category} category
2. Explains its importance/relevance to {topic}
3. Mentions the subcategories that will be covered
4. Sets the context for the detailed subsections that follow
5. Is written in academic style suitable for a survey paper

Format the output as markdown with a level 2 heading (##) for the category name, followed by the introduction.

Return ONLY the markdown content, starting with the heading."""
    
    res = llm.invoke([HumanMessage(content=section_prompt)])
    section_content = res.content.strip()
    
    # Store section
    sections[current_category] = section_content
    
    print(f"   [SUCCESS] Wrote section introduction for {current_category} ({len(section_content)} chars)")
    
    return {"sections": sections}


def report_assembler_node(state: AgentState) -> dict:
    """
    Assemble all sections and subsections into a final markdown report.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final_report
    """
    print("\n--- REPORT ASSEMBLER: Combining all sections into final report ---")
    
    topic = state.get('topic', '')
    taxonomy_structure = state.get('taxonomy_structure', {})
    sections = state.get('sections', {})
    subsections = state.get('subsections', {})
    citation_map = state.get('citation_map', {})
    target_papers = state.get('target_papers', [])
    
    # Build report
    report_parts = []
    
    # Title
    report_parts.append(f"# Survey: {topic}\n\n")
    
    # Introduction (brief)
    report_parts.append("## Introduction\n\n")
    intro_prompt = f"""Write a brief introduction for a survey paper about "{topic}".

This introduction should:
1. Introduce the topic and its importance
2. Explain the scope of the survey
3. Provide an overview of the taxonomy structure
4. Be concise (2-3 paragraphs)

Return ONLY the introduction text, no heading."""
    
    intro_res = llm.invoke([HumanMessage(content=intro_prompt)])
    report_parts.append(intro_res.content.strip())
    report_parts.append("\n\n")
    
    # Add sections and subsections in order
    for main_cat in taxonomy_structure.get('main_categories', []):
        cat_name = main_cat.get('name', '')
        if not cat_name:
            continue
        
        # Add section introduction
        if cat_name in sections:
            report_parts.append(sections[cat_name])
            report_parts.append("\n\n")
        
        # Add subsections
        for subcat in main_cat.get('subcategories', []):
            if subcat in subsections:
                report_parts.append(subsections[subcat])
                report_parts.append("\n\n")
    
    # Add references section
    report_parts.append("## References\n\n")
    
    # Sort papers by citation number
    cited_papers = [
        (title, cit_num) 
        for title, cit_num in citation_map.items()
    ]
    cited_papers.sort(key=lambda x: x[1])
    
    for title, cit_num in cited_papers:
        paper = next((p for p in target_papers if p.title == title), None)
        if paper:
            year = extract_year_from_date(paper.published_date)
            if paper.doi:
                report_parts.append(f"[{cit_num}] {paper.title} ({year}). DOI: {paper.doi}\n\n")
            elif paper.url:
                report_parts.append(f"[{cit_num}] {paper.title} ({year}). URL: {paper.url}\n\n")
            else:
                report_parts.append(f"[{cit_num}] {paper.title} ({year})\n\n")
    
    final_report = "".join(report_parts)
    
    print(f"   [SUCCESS] Assembled final report ({len(final_report)} characters, {len(cited_papers)} references)")
    
    return {"final_report": final_report}


def subsection_coordinator_node(state: AgentState) -> dict:
    """
    Coordinate the writing of all sections and subsections.
    Processes all categories and subcategories sequentially.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state indicating completion
    """
    print("\n--- SUBSECTION COORDINATOR: Processing all categories and subcategories ---")
    
    taxonomy_structure = state.get('taxonomy_structure', {})
    sections = state.get('sections', {})
    subsections = state.get('subsections', {})
    
    # Process each main category
    for main_cat in taxonomy_structure.get('main_categories', []):
        cat_name = main_cat.get('name', '')
        if not cat_name:
            continue
        
        print(f"\n   -> Processing category: {cat_name}")
        
        # Process each subcategory first
        valid_subsections_count = 0
        for subcat in main_cat.get('subcategories', []):
            if not subcat:
                continue
            
            print(f"      -> Processing subcategory: {subcat}")
            
            # Write subsection if not already written
            if subcat not in subsections:
                # Update state with current subcategory
                state['current_category'] = cat_name
                state['current_subcategory'] = subcat
                subsection_result = subsection_writer_node(state)
                # Check if subsection was actually written (not skipped)
                if subsection_result and subcat in subsection_result.get('subsections', {}):
                    subsections.update(subsection_result.get('subsections', {}))
                    state.update(subsection_result)
                    valid_subsections_count += 1
            else:
                # Subsection already exists (was written previously)
                valid_subsections_count += 1
        
        # Only write section introduction if we have valid subsections
        if valid_subsections_count > 0 and cat_name not in sections:
            # Update state with current category
            state['current_category'] = cat_name
            section_result = section_writer_node(state)
            # Check if section was actually written (not skipped)
            if section_result and cat_name in section_result.get('sections', {}):
                sections.update(section_result.get('sections', {}))
                state.update(section_result)
    
    print(f"\n   [SUCCESS] Processed {len(sections)} sections and {len(subsections)} subsections")
    
    return {
        "sections": sections,
        "subsections": subsections
    }


def filter_node(state: AgentState) -> dict:
    """
    Filter papers based on relevance to topic and exclude survey papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with filtered papers
    """
    print("\n--- FILTER: Screening papers ---")
    filtered = []
    
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
            print(f"   [SKIP - SURVEY] {format_paper_title_with_year(paper)}")
            continue
        
        # Second check: Is it relevant to the topic?
        relevance_prompt = f"""Topic: {state['topic']}

Paper Title: {paper.title}
Paper Abstract: {paper.summary}

Evaluate if this paper is relevant to the topic and is a research paper (not a survey/review).

Respond with ONLY 'YES' if the paper should be included, or 'NO' if it should be excluded."""
        
        res = llm.invoke([HumanMessage(content=relevance_prompt)])
        if "YES" in res.content.upper():
            paper.status = "kept"
            filtered.append(paper)
            print(f"   [KEPT] {format_paper_title_with_year(paper)}")
        else:
            print(f"   [SKIP] {format_paper_title_with_year(paper)}")
    
    return {"papers": filtered}


def wos_pause_node(state: AgentState) -> dict:
    """
    Pause after filtering WOS papers to allow user to download PDFs.
    Shows paper details and waits for user confirmation.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state (no changes, just pauses)
    """
    sources = state.get('search_sources', [])
    papers = state['papers']
    pdf_folder = state.get('pdf_folder', '')
    
    # Only pause if WOS is in sources and there are WOS papers
    wos_papers = [p for p in papers if p.source == "wos"]
    
    if 'wos' in sources and wos_papers:
        print("\n" + "="*80)
        print("--- WOS PAPERS SELECTED - PDF DOWNLOAD REQUIRED ---")
        print("="*80)
        print(f"\nFound {len(wos_papers)} Web of Science papers after filtering.")
        print(f"Please download the PDF files and place them in: {pdf_folder}")
        print("\nSelected Papers:")
        print("-"*80)
        
        for i, paper in enumerate(wos_papers, 1):
            print(f"\n[{i}] {format_paper_title_with_year(paper)}")
            if paper.doi:
                print(f"    DOI: {paper.doi}")
            if paper.published_date:
                print(f"    Published: {paper.published_date}")
            if paper.summary and paper.summary != "No abstract available via WOS API":
                print(f"    Abstract: {paper.summary[:200]}...")
            print()
        
        print("-"*80)
        print(f"\nInstructions:")
        print(f"1. Download the PDF files for the papers listed above")
        print(f"2. Save them in the folder: {pdf_folder}")
        print(f"3. Name the PDF files with keywords from the paper title for automatic matching")
        print(f"4. Press Enter when you have finished downloading and placing the PDFs")
        print("="*80)
        
        input("\nPress Enter when you are done downloading PDFs...")
        print("\nContinuing with analysis...\n")
    
    return {}


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
    
    # 1. Download & Chunk (handles both ArXiv downloads and local PDFs)
    pdf_folder = state.get('pdf_folder', '')
    docs = process_papers(papers, pdf_folder)
    
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


def target_paper_termination_router(state: AgentState) -> str:
    """
    Route based on termination check. If terminated, go to END, otherwise continue.
    
    Args:
        state: Current agent state
        
    Returns:
        END if terminated, "sorter" if continuing
    """
    workflow_terminated = state.get('workflow_terminated', False)
    
    if workflow_terminated:
        return END
    else:
        return "taxonomy_parser"



