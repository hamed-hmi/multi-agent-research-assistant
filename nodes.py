"""Graph nodes for the Survey Agent workflow."""
import os
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END

from models import AgentState, SearchQueries, DualSearchQueries, ReviewDecision, SelectionQualityScore
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
    
    Args:
        state: Current agent state
        
    Returns:
        Empty dict (pass-through)
    """
    print("\n--- INITIAL SPLIT: Starting parallel query generation paths ---")
    return {}


def survey_query_planner_node(state: AgentState) -> dict:
    """
    Generate search queries for finding survey/review papers.
    Can accept feedback to broaden queries on retry.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_queries
    """
    print("\n--- SURVEY QUERY PLANNER: Generating survey queries ---")
    topic = state.get('topic', '')
    feedback = state.get('survey_query_feedback', '')
    previous_queries = state.get('survey_queries', [])
    is_retry = bool(feedback)
    
    if is_retry:
        print("   [RETRY] Regenerating queries with feedback")
        previous_queries_str = "\n".join([f"- {q}" for q in previous_queries])
        system_content = f"""You are an expert at generating search queries for academic databases to find survey/review papers.

Previous queries that didn't return enough results:
{previous_queries_str}

Feedback: {feedback}

CRITICAL REQUIREMENTS:
1. Generate NEW, BROADER search queries that address the feedback
2. Ensure ALL aspects of the topic "{topic}" are covered across the queries
3. If the feedback mentions missing aspects, make sure at least one query specifically targets those missing aspects
4. BUT also ensure other queries maintain coverage of the other aspects of the topic
5. The set of queries together must provide COMPLETE coverage of the entire topic
6. Make sure the queries are different from the previous ones"""
    else:
        system_content = """You are an expert at generating search queries for academic databases to find survey/review papers.
Generate queries that will find comprehensive survey and review papers on the given topic."""
    
    prompt = f"""Generate 3 distinct search queries to find survey/review papers on: "{topic}"

Each query should:
- Target survey, review, or literature review papers
- Use appropriate boolean operators (AND, OR, NOT)
- Include variations of key terms
- Be specific enough to find relevant papers but broad enough to capture related work

Example format for ArXiv:
- (federated learning OR distributed learning) AND (channel estimation OR CSI estimation) AND (survey OR review OR literature review)
- (machine learning OR deep learning) AND (wireless communication) AND (comprehensive review OR state-of-the-art)
- (distributed systems) AND (your topic keywords) AND (survey OR systematic review)

Respond with ONLY the queries, one per line."""
    
    structured_llm = llm.with_structured_output(SearchQueries)
    try:
        result = structured_llm.invoke([SystemMessage(content=system_content), HumanMessage(content=prompt)])
        queries = result.queries[:3]  # Limit to 3 queries
        
        print(f"   [GENERATED] {len(queries)} survey queries:")
        for i, q in enumerate(queries, 1):
            print(f"   [{i}] {q}")
        
        return {
            "survey_queries": queries,
            "survey_query_feedback": ""  # Clear feedback after use
        }
    except Exception as e:
        print(f"   [ERROR] Failed to generate queries: {e}")
        # Fallback queries
        fallback = [
            f'("{topic}" AND (survey OR review OR "literature review"))',
            f'("{topic}" AND (comprehensive OR systematic OR "state-of-the-art"))'
        ]
        return {"survey_queries": fallback}


def research_query_planner_node(state: AgentState) -> dict:
    """
    Generate search queries for finding research papers (non-survey).
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with research_queries
    """
    print("\n--- RESEARCH QUERY PLANNER: Generating research queries ---")
    topic = state.get('topic', '')
    
    prompt = f"""Generate 3 distinct search queries to find research papers (NOT surveys/reviews) on: "{topic}"

Each query should:
- Exclude survey, review, or literature review papers
- Use appropriate boolean operators (AND, OR, NOT)
- Target original research contributions
- Be specific to the topic

Example format for ArXiv:
- (channel estimation OR CSI estimation) AND (federated learning OR distributed learning) ANDNOT (survey OR review OR "literature review")
- (your topic keywords) ANDNOT (survey OR review OR "systematic review")

Respond with ONLY the queries, one per line."""
    
    structured_llm = llm.with_structured_output(SearchQueries)
    try:
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        queries = result.queries[:3]
        
        print(f"   [GENERATED] {len(queries)} research queries:")
        for i, q in enumerate(queries, 1):
            print(f"   [{i}] {q}")
        
        return {"research_queries": queries}
    except Exception as e:
        print(f"   [ERROR] Failed to generate queries: {e}")
        fallback = [f'"{topic}" ANDNOT (survey OR review)']
        return {"research_queries": fallback}


def survey_search_node(state: AgentState) -> dict:
    """
    Search for survey papers using generated queries.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_papers
    """
    print("\n--- SURVEY SEARCHER: Fetching survey paper metadata ---")
    queries = state.get('survey_queries', [])
    search_sources = state.get('search_sources', ['arxiv'])
    
    all_papers = []
    seen_titles = set()
    
    for query in queries:
        print(f"   -> Searching for: {query}")
        
        for source in search_sources:
            if source == 'arxiv':
                papers = search_arxiv(query, max_results=10)
            elif source == 'wos':
                # TODO: Add WOS API key from config
                papers = search_wos(query, api_key="", max_results=10)
            else:
                continue
            
            for paper in papers:
                if paper.title.lower() not in seen_titles:
                    seen_titles.add(paper.title.lower())
                    all_papers.append(paper)
    
    print(f"   [SUMMARY] Found {len(all_papers)} unique survey papers")
    return {"survey_papers": all_papers}


def research_search_node(state: AgentState) -> dict:
    """
    Search for research papers using generated queries.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with papers (legacy field) and target_papers
    """
    print("\n--- RESEARCH SEARCHER: Fetching research paper metadata ---")
    queries = state.get('research_queries', [])
    search_sources = state.get('search_sources', ['arxiv'])
    
    all_papers = []
    seen_titles = set()
    
    for query in queries:
        print(f"   -> Searching for: {query}")
        
        for source in search_sources:
            if source == 'arxiv':
                papers = search_arxiv(query, max_results=20)
            elif source == 'wos':
                papers = search_wos(query, api_key="", max_results=20)
            else:
                continue
            
            for paper in papers:
                if paper.title.lower() not in seen_titles:
                    seen_titles.add(paper.title.lower())
                    all_papers.append(paper)
    
    print(f"   [SUMMARY] Found {len(all_papers)} unique research papers")
    return {"papers": all_papers, "target_papers": all_papers}


def survey_feedback_check_node(state: AgentState) -> dict:
    """
    Check if enough survey papers were found after initial search.
    If less than minimum, provide feedback to retry with broader queries.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_query_feedback if needed, or empty dict to continue
    """
    print("\n--- SURVEY FEEDBACK CHECK: Evaluating survey paper count ---")
    survey_papers = state.get('survey_papers', [])
    min_survey_papers = state.get('min_survey_papers', 5)
    survey_search_retry_count = state.get('survey_search_retry_count', 0)
    max_retries = state.get('max_survey_query_retries', 2)
    
    print(f"   [INFO] Found {len(survey_papers)} survey papers (minimum needed: {min_survey_papers})")
    print(f"   [INFO] Retry count: {survey_search_retry_count}/{max_retries}")
    
    if len(survey_papers) >= min_survey_papers:
        print("   [SUCCESS] Enough survey papers found, continuing...")
        return {}
    
    if survey_search_retry_count >= max_retries:
        print(f"   [WARNING] Max retries reached ({max_retries}). Continuing with {len(survey_papers)} survey papers.")
        return {}
    
    feedback = f"""Previous queries returned only {len(survey_papers)} survey/review papers, but we need at least {min_survey_papers}. Please broaden the search scope - include related domains, use more general terms, and consider parent domains that might contain relevant surveys."""
    
    print(f"   [FEEDBACK] Setting feedback for query refinement")
    print(f"   [ACTION] Will retry with broader queries (attempt {survey_search_retry_count + 1})")
    
    return {
        "survey_query_feedback": feedback,
        "survey_search_retry_count": survey_search_retry_count + 1
    }


def survey_feedback_router(state: AgentState) -> str:
    """
    Route based on survey feedback check.
    
    Args:
        state: Current agent state
        
    Returns:
        "survey_query_planner" if retry needed, "survey_selector" if enough papers
    """
    survey_query_feedback = state.get('survey_query_feedback', '')
    
    if survey_query_feedback:
        return "survey_query_planner"
    else:
        return "survey_selector"


def survey_validation_feedback_check_node(state: AgentState) -> dict:
    """
    Check if enough survey papers were found after selection and grade their quality.
    Two-stage feedback mechanism:
    1. If quality below threshold and selector retries < max: provide feedback to selector
    2. If selector retries >= max or count insufficient: provide feedback to search
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_selector_feedback or survey_query_feedback if needed, or empty dict to continue
    """
    print("\n--- SURVEY VALIDATION FEEDBACK CHECK: Evaluating selected surveys ---")
    validated_surveys = state.get('survey_papers', [])
    topic = state.get('topic', '')
    min_validated_surveys = state.get('min_validated_surveys', 3)
    quality_threshold = state.get('survey_selection_quality_threshold', 0.7)
    survey_selector_retry_count = state.get('survey_selector_retry_count', 0)
    survey_validation_retry_count = state.get('survey_validation_retry_count', 0)
    max_retries = state.get('max_survey_query_retries', 2)
    
    print(f"   [INFO] Selected and validated {len(validated_surveys)} survey papers (minimum needed: {min_validated_surveys})")
    print(f"   [INFO] Selector retry count: {survey_selector_retry_count}/{max_retries}")
    print(f"   [INFO] Validation retry count: {survey_validation_retry_count}/{max_retries}")
    
    # First check: count
    if len(validated_surveys) < min_validated_surveys:
        print(f"   [ISSUE] Insufficient number of validated surveys ({len(validated_surveys)} < {min_validated_surveys})")
        if survey_validation_retry_count >= max_retries:
            print(f"   [WARNING] Max retries reached ({max_retries}). Continuing with {len(validated_surveys)} validated survey papers.")
            return {}
        
        feedback = f"""Previous queries returned only {len(validated_surveys)} valid survey/review papers, but we need at least {min_validated_surveys}. The selected papers must be: (1) actually survey/review papers, (2) relevant to the topic, and (3) of sufficient quality. Please broaden the search scope - include related domains, use more general terms, and consider parent domains that might contain relevant surveys."""
        
        print(f"   [FEEDBACK] Setting feedback for query refinement")
        print(f"   [ACTION] Will retry search with broader queries (attempt {survey_validation_retry_count + 1})")
        
        return {
            "survey_query_feedback": feedback,
            "survey_validation_retry_count": survey_validation_retry_count + 1,
            "survey_papers": []
        }
    
    # Second check: quality (only if we have enough papers)
    print(f"   [INFO] Grading selection quality (threshold: {quality_threshold})...")
    
    papers_info = []
    for paper in validated_surveys:
        year = extract_year_from_date(paper.published_date)
        papers_info.append({
            "title": paper.title,
            "abstract": paper.summary[:500] if paper.summary else "No abstract",
            "year": year,
            "source": paper.source
        })
    
    papers_list = "\n\n".join([
        f"Title: {p['title']}\n"
        f"    Abstract: {p['abstract']}\n"
        f"    Published: {p['year']} ({p['source']})"
        for p in papers_info
    ])
    
    quality_prompt = f"""Evaluate the quality of this selection of 3 survey/review papers for writing a new survey paper on the topic: "{topic}"

Selected Papers:
{papers_list}

Evaluate based on the following simplified rubric (score each criterion 0.0-1.0, then calculate weighted average):
1. Relevance (weight: 0.30): Are all selected papers relevant to the topic? Score 0.0-1.0.
2. Coverage (weight: 0.60): Do the papers, together, cover the main aspects of the topic? Score 0.0-1.0.))
3. Recency (weight: 0.10): Are the papers recent enough for the topic (e.g., published within the last few years if the field is fast-moving)? Score 0.0-1.0.

Calculate the final score as: (Relevance × 0.30) + (Coverage × 0.60) + (Recency × 0.10)

Consider if a better combination could exist from the available papers."""
     
    structured_llm = llm.with_structured_output(SelectionQualityScore)
    try:
        quality_score = structured_llm.invoke([HumanMessage(content=quality_prompt)])
        
        print(f"   [QUALITY SCORE] {quality_score.score:.2f}/1.0 (threshold: {quality_threshold})")
        print(f"   [REASONING] {quality_score.reasoning[:200]}...")
        
        if quality_score.score >= quality_threshold:
            print("   [SUCCESS] Selection quality meets threshold, continuing...")
            return {}
        
        # Quality below threshold - check if we should retry selector or search
        if survey_selector_retry_count < max_retries:
            print(f"   [ISSUE] Selection quality below threshold ({quality_score.score:.2f} < {quality_threshold})")
            print(f"   [FEEDBACK] Providing constructive feedback to selector")
            print(f"   [ACTION] Will retry selection (attempt {survey_selector_retry_count + 1})")
            
            feedback_parts = [f"Previous selection scored {quality_score.score:.2f}/1.0, below the threshold of {quality_threshold}."]
            
            if quality_score.strengths:
                feedback_parts.append(f"\nStrengths: {', '.join(quality_score.strengths)}")
            
            if quality_score.weaknesses:
                feedback_parts.append(f"\nAreas for improvement: {', '.join(quality_score.weaknesses)}")
                feedback_parts.append("\nPlease select a different combination of 3 papers that addresses these weaknesses while maintaining or improving the strengths.")
            
            feedback_parts.append(f"\nDetailed reasoning: {quality_score.reasoning}")
            
            selector_feedback = "".join(feedback_parts)
            
            return {
                "survey_selector_feedback": selector_feedback,
                "survey_selector_retry_count": survey_selector_retry_count + 1,
            }
        else:
            print(f"   [WARNING] Selector retries exhausted ({max_retries}). Falling back to re-searching.")
            print(f"   [ACTION] Will retry search with broader queries")
            
            feedback_parts = [
                f"Previous selection of survey papers scored {quality_score.score:.2f}/1.0, below the quality threshold of {quality_threshold}.",
                f"After {max_retries} attempts to improve selection, we need to broaden the search to find better survey papers."
            ]
            
            # Include specific weaknesses so query planner knows what aspects are missing
            if quality_score.weaknesses:
                feedback_parts.append(f"\nSpecific issues with previous selection:")
                for weakness in quality_score.weaknesses:
                    feedback_parts.append(f"- {weakness}")
                feedback_parts.append(f"\nIMPORTANT: The new queries must ensure coverage of ALL aspects of the topic '{topic}', including both:")
                feedback_parts.append(f"1. The aspects that were missing or poorly covered (mentioned above)")
                feedback_parts.append(f"2. The aspects that were already covered (to maintain complete topic coverage)")
                feedback_parts.append(f"\nDo NOT generate queries that only focus on the missing aspects - you must maintain comprehensive coverage of the entire topic.")
            
            feedback_parts.append("\nPlease broaden the search scope - include related domains, use more general terms, and consider parent domains that might contain higher-quality relevant surveys.")
            
            feedback = "".join(feedback_parts)
            
            return {
                "survey_query_feedback": feedback,
                "survey_validation_retry_count": survey_validation_retry_count + 1,
                "survey_papers": []
            }
            
    except Exception as e:
        print(f"   [WARNING] Error grading selection quality: {e}")
        print("   [FALLBACK] Continuing with current selection")
        return {}


def survey_validation_feedback_router(state: AgentState) -> str:
    """
    Route based on survey validation feedback check (after selector).
    Two-stage routing:
    1. If selector feedback exists: route back to selector to improve selection
    2. If query feedback exists: route to query planner to re-search
    3. Otherwise: continue to taxonomy extraction
    
    Args:
        state: Current agent state
        
    Returns:
        "survey_selector" if selector feedback exists,
        "survey_query_planner" if query feedback exists,
        "taxonomy_extractor" if enough validated surveys
    """
    survey_selector_feedback = state.get('survey_selector_feedback', '')
    survey_query_feedback = state.get('survey_query_feedback', '')
    
    if survey_selector_feedback:
        return "survey_selector"
    elif survey_query_feedback:
        return "survey_query_planner"
    else:
        return "taxonomy_extractor"


def survey_selector_node(state: AgentState) -> dict:
    """
    Select and validate the top 3 survey papers based on:
    1. Usefulness for writing a survey paper on the topic
    2. Ability to help find/extract taxonomy
    3. Recency (more recent papers are preferred)
    4. Complementarity (papers should work together)
    
    Can accept feedback to improve selection quality without re-searching.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with top 3 selected and validated survey_papers
    """
    print("\n--- SURVEY SELECTOR: Selecting and validating top 3 surveys ---")
    survey_papers = state.get('survey_papers', [])
    topic = state.get('topic', '')
    selector_feedback = state.get('survey_selector_feedback', '')
    is_retry = bool(selector_feedback)
    
    if not survey_papers:
        print("   [WARNING] No survey papers to select from")
        return {"survey_papers": []}
    
    if is_retry:
        print("   [RETRY] Improving selection based on quality feedback")
    
    print(f"   [INFO] Selecting top 3 from {len(survey_papers)} survey papers")
    
    papers_info = []
    for i, paper in enumerate(survey_papers):
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

"""
    
    if is_retry:
        selection_prompt += f"""
IMPORTANT: You are re-selecting because the previous selection did not meet quality standards.

Previous Feedback:
{selector_feedback}

Please select a DIFFERENT combination of 3 papers that addresses the feedback above. Consider all available papers and choose the best combination that:
- Addresses the weaknesses mentioned in the feedback
- Maintains or improves upon the strengths
- Provides better complementarity and coverage

"""
    
    selection_prompt += f"""
Topic: {topic}

IMPORTANT: The topic could contain multiple key aspects that MUST be covered by the selected papers.
You MUST ensure that ALL key aspects of the topic are covered across the 3 selected papers.

Survey Papers:
{papers_list}


Think about:
- Which 3 papers together would give the best overall coverage?
- Which combination avoids redundancy and maximizes diversity?
- Which set of taxonomies would be most useful to learn from and combine?
MANDATORY: All key aspects of the topic must be covered across the 3 papers

Respond with ONLY the indices (numbers) of the 3 selected papers (if exists and all valid), separated by commas.
Example: "0, 3, 7"
"""
    
    res = llm.invoke([HumanMessage(content=selection_prompt)])
    selected_indices_str = res.content.strip()
    
    try:
        indices = [int(x.strip()) for x in re.findall(r'\d+', selected_indices_str)]
        selected_indices = []
        for idx in indices:
            if idx not in selected_indices and 0 <= idx < len(survey_papers):
                selected_indices.append(idx)
                if len(selected_indices) >= 3:
                    break
        
        if len(selected_indices) < 3:
            for i in range(len(survey_papers)):
                if i not in selected_indices:
                    selected_indices.append(i)
                    if len(selected_indices) >= 3:
                        break
        
        selected_papers = [survey_papers[i] for i in selected_indices[:3]]
        
        if selected_papers:
            print(f"\n   [SELECTED] Top {len(selected_papers)} surveys:")
            for i, paper in enumerate(selected_papers, 1):
                print(f"   [{i}] {format_paper_title_with_year(paper)}")
        
        return {
            "survey_papers": selected_papers,
            "survey_selector_feedback": ""
        }
        
    except Exception as e:
        print(f"   [ERROR] Failed to parse selection: {e}")
        print(f"   [FALLBACK] Using first 3 papers")
        return {"survey_papers": survey_papers[:3]}


def relevance_judge_node(state: AgentState) -> dict:
    """
    Validate research papers for relevance and usefulness using structured output.
    Merged functionality of relevance_judge and paper_validator nodes.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with validated target_papers
    """
    print("\n--- PAPER VALIDATOR: Evaluating research papers ---")
    papers = state.get('papers', [])
    topic = state.get('topic', '')
    
    if not papers:
        print("   [WARNING] No papers to validate")
        return {"target_papers": []}
    
    validated = []
    structured_llm = llm.with_structured_output(ReviewDecision)
    
    for paper in papers:
        # Build list of already validated papers for duplicate checking
        already_selected = ""
        if validated:
            already_selected = "\n\nAlready selected papers:\n"
            for v in validated:
                already_selected += f"- {v.title}\n"
        
        prompt = f"""Evaluate if this paper is relevant and useful for a survey on: "{topic}"

Title: {paper.title}
Abstract: {paper.summary[:500] if paper.summary else "No abstract"}
{already_selected}
A paper should PASS if it:
- Is directly relevant to the topic
- Is not a duplicate or very similar to other papers already selected

Decision: PASS if relevant and useful, FAIL if not."""
        
        try:
            decision = structured_llm.invoke([HumanMessage(content=prompt)])
            if decision.decision == "PASS":
                validated.append(paper)
                print(f"   [KEPT] {format_paper_title_with_year(paper)}")
            else:
                print(f"   [SKIP] {format_paper_title_with_year(paper)}")
                if decision.feedback:
                    print(f"      Reason: {decision.feedback}")
        except Exception as e:
            print(f"   [WARNING] Validation error for {paper.title}: {e}")
            # Default to keeping on error
            validated.append(paper)
    
    print(f"   [SUMMARY] Validated {len(validated)} out of {len(papers)} research papers")
    return {"target_papers": validated}


def target_paper_termination_check_node(state: AgentState) -> dict:
    """
    Check if enough target papers were found. Terminate if less than 2.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with termination flag if needed
    """
    print("\n--- TARGET PAPER TERMINATION CHECK: Evaluating target paper count ---")
    target_papers = state.get('target_papers', [])
    min_target_papers = 2
    
    print(f"   [INFO] Found {len(target_papers)} target papers (minimum needed: {min_target_papers})")
    
    if len(target_papers) >= min_target_papers:
        print("   [SUCCESS] Enough target papers found, continuing...")
        return {}
    
    topic = state.get('topic', '')
    message = f"""Insufficient papers found for the topic: "{topic}"

Only {len(target_papers)} relevant papers were found, but at least {min_target_papers} are needed.

Suggestions:
1. Try a broader or more general topic
2. Consider related domains or parent topics
3. Adjust search terms to be less specific

Please try again with a different topic."""
    
    print(f"   [TERMINATION] Not enough papers found")
    return {
        "termination_message": message,
        "workflow_terminated": True
    }


def target_paper_termination_router(state: AgentState) -> str:
    """
    Route based on target paper termination check.
    
    Args:
        state: Current agent state
        
    Returns:
        "END" if terminated, "taxonomy_parser" if continuing
    """
    if state.get('workflow_terminated', False):
        return "END"
    else:
        return "taxonomy_parser"


def taxonomy_extractor_node(state: AgentState) -> dict:
    """
    Extract taxonomy structures from validated survey papers using RAG.
    Uses semantic search to find taxonomy-related content in papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with extracted_taxonomies
    """
    print("\n--- TAXONOMY EXTRACTOR: Extracting taxonomies using RAG ---")
    survey_papers = state.get('survey_papers', [])
    topic = state.get('topic', '')
    
    if not survey_papers:
        print("   [WARNING] No survey papers to extract from")
        return {"extracted_taxonomies": []}
    
    extracted = []
    
    for paper in survey_papers:
        print(f"   [PROCESSING] {format_paper_title_with_year(paper)}")
        
        # Download and process PDF if URL available
        try:
            if paper.url and 'arxiv.org' in paper.url:
                import arxiv
                import tempfile
                import requests
                import fitz
                
                paper_id = paper.url.split('/')[-1]
                search = arxiv.Search(id_list=[paper_id])
                result = next(search.results(), None)
                if result:
                    # Download PDF
                    pdf_url = result.pdf_url
                    response = requests.get(pdf_url)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(response.content)
                        tmp_path = tmp.name
                    
                    # Extract text
                    doc = fitz.open(tmp_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                    os.unlink(tmp_path)
                    
                    # Create vector store for this paper
                    from langchain_core.documents import Document
                    chunks = [Document(page_content=full_text[i:i+1000]) 
                             for i in range(0, len(full_text), 1000)]
                    
                    if chunks:
                        retriever = create_vectorstore(chunks)
                        
                        # Search for taxonomy-related content
                        taxonomy_query = "What is the taxonomy, classification, categorization, or organization structure used to categorize studies in this survey paper?"
                        # Support both old and new LangChain APIs
                        if hasattr(retriever, 'get_relevant_documents'):
                            relevant_docs = retriever.get_relevant_documents(taxonomy_query)
                        else:
                            relevant_docs = retriever.invoke(taxonomy_query)
                        
                        taxonomy_text = "\n".join([doc.page_content for doc in relevant_docs[:5]])
                        
                        # Extract taxonomy using LLM
                        extract_prompt = f"""Extract the taxonomy, classification, or categorization structure from this survey paper.

Paper Title: {paper.title}
Topic: {topic}

Relevant Content:
{taxonomy_text}

Provide a clear, structured taxonomy used to categorize studies in this survey paper."""
                        
                        res = llm.invoke([HumanMessage(content=extract_prompt)])
                        taxonomy = res.content.strip()
                        
                        extracted.append({
                            "source_paper": paper.title,
                            "taxonomy": taxonomy
                        })
                        print(f"   [EXTRACTED] Taxonomy from: {format_paper_title_with_year(paper)}")
        except Exception as e:
            print(f"   [WARNING] Failed to extract taxonomy from {paper.title}: {e}")
            # Fallback: use abstract
            extract_prompt = f"""Extract any taxonomy or categorization from this paper's abstract.

Paper: {paper.title}
Abstract: {paper.summary[:1000] if paper.summary else "No abstract"}

Provide the taxonomy structure if present."""
            res = llm.invoke([HumanMessage(content=extract_prompt)])
            extracted.append({
                "source_paper": paper.title,
                "taxonomy": res.content.strip()
            })
    
    return {"extracted_taxonomies": extracted}


def taxonomy_designer_node(state: AgentState) -> dict:
    """
    Design a unified taxonomy based on extracted taxonomies.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with taxonomy_json
    """
    print("\n--- TAXONOMY DESIGNER: Creating unified taxonomy ---")
    extracted_taxonomies = state.get('extracted_taxonomies', [])
    topic = state.get('topic', '')
    
    if not extracted_taxonomies:
        print("   [WARNING] No taxonomies extracted")
        return {"taxonomy_json": "{}"}
    
    taxonomies_text = "\n\n".join([
        f"From {t['source_paper']}:\n{t['taxonomy']}"
        for t in extracted_taxonomies
    ])
    
    prompt = f"""Design a comprehensive categorization structure for a survey paper on: "{topic}"

Here are some related surveys and their categorization structre for your reference:
{taxonomies_text}

Create a unified categorization structure in JSON format with this structure:
{{
  "main_categories": [
    {{
      "name": "Category Name",
      "subcategories": ["Subcategory 1", "Subcategory 2", ...]
    }},
    ...
  ]
}}

Make sure the taxonomy:
- Covers all important aspects of the topic
- Is well-organized and hierarchical
- Avoids redundancy
- Is comprehensive but not overly detailed

Respond with ONLY valid JSON, no additional text."""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    taxonomy_json = res.content.strip()
    
    # Clean JSON (remove markdown code blocks if present)
    taxonomy_json = re.sub(r'```json\s*', '', taxonomy_json)
    taxonomy_json = re.sub(r'```\s*', '', taxonomy_json)
    taxonomy_json = taxonomy_json.strip()
    
    print(f"   [SUCCESS] Taxonomy designed ({len(taxonomy_json)} characters)")
    return {"taxonomy_json": taxonomy_json}


def taxonomy_parser_node(state: AgentState) -> dict:
    """
    Parse taxonomy JSON and extract structure. Waits for both taxonomy and target papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with taxonomy_structure
    """
    print("\n--- TAXONOMY PARSER: Extracting taxonomy structure ---")
    taxonomy_json = state.get('taxonomy_json', '')
    target_papers = state.get('target_papers', [])
    
    if not taxonomy_json or not target_papers:
        print("   [WAITING] Taxonomy or target papers not ready yet")
        return {}
    
    print("   [READY] Both tracks completed, parsing taxonomy...")
    
    try:
        taxonomy_data = json.loads(taxonomy_json)
        taxonomy_structure = {
            "main_categories": taxonomy_data.get("main_categories", [])
        }
        
        total_subcategories = sum(len(cat.get("subcategories", [])) for cat in taxonomy_structure["main_categories"])
        print(f"   [SUCCESS] Parsed {len(taxonomy_structure['main_categories'])} main categories with {total_subcategories} subcategories")
        
        return {"taxonomy_structure": taxonomy_structure}
    except Exception as e:
        print(f"   [ERROR] Failed to parse taxonomy JSON: {e}")
        return {"taxonomy_structure": {"main_categories": []}}


def paper_indexer_node(state: AgentState) -> dict:
    """
    Build vector store from all target papers and initialize citation map.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with paper_retriever and citation_map
    """
    print("\n--- PAPER INDEXER: Building vector store from all papers ---")
    target_papers = state.get('target_papers', [])
    
    if not target_papers:
        print("   [WARNING] No target papers to index")
        return {"paper_retriever": None, "citation_map": {}}
    
    print(f"   -> Processing {len(target_papers)} papers...")
    
    documents = []
    for paper in target_papers:
        # Download and process PDF if available
        try:
            if paper.url and 'arxiv.org' in paper.url:
                import arxiv
                import tempfile
                import requests
                import fitz
                
                paper_id = paper.url.split('/')[-1]
                search = arxiv.Search(id_list=[paper_id])
                result = next(search.results(), None)
                if result:
                    pdf_url = result.pdf_url
                    response = requests.get(pdf_url)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(response.content)
                        tmp_path = tmp.name
                    
                    doc = fitz.open(tmp_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                    os.unlink(tmp_path)
                    
                    # Create chunks
                    from langchain_core.documents import Document
                    chunk_size = 1000
                    for i in range(0, len(full_text), chunk_size):
                        chunk = full_text[i:i+chunk_size]
                        if chunk.strip():
                            documents.append(Document(
                                page_content=chunk,
                                metadata={"paper_title": paper.title, "paper_year": extract_year_from_date(paper.published_date)}
                            ))
        except Exception as e:
            print(f"   [WARNING] Failed to process {paper.title}: {e}")
            # Fallback: use abstract
            if paper.summary:
                from langchain_core.documents import Document
                documents.append(Document(
                    page_content=paper.summary,
                    metadata={"paper_title": paper.title, "paper_year": extract_year_from_date(paper.published_date)}
                ))
    
    if documents:
        retriever = create_vectorstore(documents)
        print(f"   -> Created {len(documents)} document chunks")
        print("   [SUCCESS] Indexed papers into vector store")
    else:
        retriever = None
        print("   [WARNING] No documents created")
    
    citation_map = {}
    return {"paper_retriever": retriever, "citation_map": citation_map}


def subsection_writer_node(state: AgentState) -> dict:
    """
    Write a detailed subsection for a subcategory using RAG to find relevant papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with subsection content added to subsections dict
    """
    current_subcategory = state.get('current_subcategory', '')
    topic = state.get('topic', '')
    paper_retriever = state.get('paper_retriever')
    citation_map = state.get('citation_map', {})
    subsections = state.get('subsections', {})
    
    if not current_subcategory:
        return {}
    
    print(f"   -> Writing subsection: {current_subcategory}")
    
    # Skip if this is metadata (year, author, etc.)
    metadata_keywords = ['year', 'author', 'publication', 'date', 'journal', 'conference']
    if any(keyword in current_subcategory.lower() for keyword in metadata_keywords):
        print(f"   [SKIP] Skipping metadata subcategory: {current_subcategory}")
        return {}
    
    # Use RAG to find relevant papers
    if paper_retriever:
        try:
            query = f"{current_subcategory} in the context of {topic}"
            # Support both old and new LangChain APIs
            if hasattr(paper_retriever, 'get_relevant_documents'):
                relevant_docs = paper_retriever.get_relevant_documents(query)
            else:
                relevant_docs = paper_retriever.invoke(query)
            
            if not relevant_docs:
                print(f"   [SKIP] No relevant papers found for: {current_subcategory}")
                return {}
            
            # Extract paper information
            paper_chunks = {}
            for doc in relevant_docs[:5]:  # Top 5 chunks
                paper_title = doc.metadata.get('paper_title', '')
                if paper_title:
                    if paper_title not in paper_chunks:
                        paper_chunks[paper_title] = []
                    paper_chunks[paper_title].append(doc.page_content[:300])
            
            if not paper_chunks:
                print(f"   [SKIP] No papers found for: {current_subcategory}")
                return {}
            
            # Build context
            papers_context = []
            for paper_title, chunks in paper_chunks.items():
                citation_num = get_or_assign_citation(citation_map, paper_title)
                papers_context.append(f"{format_citation(citation_num)} {paper_title}: {' '.join(chunks[:2])}")
            
            papers_text = "\n\n".join(papers_context)
            
        except Exception as e:
            print(f"   [WARNING] RAG retrieval failed: {e}")
            papers_text = ""
    else:
        papers_text = ""
    
    # Generate subsection
    prompt = f"""Write a detailed subsection about "{current_subcategory}" for a survey paper on "{topic}".

Relevant Papers and Content:
{papers_text}

Requirements:
- Provide a clear definition and explanation
- Discuss key contributions from the papers
- Use numbered citations like [1], [2], etc.
- Be comprehensive but concise
- If no relevant content exists, skip this subsection

Write the subsection content in markdown format. If this subcategory is not relevant or has no supporting papers, respond with "SKIP"."""
    
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        content = res.content.strip()
        
        if "SKIP" in content.upper() or len(content) < 50:
            print(f"   [SKIP] Subsection not relevant or insufficient content")
            return {}
        
        subsections[current_subcategory] = content
        print(f"   [SUCCESS] Wrote subsection ({len(content)} characters)")
        
        return {
            "subsections": subsections,
            "citation_map": citation_map
        }
    except Exception as e:
        print(f"   [ERROR] Failed to write subsection: {e}")
        return {}


def section_writer_node(state: AgentState) -> dict:
    """
    Write an introduction section for a main category.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with section content added to sections dict
    """
    current_category = state.get('current_category', '')
    topic = state.get('topic', '')
    taxonomy_structure = state.get('taxonomy_structure', {})
    sections = state.get('sections', {})
    subsections = state.get('subsections', {})
    
    if not current_category:
        return {}
    
    print(f"   -> Writing section: {current_category}")
    
    # Check if this category has valid subsections
    main_categories = taxonomy_structure.get('main_categories', [])
    category_data = next((cat for cat in main_categories if cat.get('name') == current_category), None)
    
    if not category_data:
        return {}
    
    subcategory_names = category_data.get('subcategories', [])
    valid_subsections = [sub for sub in subcategory_names if sub in subsections]
    
    if not valid_subsections:
        print(f"   [SKIP] No valid subsections for category: {current_category}")
        return {}
    
    # Skip metadata categories
    metadata_keywords = ['year', 'author', 'publication', 'date']
    if any(keyword in current_category.lower() for keyword in metadata_keywords):
        print(f"   [SKIP] Skipping metadata category: {current_category}")
        return {}
    
    prompt = f"""Write an introduction section for the category "{current_category}" in a survey paper on "{topic}".

This category contains the following subsections:
{', '.join(valid_subsections)}

Requirements:
- Provide an overview of the category
- Explain its importance in the context of {topic}
- Introduce the subsections that follow
- Be concise (2-3 paragraphs)

Write the section content in markdown format."""
    
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        content = res.content.strip()
        
        sections[current_category] = content
        print(f"   [SUCCESS] Wrote section ({len(content)} characters)")
        
        return {"sections": sections}
    except Exception as e:
        print(f"   [ERROR] Failed to write section: {e}")
        return {}


def subsection_coordinator_node(state: AgentState) -> dict:
    """
    Coordinate writing all sections and subsections by iterating through taxonomy.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with all sections and subsections written
    """
    print("\n--- SUBSECTION COORDINATOR: Processing all categories and subcategories ---")
    taxonomy_structure = state.get('taxonomy_structure', {})
    main_categories = taxonomy_structure.get('main_categories', [])
    
    sections = state.get('sections', {})
    subsections = state.get('subsections', {})
    
    # Process each category and its subcategories
    for category in main_categories:
        category_name = category.get('name', '')
        if not category_name:
            continue
        
        # Write section introduction
        state['current_category'] = category_name
        section_result = section_writer_node(state)
        if section_result:
            sections.update(section_result.get('sections', {}))
        
        # Write subsections
        subcategories = category.get('subcategories', [])
        for subcategory in subcategories:
            if not subcategory:
                continue
            
            state['current_subcategory'] = subcategory
            subsection_result = subsection_writer_node(state)
            if subsection_result:
                subsections.update(subsection_result.get('subsections', {}))
                if 'citation_map' in subsection_result:
                    state['citation_map'] = subsection_result['citation_map']
    
    # Clean up temporary state
    state.pop('current_category', None)
    state.pop('current_subcategory', None)
    
    section_count = len(sections)
    subsection_count = len(subsections)
    print(f"   [SUCCESS] Processed {section_count} sections and {subsection_count} subsections")
    
    return {
        "sections": sections,
        "subsections": subsections
    }


def report_assembler_node(state: AgentState) -> dict:
    """
    Assemble final markdown report from all sections and subsections.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with final_report
    """
    print("\n--- REPORT ASSEMBLER: Combining all sections into final report ---")
    topic = state.get('topic', '')
    sections = state.get('sections', {})
    subsections = state.get('subsections', {})
    taxonomy_structure = state.get('taxonomy_structure', {})
    citation_map = state.get('citation_map', {})
    target_papers = state.get('target_papers', [])
    
    # Build report
    report_parts = [f"# Survey: {topic}\n\n"]
    
    # Add abstract/introduction (placeholder)
    report_parts.append("## Abstract\n\n")
    report_parts.append("This survey paper provides a comprehensive overview of recent advances in the field.\n\n")
    
    # Add sections and subsections
    main_categories = taxonomy_structure.get('main_categories', [])
    for category in main_categories:
        category_name = category.get('name', '')
        if category_name in sections:
            report_parts.append(f"## {category_name}\n\n")
            report_parts.append(sections[category_name])
            report_parts.append("\n\n")
        
        subcategories = category.get('subcategories', [])
        for subcategory in subcategories:
            if subcategory in subsections:
                report_parts.append(f"### {subcategory}\n\n")
                report_parts.append(subsections[subcategory])
                report_parts.append("\n\n")
    
    # Add references
    report_parts.append("## References\n\n")
    
    # Sort papers by citation number
    sorted_papers = sorted(target_papers, key=lambda p: citation_map.get(p.title, 999))
    
    for paper in sorted_papers:
        citation_num = citation_map.get(paper.title, 0)
        if citation_num > 0:
            year = extract_year_from_date(paper.published_date)
            report_parts.append(f"{citation_num}. {paper.title} ({year})\n")
            if paper.url:
                report_parts.append(f"   URL: {paper.url}\n")
            report_parts.append("\n")
    
    final_report = "".join(report_parts)
    
    print(f"   [SUCCESS] Assembled final report ({len(final_report)} characters)")
    return {"final_report": final_report}
