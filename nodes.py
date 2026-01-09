"""Graph nodes for the Survey Agent workflow."""
import os
import json
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
    format_graph_edges
)


def planner_node(state: AgentState) -> dict:
    """
    Generate sophisticated search queries for both survey papers and research papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_queries and research_queries
    """
    print(f"\n--- PLANNER: Processing '{state['topic']}' ---")
    
    system_msg = SystemMessage(content="""You are an expert at creating precise ArXiv and Web of Science search queries. 
Generate two sets of queries:
1. Survey queries: To find survey, review, and literature review papers
2. Research queries: To find original research papers (NOT surveys)

ArXiv/WOS Query Syntax Rules:
- Space-separated terms are implicitly ANDed (e.g., "machine learning neural networks")
- Use explicit "OR" for alternatives (e.g., "(transformer OR attention) language model")
- Use "ANDNOT" to exclude terms (e.g., "transformer ANDNOT CNN")
- Use parentheses to group logical operations
- Terms can be quoted for exact phrases: "exact phrase"
- Make queries specific and focused, not generic
- Each query should be a complete, valid search string

For Survey Queries: Include terms like "survey", "review", "systematic review", "literature review", "state-of-the-art", "comprehensive review"
For Research Queries: Use the topic directly, but EXCLUDE survey/review terms

Example: If topic is "federated learning for channel estimation":
Survey Query: "(federated learning OR distributed learning) AND (channel estimation OR CSI estimation) AND (survey OR review OR literature review)"
Research Query: "(channel estimation OR CSI estimation) AND (federated OR distributed OR FL) ANDNOT (survey OR review OR literature review)"

""")
    
    structured_llm = llm.with_structured_output(DualSearchQueries)
    result = structured_llm.invoke([system_msg, HumanMessage(content=f"Topic: {state['topic']}")])
    return {
        "survey_queries": result.survey_queries,
        "research_queries": result.research_queries,
        "search_queries": result.research_queries  # Legacy compatibility
    }


def parallel_search_node(state: AgentState) -> dict:
    """
    Search for both survey papers and research papers in parallel.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with survey_papers and target_papers (initial separation)
    """
    print("\n--- PARALLEL SEARCHER: Fetching metadata for both tracks ---")
    sources = state.get('search_sources', ['arxiv'])
    survey_papers = []
    research_papers = []
    
    # Search for survey papers
    print("\n   [SURVEY TRACK] Searching for survey papers...")
    survey_queries = state.get('survey_queries', [])
    if survey_queries:
        if 'arxiv' in sources:
            print("   -> Searching ArXiv for surveys...")
            arxiv_surveys = search_arxiv(survey_queries, max_results=5)
            survey_papers.extend(arxiv_surveys)
        
        if 'wos' in sources:
            print("   -> Searching Web of Science for surveys...")
            wos_surveys = search_wos(survey_queries, max_results=5)
            survey_papers.extend(wos_surveys)
            print(f"   [INFO] Found {len(wos_surveys)} WOS survey papers. PDFs must be provided manually.")
    
    # Search for research papers
    print("\n   [PAPER TRACK] Searching for research papers...")
    research_queries = state.get('research_queries', [])
    if research_queries:
        if 'arxiv' in sources:
            print("   -> Searching ArXiv for research papers...")
            arxiv_research = search_arxiv(research_queries, max_results=10)
            research_papers.extend(arxiv_research)
        
        if 'wos' in sources:
            print("   -> Searching Web of Science for research papers...")
            wos_research = search_wos(research_queries, max_results=10)
            research_papers.extend(wos_research)
            print(f"   [INFO] Found {len(wos_research)} WOS research papers. PDFs must be provided manually.")
    
    # Deduplicate by title
    unique_surveys = {}
    for paper in survey_papers:
        title_key = paper.title.lower().strip()
        if title_key not in unique_surveys:
            unique_surveys[title_key] = paper
    
    unique_research = {}
    for paper in research_papers:
        title_key = paper.title.lower().strip()
        if title_key not in unique_research:
            unique_research[title_key] = paper
    
    print(f"\n   [SUMMARY] Found {len(unique_surveys)} unique survey papers and {len(unique_research)} unique research papers")
    
    return {
        "survey_papers": list(unique_surveys.values()),
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
1. Is it actually a survey/review paper? (Look for keywords: survey, review, literature review, systematic review, state-of-the-art)
2. Is it relevant to the topic?
3. Does it appear to have a taxonomy, classification, or structured organization?
4. Is it of reasonable quality? (Not too old, not too brief)

Respond with ONLY 'YES' if it should be kept, or 'NO' if it should be excluded."""
        
        res = llm.invoke([HumanMessage(content=validation_prompt)])
        if "YES" in res.content.upper():
            validated.append(paper)
            print(f"   [KEPT] {paper.title[:50]}...")
        else:
            print(f"   [SKIP] {paper.title[:50]}...")
    
    print(f"\n   [SUMMARY] Validated {len(validated)} out of {len(survey_papers)} survey papers")
    return {"survey_papers": validated}


def taxonomy_extractor_node(state: AgentState) -> dict:
    """
    Extract taxonomy structures from validated survey papers.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with extracted_taxonomies (list of taxonomy strings)
    """
    print("\n--- TAXONOMY EXTRACTOR: Extracting taxonomies from surveys ---")
    survey_papers = state.get('survey_papers', [])
    extracted_taxonomies = []
    
    if not survey_papers:
        print("   [INFO] No survey papers to extract taxonomies from")
        return {"extracted_taxonomies": []}
    
    # Process papers (download/read PDFs)
    pdf_folder = state.get('pdf_folder', '')
    docs = process_papers(survey_papers, pdf_folder)
    
    if not docs:
        print("   [WARNING] Could not process survey papers for taxonomy extraction")
        return {"extracted_taxonomies": []}
    
    # Group docs by paper title
    papers_docs = {}
    for doc in docs:
        paper_title = doc.metadata.get('source', 'Unknown')
        if paper_title not in papers_docs:
            papers_docs[paper_title] = []
        papers_docs[paper_title].append(doc)
    
    # Extract taxonomy from each survey paper
    for paper in survey_papers:
        paper_docs = papers_docs.get(paper.title, [])
        if not paper_docs:
            print(f"   [SKIP] No content extracted for: {paper.title[:40]}...")
            continue
        
        # Combine text from this paper
        paper_text = "\n\n".join([doc.page_content for doc in paper_docs[:5]])  # Limit to first 5 chunks
        paper_text = paper_text[:5000]  # Limit total length
        
        extraction_prompt = f"""Extract the taxonomy, classification, or organizational structure from this survey paper.

Topic: {state['topic']}
Paper Title: {paper.title}

Paper Content (excerpt):
{paper_text}

Look for:
- Hierarchical classifications
- Taxonomy structures
- Categorization schemes
- Method classifications
- Any structured organization of the field

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
            print(f"   [EXTRACTED] Taxonomy from: {paper.title[:40]}...")
        else:
            print(f"   [NO_TAXONOMY] {paper.title[:40]}...")
    
    return {"extracted_taxonomies": extracted_taxonomies}


def taxonomy_designer_node(state: AgentState) -> dict:
    """
    Design unified taxonomy JSON structure from multiple extracted taxonomies.
    
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
        design_prompt = f"""Create a hierarchical taxonomy structure for the research topic.

Topic: {state['topic']}

Design a taxonomy with main categories and subcategories that would organize research papers in this field.
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
        
        design_prompt = f"""Design a unified, comprehensive taxonomy for the research topic by combining and refining these extracted taxonomies.

Topic: {state['topic']}

Extracted Taxonomies:
{taxonomy_texts}

Create a unified taxonomy that:
1. Combines the best elements from each extracted taxonomy
2. Ensures comprehensive coverage of the research domain
3. Has a clear hierarchical structure
4. Is suitable for organizing research papers

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
            print(f"   [SKIP - SURVEY] {paper.title[:40]}...")
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
            print(f"   [KEPT] {paper.title[:40]}...")
        else:
            print(f"   [SKIP] {paper.title[:40]}...")
    
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


# ========== SORTER NODE ==========

def sorter_node(state: AgentState) -> dict:
    """
    Organize target papers by taxonomy categories.
    Waits for both Survey Track and Paper Track to complete.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with organized_papers dict
    """
    target_papers = state.get('target_papers', [])
    taxonomy_json = state.get('taxonomy_json', '')
    
    # Check if both tracks have completed
    # Survey track should have produced taxonomy_json
    # Paper track should have produced target_papers
    if not taxonomy_json or not taxonomy_json.strip():
        print("\n--- SORTER: Waiting for Survey Track (taxonomy not ready) ---")
        return {}  # Return empty update to wait
    
    if not target_papers or len(target_papers) == 0:
        print("\n--- SORTER: Waiting for Paper Track (papers not ready) ---")
        return {}  # Return empty update to wait
    
    print("\n--- SORTER: Organizing papers by taxonomy ---")
    print("   [READY] Both tracks completed, organizing papers...")
    
    # Parse taxonomy to get category names
    try:
        taxonomy_data = json.loads(taxonomy_json)
        categories = []
        for main_cat in taxonomy_data.get('main_categories', []):
            cat_name = main_cat.get('name', '')
            if cat_name:
                categories.append(cat_name)
            for subcat in main_cat.get('subcategories', []):
                if subcat:
                    categories.append(subcat)
    except:
        print("   [WARNING] Could not parse taxonomy JSON, using text-based matching")
        categories = []
    
    organized = {}
    uncategorized = []
    
    # Process each paper
    pdf_folder = state.get('pdf_folder', '')
    for paper in target_papers:
        # Get paper content for better categorization
        paper_text = ""
        if paper.pdf_path and os.path.exists(paper.pdf_path):
            from tools import read_local_pdf
            paper_text = read_local_pdf(paper.pdf_path)
        elif paper.url and paper.source == "arxiv":
            from tools import download_and_read_pdf
            paper_text = download_and_read_pdf(paper.url)
        
        # Use abstract if no full text
        if not paper_text:
            paper_text = paper.summary
        
        # Limit text length
        paper_text = paper_text[:2000] if paper_text else paper.summary
        
        # Categorize paper
        if categories:
            category_prompt = f"""Categorize this research paper into one of these taxonomy categories.

Taxonomy Categories:
{chr(10).join([f"- {cat}" for cat in categories])}

Paper Title: {paper.title}
Paper Abstract/Content: {paper_text[:1000]}

Respond with ONLY the category name that best fits this paper, or "UNCATEGORIZED" if none fit."""
        else:
            category_prompt = f"""Based on the taxonomy structure, categorize this research paper.

Taxonomy Structure:
{taxonomy_json[:500]}

Paper Title: {paper.title}
Paper Abstract/Content: {paper_text[:1000]}

Respond with ONLY the category name that best fits this paper, or "UNCATEGORIZED" if none fit."""
        
        res = llm.invoke([HumanMessage(content=category_prompt)])
        category = res.content.strip()
        
        if category and category.upper() != "UNCATEGORIZED":
            if category not in organized:
                organized[category] = []
            organized[category].append(paper)
            print(f"   [CATEGORIZED] {paper.title[:40]}... -> {category}")
        else:
            uncategorized.append(paper)
            print(f"   [UNCATEGORIZED] {paper.title[:40]}...")
    
    if uncategorized:
        organized["uncategorized"] = uncategorized
    
    print(f"\n   [SUMMARY] Organized {len(target_papers)} papers into {len(organized)} categories")
    return {"organized_papers": organized}


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
            print(f"   [SKIP - SURVEY] {paper.title[:40]}...")
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
            print(f"   [KEPT] {paper.title[:40]}...")
        else:
            print(f"   [SKIP] {paper.title[:40]}...")
    
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
            print(f"\n[{i}] {paper.title}")
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


def sorter_router(state: AgentState) -> str:
    """
    Route to sorter only when both tracks are complete.
    
    Args:
        state: Current agent state
        
    Returns:
        "sorter" if both tracks ready, "wait" otherwise
    """
    taxonomy_json = state.get('taxonomy_json', '')
    target_papers = state.get('target_papers', [])
    
    # Check if both tracks have completed
    has_taxonomy = bool(taxonomy_json and taxonomy_json.strip())
    has_papers = bool(target_papers and len(target_papers) > 0)
    
    if has_taxonomy and has_papers:
        return "sorter"
    else:
        # Return to wait - but we need a wait node or loop back
        # For now, just go to sorter and let it handle partial state
        return "sorter"

