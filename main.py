"""Main entry point for the Survey Agent system."""
import os
from graph import app

# --- MAIN ---

if __name__ == "__main__":
    # 1. Visualization
    try:
        print("Saving graph to 'agent_workflow.png'...")
        with open("agent_workflow.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("Graph saved!")
    except Exception:
        print("Could not save image (requires graphviz). Printing ASCII:")
        app.get_graph().print_ascii()

    # 2. Execution
    topic = input("What is your specific target topic? ")
    
    # Ask for search sources
    print("\nSearch Sources:")
    print("  1. ArXiv only")
    print("  2. Web of Science (WOS) only")
    print("  3. Both ArXiv and WOS")
    source_choice = input("Select search source (1/2/3, default=1): ").strip() or "1"
    
    search_sources = []
    pdf_folder = ""
    
    if source_choice == "1":
        search_sources = ["arxiv"]
    elif source_choice == "2":
        search_sources = ["wos"]
        pdf_folder = input("Enter folder path where you will place PDF files for WOS papers: ").strip()
        if not pdf_folder:
            pdf_folder = "./wos_pdfs"  # Default folder
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder, exist_ok=True)
            print(f"   [INFO] Created folder: {pdf_folder}")
    elif source_choice == "3":
        search_sources = ["arxiv", "wos"]
        pdf_folder = input("Enter folder path where you will place PDF files for WOS papers: ").strip()
        if not pdf_folder:
            pdf_folder = "./wos_pdfs"  # Default folder
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder, exist_ok=True)
            print(f"   [INFO] Created folder: {pdf_folder}")
    else:
        print("   [WARNING] Invalid choice, defaulting to ArXiv only")
        search_sources = ["arxiv"]
    
    print(f"\nStarting Multi-Agent System for: {topic}...")
    print(f"  Search sources: {', '.join(search_sources)}")
    if pdf_folder:
        print(f"  PDF folder: {pdf_folder}")
    
    # Use invoke() to run to completion and get the final result
    final_state = app.invoke({
        "topic": topic,
        "search_sources": search_sources,
        "pdf_folder": pdf_folder,
        "search_queries": [],  # Legacy field
        "survey_queries": [],  # Will be set by planner
        "research_queries": [],  # Will be set by planner
        "papers": [],  # Legacy field
        "survey_papers": [],  # Will be populated by parallel_search
        "target_papers": [],  # Will be populated by parallel_search
        "taxonomy": "",  # Legacy field
        "extracted_taxonomies": [],  # Will be set by taxonomy_extractor
        "taxonomy_json": "",  # Will be set by taxonomy_designer
        "organized_papers": {},  # Will be set by sorter
        "revision_number": 0, 
        "reviewer_comments": "None",
        "review_status": "START"
    })
    
    print("\n\n" + "="*80)
    print("==================== FINAL RESULTS ====================")
    print("="*80)
    
    # Display Taxonomy
    if final_state.get('taxonomy_json'):
        print("\n--- TAXONOMY (JSON) ---")
        print(final_state['taxonomy_json'])
    else:
        print("\n--- TAXONOMY ---")
        print("No taxonomy generated.")
    
    # Display Organized Papers
    organized_papers = final_state.get('organized_papers', {})
    if organized_papers:
        print("\n--- ORGANIZED PAPERS BY CATEGORY ---")
        for category, papers in organized_papers.items():
            print(f"\n[{category}] ({len(papers)} papers)")
            for i, paper in enumerate(papers, 1):
                print(f"  {i}. {paper.title}")
                if paper.doi:
                    print(f"     DOI: {paper.doi}")
    else:
        print("\n--- ORGANIZED PAPERS ---")
        print("No papers organized.")
    
    # Display Summary
    print("\n--- SUMMARY ---")
    print(f"Survey Papers Found: {len(final_state.get('survey_papers', []))}")
    print(f"Target Papers Validated: {len(final_state.get('target_papers', []))}")
    print(f"Categories Created: {len(organized_papers)}")
    total_organized = sum(len(papers) for papers in organized_papers.values())
    print(f"Total Papers Organized: {total_organized}")
    
    print("\n" + "="*80)