"""Main entry point for the Survey Agent system."""
import os
import sys
from datetime import datetime
from graph import app
from tools import visualize_categorization_graph

# --- MAIN ---

class TeeOutput:
    """Class to write output to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        sys.stdout = self.stdout
        self.file.close()

if __name__ == "__main__":
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"survey_agent_output_{timestamp}.txt"
    
    # Redirect all output to both console and file
    tee = TeeOutput(output_file)
    
    try:
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
            "taxonomy_structure": {},  # Will be set by taxonomy_parser
            "paper_retriever": None,  # Will be set by paper_indexer
            "citation_map": {},  # Will be set by paper_indexer
            "subsections": {},  # Will be set by subsection_writer
            "sections": {},  # Will be set by section_writer
            "final_report": "",  # Will be set by report_assembler
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
        
        # Display Final Report
        final_report = final_state.get('final_report', '')
        if final_report:
            print("\n--- FINAL SURVEY REPORT ---")
            print(final_report)
            
            # Also save report to a separate file
            report_file = f"survey_report_{timestamp}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(final_report)
            print(f"\n[INFO] Full report saved to: {report_file}")
        else:
            print("\n--- FINAL SURVEY REPORT ---")
            print("No report generated.")
        
        # Display Summary
        print("\n--- SUMMARY ---")
        print(f"Survey Papers Found: {len(final_state.get('survey_papers', []))}")
        print(f"Target Papers Validated: {len(final_state.get('target_papers', []))}")
        sections = final_state.get('sections', {})
        subsections = final_state.get('subsections', {})
        print(f"Sections Written: {len(sections)}")
        print(f"Subsections Written: {len(subsections)}")
        citation_map = final_state.get('citation_map', {})
        print(f"Papers Cited: {len(citation_map)}")
        
        # Generate categorization visualization
        print("\n--- GENERATING CATEGORIZATION VISUALIZATION ---")
        graph_file = f"categorization_graph_{timestamp}.png"
        graph_path = visualize_categorization_graph(final_state, graph_file)
        if graph_path:
            print(f"[INFO] Categorization graph saved to: {graph_path}")
        
        print("\n" + "="*80)
        print(f"\n[INFO] All output saved to: {output_file}")
    
    finally:
        # Restore stdout and close file
        tee.close()