"""Main entry point for the Survey Agent."""
import os
from datetime import datetime
from graph import workflow, build_workflow
from tools import visualize_categorization_graph
from models import AgentState


def main():
    """Main function to run the survey agent workflow."""
    print("="*80)
    print("Survey Agent - Multi-Agent System for Survey Paper Generation")
    print("="*80)
    
    # Get user input
    topic = input("\nWhat is your specific target topic? ").strip()
    if not topic:
        print("Error: Topic is required")
        return
    
    print("\nSearch Sources:")
    print("  1. ArXiv only")
    print("  2. Web of Science (WOS) only")
    print("  3. Both ArXiv and WOS")
    source_choice = input("Select search source (1/2/3, default=1): ").strip() or "1"
    
    source_map = {"1": ["arxiv"], "2": ["wos"], "3": ["arxiv", "wos"]}
    search_sources = source_map.get(source_choice, ["arxiv"])
    
    print(f"\nStarting Multi-Agent System for: {topic} ...")
    print(f"  Search sources: {', '.join(search_sources)}")
    
    # Save workflow visualization
    try:
        workflow_viz_file = "agent_workflow.png"
        # Get the graph structure and visualize it
        graph_structure = workflow.get_graph()
        
        # Try different visualization methods based on LangGraph version
        try:
            # Method 1: draw_mermaid_png (newer versions)
            graph_structure.draw_mermaid_png(output_file_path=workflow_viz_file)
            print(f"[SUCCESS] Workflow graph saved to: {workflow_viz_file}")
        except AttributeError:
            # Method 2: draw_mermaid with manual PNG conversion
            try:
                mermaid_code = graph_structure.draw_mermaid()
                # Save mermaid code to file
                mermaid_file = workflow_viz_file.replace('.png', '.mmd')
                with open(mermaid_file, 'w') as f:
                    f.write(mermaid_code)
                print(f"[INFO] Workflow graph (Mermaid format) saved to: {mermaid_file}")
                print(f"[INFO] You can convert it to PNG using: https://mermaid.live/ or mermaid-cli")
            except Exception as e2:
                # Method 3: Print ASCII representation
                print("\n[INFO] Workflow structure (ASCII):")
                graph_structure.print_ascii()
    except Exception as e:
        print(f"[WARNING] Failed to save workflow visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize state
    initial_state: AgentState = {
        "topic": topic,
        "search_sources": search_sources,
        "survey_queries": [],
        "research_queries": [],
        "survey_papers": [],
        "target_papers": [],
        "extracted_taxonomies": [],
        "taxonomy_json": "",
        "taxonomy_structure": {},
        "paper_retriever": None,
        "citation_map": {},
        "subsections": {},
        "sections": {},
        "final_report": "",
        "survey_query_feedback": "",
        "survey_selector_feedback": "",
        "survey_search_retry_count": 0,
        "survey_validation_retry_count": 0,
        "survey_selector_retry_count": 0,
        "min_survey_papers": 5,
        "min_validated_surveys": 3,
        "survey_selection_quality_threshold": 0.7,
        "max_survey_query_retries": 2,
        "workflow_terminated": False,
        "termination_message": ""
    }
    
    # Run workflow
    try:
        final_state = workflow.invoke(initial_state)
        
        # Check for termination
        if final_state.get('workflow_terminated', False):
            print("\n" + "="*80)
            print("WORKFLOW TERMINATED")
            print("="*80)
            print(final_state.get('termination_message', ''))
            return
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Display results
        print("\n" + "="*80)
        print("WORKFLOW COMPLETED")
        print("="*80)
        
        print(f"\nSelected Survey Papers: {len(final_state.get('survey_papers', []))}")
        for i, paper in enumerate(final_state.get('survey_papers', []), 1):
            from tools import format_paper_title_with_year
            print(f"  [{i}] {format_paper_title_with_year(paper)}")
        
        print(f"\nTarget Papers: {len(final_state.get('target_papers', []))}")
        print(f"Taxonomy Categories: {len(final_state.get('taxonomy_structure', {}).get('main_categories', []))}")
        print(f"Sections Written: {len(final_state.get('sections', {}))}")
        print(f"Subsections Written: {len(final_state.get('subsections', {}))}")
        print(f"Papers Cited: {len(final_state.get('citation_map', {}))}")
        
        # Save final report
        final_report = final_state.get('final_report', '')
        if final_report:
            report_file = f"survey_report_{timestamp}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(final_report)
            print(f"\n[SUCCESS] Final report saved to: {report_file}")
        
        # Generate visualization
        graph_file = f"categorization_graph_{timestamp}.png"
        visualize_categorization_graph(final_state, graph_file)
        # Save output log
        output_file = f"survey_agent_output_{timestamp}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Survey Agent Run - {timestamp}\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Search Sources: {', '.join(search_sources)}\n\n")
            f.write(f"Survey Papers: {len(final_state.get('survey_papers', []))}\n")
            f.write(f"Target Papers: {len(final_state.get('target_papers', []))}\n")
            f.write(f"Final Report Length: {len(final_report)} characters\n")
        
        print(f"[SUCCESS] Output log saved to: {output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
