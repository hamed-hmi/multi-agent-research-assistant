"""Graph construction and compilation for the Survey Agent workflow."""
from langgraph.graph import StateGraph, END

from models import AgentState
from nodes import (
    initial_split_node,
    survey_query_planner_node,
    research_query_planner_node,
    survey_search_node,
    research_search_node,
    survey_feedback_check_node,
    survey_feedback_router,
    survey_selector_node,
    survey_validation_feedback_check_node,
    survey_validation_feedback_router,
    taxonomy_extractor_node,
    taxonomy_designer_node,
    relevance_judge_node,
    paper_validator_node,
    target_paper_termination_check_node,
    target_paper_termination_router,
    taxonomy_parser_node,
    paper_indexer_node,
    subsection_coordinator_node,
    report_assembler_node
)


def build_workflow() -> StateGraph:
    """
    Build and configure the agent workflow graph with separate paths for survey and research queries.
    
    Survey Path: 
        survey_query_planner → survey_search → feedback_check → (retry OR continue) → 
        survey_selector (select & validate top 3) → validation_feedback_check → 
        (retry OR continue) → taxonomy_extractor → taxonomy_designer → taxonomy_parser
    
    Research Path: 
        research_query_planner → research_search → relevance_judge → validator → 
        termination_check → (END OR taxonomy_parser)
    
    Convergence:
        taxonomy_parser → paper_indexer → subsection_coordinator → report_assembler → END
    
    Returns:
        Compiled workflow application
    """
    workflow = StateGraph(AgentState)
    
    # Initial split node
    workflow.add_node("initial_split", initial_split_node)
    
    # Survey Track nodes (with feedback mechanism)
    workflow.add_node("survey_query_planner", survey_query_planner_node)
    workflow.add_node("survey_search", survey_search_node)
    workflow.add_node("survey_feedback_check", survey_feedback_check_node)
    workflow.add_node("survey_selector", survey_selector_node)
    workflow.add_node("survey_validation_feedback_check", survey_validation_feedback_check_node)
    workflow.add_node("taxonomy_extractor", taxonomy_extractor_node)
    workflow.add_node("taxonomy_designer", taxonomy_designer_node)
    
    # Research Track nodes (with termination check)
    workflow.add_node("research_query_planner", research_query_planner_node)
    workflow.add_node("research_search", research_search_node)
    workflow.add_node("relevance_judge", relevance_judge_node)
    workflow.add_node("paper_validator", paper_validator_node)
    workflow.add_node("target_paper_termination_check", target_paper_termination_check_node)
    
    # Subsection writing nodes
    workflow.add_node("taxonomy_parser", taxonomy_parser_node)
    workflow.add_node("paper_indexer", paper_indexer_node)
    workflow.add_node("subsection_coordinator", subsection_coordinator_node)
    workflow.add_node("report_assembler", report_assembler_node)
    
    # Set entry point - split into two parallel paths
    workflow.set_entry_point("initial_split")
    
    # Split into two parallel paths from initial_split
    workflow.add_edge("initial_split", "survey_query_planner")
    workflow.add_edge("initial_split", "research_query_planner")
    
    # Survey Track flow
    workflow.add_edge("survey_query_planner", "survey_search")
    workflow.add_edge("survey_search", "survey_feedback_check")
    
    # Survey feedback routing: retry survey_query_planner or continue to selector
    workflow.add_conditional_edges(
        "survey_feedback_check",
        survey_feedback_router,
        {
            "survey_query_planner": "survey_query_planner",  # Retry with feedback
            "survey_selector": "survey_selector"  # Continue to selection (with validation)
        }
    )
    
    # Survey Track continuation - check if selector found enough valid papers
    workflow.add_edge("survey_selector", "survey_validation_feedback_check")
    
    # Survey validation feedback routing: retry survey_query_planner or continue to taxonomy_extractor
    workflow.add_conditional_edges(
        "survey_validation_feedback_check",
        survey_validation_feedback_router,
        {
            "survey_query_planner": "survey_query_planner",  # Retry with feedback
            "taxonomy_extractor": "taxonomy_extractor"  # Continue to taxonomy extraction
        }
    )
    workflow.add_edge("taxonomy_extractor", "taxonomy_designer")
    workflow.add_edge("taxonomy_designer", "taxonomy_parser")
    
    # Research Track flow (runs in parallel with survey track)
    workflow.add_edge("research_query_planner", "research_search")
    workflow.add_edge("research_search", "relevance_judge")
    workflow.add_edge("relevance_judge", "paper_validator")
    workflow.add_edge("paper_validator", "target_paper_termination_check")
    
    # Termination check routing: terminate or continue to taxonomy_parser
    workflow.add_conditional_edges(
        "target_paper_termination_check",
        target_paper_termination_router,
        {
            END: END,  # Terminate if insufficient papers
            "taxonomy_parser": "taxonomy_parser"  # Continue to subsection writing if enough papers
        }
    )
    
    # Convergence: Both tracks lead to taxonomy_parser, then subsection writing
    # Note: taxonomy_parser will wait if either taxonomy_json or target_papers are missing
    workflow.add_edge("taxonomy_parser", "paper_indexer")
    workflow.add_edge("paper_indexer", "subsection_coordinator")
    workflow.add_edge("subsection_coordinator", "report_assembler")
    workflow.add_edge("report_assembler", END)
    
    # Compile and return
    return workflow.compile()


# Create the app instance
app = build_workflow()

