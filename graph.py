"""LangGraph workflow definition for the Survey Agent."""
from langgraph.graph import StateGraph, END
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
    relevance_judge_node,
    target_paper_termination_check_node,
    target_paper_termination_router,
    taxonomy_extractor_node,
    taxonomy_designer_node,
    taxonomy_parser_node,
    paper_indexer_node,
    subsection_coordinator_node,
    report_assembler_node
)
from models import AgentState


def build_workflow():
    """Build and return the compiled workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("initial_split", initial_split_node)
    workflow.add_node("survey_query_planner", survey_query_planner_node)
    workflow.add_node("research_query_planner", research_query_planner_node)
    workflow.add_node("survey_search", survey_search_node)
    workflow.add_node("research_search", research_search_node)
    workflow.add_node("survey_feedback_check", survey_feedback_check_node)
    workflow.add_node("survey_selector", survey_selector_node)
    workflow.add_node("survey_validation_feedback_check", survey_validation_feedback_check_node)
    workflow.add_node("relevance_judge", relevance_judge_node)
    workflow.add_node("target_paper_termination_check", target_paper_termination_check_node)
    workflow.add_node("taxonomy_extractor", taxonomy_extractor_node)
    workflow.add_node("taxonomy_designer", taxonomy_designer_node)
    workflow.add_node("taxonomy_parser", taxonomy_parser_node)
    workflow.add_node("paper_indexer", paper_indexer_node)
    workflow.add_node("subsection_coordinator", subsection_coordinator_node)
    workflow.add_node("report_assembler", report_assembler_node)
    
    # Set entry point
    workflow.set_entry_point("initial_split")
    
    # Initial split: both query planners start in parallel
    workflow.add_edge("initial_split", "survey_query_planner")
    workflow.add_edge("initial_split", "research_query_planner")
    
    # Survey track: query → search → feedback check → selector
    workflow.add_edge("survey_query_planner", "survey_search")
    workflow.add_edge("survey_search", "survey_feedback_check")
    
    # Survey feedback routing: retry planner or continue to selector
    workflow.add_conditional_edges(
        "survey_feedback_check",
        survey_feedback_router,
        {
            "survey_query_planner": "survey_query_planner",
            "survey_selector": "survey_selector"
        }
    )
    
    # Survey Track continuation - check if selector found enough valid papers
    workflow.add_edge("survey_selector", "survey_validation_feedback_check")
    
    # Survey validation feedback routing: retry selector, retry query planner, or continue to taxonomy_extractor
    workflow.add_conditional_edges(
        "survey_validation_feedback_check",
        survey_validation_feedback_router,
        {
            "survey_selector": "survey_selector",  # Retry selection with quality feedback
            "survey_query_planner": "survey_query_planner",  # Retry search with feedback
            "taxonomy_extractor": "taxonomy_extractor"  # Continue to taxonomy extraction
        }
    )
    
    # Research track: query → search → relevance judge → termination check
    workflow.add_edge("research_query_planner", "research_search")
    workflow.add_edge("research_search", "relevance_judge")
    workflow.add_edge("relevance_judge", "target_paper_termination_check")
    
    # Target paper termination routing
    workflow.add_conditional_edges(
        "target_paper_termination_check",
        target_paper_termination_router,
        {
            "END": END,
            "taxonomy_parser": "taxonomy_parser"
        }
    )
    
    # Taxonomy track: extractor → designer → parser
    workflow.add_edge("taxonomy_extractor", "taxonomy_designer")
    workflow.add_edge("taxonomy_designer", "taxonomy_parser")
    
    # After parser: index papers → write subsections → assemble report
    workflow.add_edge("taxonomy_parser", "paper_indexer")
    workflow.add_edge("paper_indexer", "subsection_coordinator")
    workflow.add_edge("subsection_coordinator", "report_assembler")
    workflow.add_edge("report_assembler", END)
    
    return workflow.compile()


# Create the workflow instance
workflow = build_workflow()
