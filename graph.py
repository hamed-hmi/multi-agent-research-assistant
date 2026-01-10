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
    sorter_node
)


def build_workflow() -> StateGraph:
    """
    Build and configure the agent workflow graph with separate paths for survey and research queries.
    
    Survey Path: 
        survey_query_planner → survey_search → feedback_check → (retry OR continue) → 
        survey_selector (select & validate top 3) → validation_feedback_check → 
        (retry OR continue) → taxonomy → sorter
    
    Research Path: 
        research_query_planner → research_search → relevance_judge → validator → 
        termination_check → (END OR sorter)
    
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
    
    # Convergence node
    workflow.add_node("sorter", sorter_node)
    
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
    workflow.add_edge("taxonomy_designer", "sorter")
    
    # Research Track flow (runs in parallel with survey track)
    workflow.add_edge("research_query_planner", "research_search")
    workflow.add_edge("research_search", "relevance_judge")
    workflow.add_edge("relevance_judge", "paper_validator")
    workflow.add_edge("paper_validator", "target_paper_termination_check")
    
    # Termination check routing: terminate or continue to sorter
    workflow.add_conditional_edges(
        "target_paper_termination_check",
        target_paper_termination_router,
        {
            END: END,  # Terminate if insufficient papers
            "sorter": "sorter"  # Continue to sorter if enough papers
        }
    )
    
    # End after sorter
    workflow.add_edge("sorter", END)
    
    # Compile and return
    return workflow.compile()


# Create the app instance
app = build_workflow()

