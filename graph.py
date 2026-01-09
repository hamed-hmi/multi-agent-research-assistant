"""Graph construction and compilation for the Survey Agent workflow."""
from langgraph.graph import StateGraph, END

from models import AgentState
from nodes import (
    planner_node,
    parallel_search_node,
    survey_validator_node,
    taxonomy_extractor_node,
    taxonomy_designer_node,
    relevance_judge_node,
    paper_validator_node,
    sorter_node
)


def build_workflow() -> StateGraph:
    """
    Build and configure the agent workflow graph with parallel tracks.
    
    Returns:
        Compiled workflow application
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("parallel_search", parallel_search_node)
    
    # Survey Track nodes
    workflow.add_node("survey_validator", survey_validator_node)
    workflow.add_node("taxonomy_extractor", taxonomy_extractor_node)
    workflow.add_node("taxonomy_designer", taxonomy_designer_node)
    
    # Paper Track nodes
    workflow.add_node("relevance_judge", relevance_judge_node)
    workflow.add_node("paper_validator", paper_validator_node)
    
    # Convergence node
    workflow.add_node("sorter", sorter_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Main flow: planner -> parallel search
    workflow.add_edge("planner", "parallel_search")
    
    # Split into parallel tracks after search
    workflow.add_edge("parallel_search", "survey_validator")  # Survey Track
    workflow.add_edge("parallel_search", "relevance_judge")   # Paper Track
    
    # Survey Track flow
    workflow.add_edge("survey_validator", "taxonomy_extractor")
    workflow.add_edge("taxonomy_extractor", "taxonomy_designer")
    
    # Paper Track flow
    workflow.add_edge("relevance_judge", "paper_validator")
    
    # Convergence: both tracks feed into sorter
    # Note: Sorter will execute when either edge completes, but will check if both are ready
    workflow.add_edge("taxonomy_designer", "sorter")
    workflow.add_edge("paper_validator", "sorter")
    
    # End after sorter
    workflow.add_edge("sorter", END)
    
    # Compile and return
    return workflow.compile()


# Create the app instance
app = build_workflow()

