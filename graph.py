"""Graph construction and compilation for the Survey Agent workflow."""
from langgraph.graph import StateGraph, END

from models import AgentState
from nodes import (
    planner_node,
    search_node,
    filter_node,
    wos_pause_node,
    analyst_node,
    writer_node,
    reviewer_node,
    router
)


def build_workflow() -> StateGraph:
    """
    Build and configure the agent workflow graph.
    
    Returns:
        Compiled workflow application
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("searcher", search_node)
    workflow.add_node("filter", filter_node)
    workflow.add_node("wos_pause", wos_pause_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "filter")
    workflow.add_edge("filter", "wos_pause")
    workflow.add_edge("wos_pause", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_conditional_edges("reviewer", router, {"writer": "writer", END: END})
    
    # Compile and return
    return workflow.compile()


# Create the app instance
app = build_workflow()

