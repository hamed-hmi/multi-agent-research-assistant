"""Main entry point for the Survey Agent system."""
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
    topic = input("Enter topic: ")
    print(f"Starting Multi-Agent System for: {topic}...")
    
    # Use invoke() to run to completion and get the final result
    final_state = app.invoke({
        "topic": topic, 
        "revision_number": 0, 
        "reviewer_comments": "None",
        "review_status": "START"
    })
    
    print("\n\n================ FINAL REPORT ================\n")
    print(final_state['final_report'])
    print("\n==============================================")