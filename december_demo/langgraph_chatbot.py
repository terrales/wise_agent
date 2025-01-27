import streamlit as st
import uuid
import utils_langgraph
import os


# Initialize session state for the conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "results" not in st.session_state:
    st.session_state["results"] = []

# Generate unique thread ID for each session
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())


# Configuration for LangGraph
config = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    }
}

st.title("LangGraph Chatbot Demo")
st.info("Type your message below and interact with the LangGraph agent. Type 'exit' or 'quit' to end the session.")

# Chat interface
user_input = st.text_input("Your message:", key="user_input")

if user_input:
    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        st.write("Session ended. Refresh the page to start again.")
        st.stop()

    # Process user input
    st.session_state["messages"].append(("User", user_input))
    result = {}
    graph = utils_langgraph.initialize_part_1_graph()
    events = graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="values"
    )

    # Handle streaming events
    _printed = set()
    assistant_response = ""
    tool_output = ""
    for event in events:
        utils_langgraph._print_event(event, _printed)
        utils_langgraph.add_to_results(event, result)

        current_state = event.get("dialog_state")
        
        message = event.get("messages")
        if message:
            if isinstance(message, list):
                message = message[-1]
            if message.type == 'tool':
                tool_output += message.content
            if message.type == 'ai':
                assistant_response += message.content

    # Fallback if no response
    if not assistant_response:
        assistant_response = "I couldn't process your request. Please try again."

    # Save and display assistant's response
    st.session_state["results"].append(result)
    st.session_state["messages"].append(("Tool output", tool_output))
    st.session_state["messages"].append(("Assistant", assistant_response))

# Display the conversation
for sender, message in st.session_state["messages"]:
    st.markdown(f"**{sender}:** {message}")
