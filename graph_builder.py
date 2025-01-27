from langgraph.graph import StateGraph, START, END
from tools_utils import *
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.prompts import ChatPromptTemplate
from agent import Assistant
from memory import Memory, retrieve_from_memory, update_memory  
from evaluate import Evaluate, validate_answer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import Command, interrupt
from auto_clarification import resolve_reading
from langchain_core.tools import tool

import logging


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Initialize memory instance
memory = Memory()

# Initialize evaluator
evaluator = Evaluate()

# Wrap retrieve_from_memory tool
@tool
def retrieve_tool(key: str):
    """
    Tool to access your memory of past questions and how you answered them.

    Args: 
    key: the original user query and if it was ambuiguous or unambiguous. 
            use this format: f"{user query} - {ambiguity type}".
    """
    return retrieve_from_memory(key, memory)  

@tool
def update_tool(query: str, steps_to_answer: dict):
    """
    Tool to update memory with query and steps to answer.

    Args: 
    query: the string containing the original user querynd if it was ambuiguous or unambiguous. 
            use this format: f"{user query} - {ambiguity type}".
    steps_to_answer (Dict{steps: list, answer: str}): the dictionary containing the list of tools 
                                    called to answer the query and the string containing the answer.
    """
    return update_memory(query, steps_to_answer, memory)

@tool
def get_user_clarification(query: str) -> str:
    """
    This function interrupts the current conversation to ask the user for clarification.

    Args: 
    query: the string containing what you need clarification on with regards to the original query.
    """
    human_response = interrupt({"query": query})
    return Command(resume={"data": query + human_response})


# Define tools, explicitly passing the memory instance to memory-related tools
uncertainty_tools = [retrieve_tool]
clarfication_tools = [resolve_reading]
eval_tools = [validate_answer]
memory_update_tools = [update_tool]

# Create tool name sets
uncertainty_tools_names = {t.name for t in uncertainty_tools}
clarfication_tools_names = {t.name for t in clarfication_tools}
eval_tools_names = {t.name for t in eval_tools}
memory_update_tools_names = {t.name for t in memory_update_tools}


# Set up rate limiter
my_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.05,
    check_every_n_seconds=0.5,
    max_bucket_size=5,
)


# Set up language model with tools
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=3,
    rate_limiter=my_rate_limiter,
    timeout=240,
)


# Define the assistant's prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a question-answering agent helping the user with some ambiguous sentences."
          #  "You can ask for clarification if a sentence is underspecified and access your memory of past questions and how you answered them."
            "Use the tools available to you to answer the user's questions."
            "You can use resolve_reading if the sentence is underspecified and you'd like clarification."
            "You can call the tool retrieve_tool to access your memory of past questions and how you answered them if you're unsure of what to do."
            "Your final answer needs to be 'Yes' or 'No'."
            "After providing an answer, always call the tool validate_answer to evaluate it."
            "If the answer is correct, update your memory using update_tool.",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Combine the prompt with the tools for the assistant
exp_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(uncertainty_tools + clarfication_tools + eval_tools + memory_update_tools)


# Build the state graph
def build_graph():
    def route_tools(state: State):
        route = tools_condition(state)
        if route == END:
            return END

        ai_message = state["messages"][-1]

   #     previous_message = state["messages"][-2] if (len(state["messages"]) > 1) else None
   #     previous_ai_message = previous_message if previous_message.type == "ai" else None

        # Check if tool_calls exist and are valid
        if not ai_message.tool_calls:
            logging.error("No tool calls in the message.")
            return None

        first_tool_call = ai_message.tool_calls[0]
  #      previous_tool_call = previous_ai_message.tool_calls[0] if previous_ai_message else None

  #      if previous_tool_call:

   #         if previous_tool_call["name"] in eval_tools_names:
   #             if "Correct" in previous_tool_call:
   #                 return "memory_update_tools"

        if first_tool_call["name"] in uncertainty_tools_names:
            return "uncertainty_tools"
        elif first_tool_call["name"] in clarfication_tools_names:
            return "clarfication_tools"
        elif first_tool_call["name"] in eval_tools_names:
            return "eval_tools"
        elif first_tool_call["name"] in memory_update_tools_names:
            return "memory_update_tools"
        else:
            logging.error(f"Unknown tool call: {first_tool_call['name']}")
            return None

    builder = StateGraph(State)

    # Define nodes
    builder.add_node("assistant", Assistant(exp_1_assistant_runnable))
    builder.add_node("uncertainty_tools", create_tool_node_with_fallback(uncertainty_tools))
    builder.add_node("clarfication_tools", create_tool_node_with_fallback(clarfication_tools))
    builder.add_node("eval_tools", create_tool_node_with_fallback(eval_tools))
    builder.add_node("memory_update_tools", create_tool_node_with_fallback(memory_update_tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant", route_tools, ["uncertainty_tools", "clarfication_tools", "eval_tools", "memory_update_tools", END]
    )
    builder.add_edge("uncertainty_tools", "assistant")
    builder.add_edge("clarfication_tools", "assistant")
    builder.add_edge("eval_tools", "assistant")
    builder.add_edge("memory_update_tools", "assistant")

    # Use memory saver for checkpointing
    checkpoint_memory = MemorySaver()
    return builder.compile(
        checkpointer=checkpoint_memory,
    )

