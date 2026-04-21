"""
Agentic AI Pattern #2: ReAct (Reasoning + Acting) with Grounding Validation
=============================================================================
Demonstrates the ReAct pattern using LangGraph with:
  - Thought → Action → Observation loop
  - Tool calling (web search, calculator, weather)
  - Answer grounding validation (verifies claims against tool observations)
  - 2-layer claim validation: Keywords → Embeddings
  - Retry mechanism when answer contains ungrounded claims

The Flow:
  AGENT (think + decide) → TOOL (execute) → AGENT (observe + think again)
                         → GROUNDING CHECK (validate final answer)
                         → END (if grounded) or RETRY (if hallucinated)

Usage:
  export ANTHROPIC_API_KEY=your-key-here
  python react_langgraph.py
  python react_langgraph.py "What is the population of the capital of France?"
"""

import sys
import json
import math
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

MODEL = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 10          # Max tool calls before forced stop
GROUNDING_THRESHOLD = 0.7    # 70% of claims must be grounded in tool results
MAX_RETRIES = 2              # Max times to retry after grounding failure
KEYWORD_THRESHOLD = 0.5      # Layer 1: 50% keywords must match
EMBEDDING_THRESHOLD = 0.7    # Layer 2: cosine similarity minimum

DEFAULT_QUESTION = "What is the population of the capital of France and what's the weather like there today?"


# =============================================================================
# Tools — The actions the agent can take
# =============================================================================

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use this for facts, statistics, and general knowledge."""
    # Using DuckDuckGo for real search results
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n\n".join(
                    f"Source: {r.get('href', 'N/A')}\n{r.get('body', 'No content')}"
                    for r in results
                )
            return "No results found for this query."
    except Exception as e:
        return f"Search error: {str(e)}. Please try a different query."


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression. Use this for any math operations.
    Examples: '2 + 2', '100 * 0.15', 'sqrt(144)', '2**10'"""
    # Sandboxed math evaluation — only allows safe math operations
    allowed_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
    }
    try:
        # Only allow math operations, no builtins
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}. Please check the expression."


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this for weather-related questions."""
    # Simulated weather data for demo purposes
    # In production, this would call a real weather API
    weather_data = {
        "paris": "Paris: 18°C, partly cloudy, humidity 65%, wind 12 km/h NW",
        "london": "London: 14°C, light rain, humidity 78%, wind 20 km/h W",
        "tokyo": "Tokyo: 22°C, sunny, humidity 55%, wind 8 km/h E",
        "new york": "New York: 20°C, clear skies, humidity 50%, wind 15 km/h SW",
        "sydney": "Sydney: 16°C, overcast, humidity 72%, wind 18 km/h SE",
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        return weather_data[city_lower]
    return f"Weather data not available for '{city}'. Available cities: Paris, London, Tokyo, New York, Sydney."


# All tools available to the agent
tools = [web_search, calculator, get_weather]


# =============================================================================
# Initialize LLM and Embedding Model
# =============================================================================

llm = ChatAnthropic(model=MODEL, max_tokens=1024)
llm_with_tools = llm.bind_tools(tools)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =============================================================================
# System Prompt for the Agent
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.

Your approach:
1. THINK about what information you need to answer the question
2. USE tools to gather that information
3. THINK about what you learned and if you need more information
4. REPEAT until you have enough information
5. PROVIDE a final answer based ONLY on what your tools returned

IMPORTANT RULES:
- ONLY include facts in your final answer that came from your tool results
- Do NOT add information from your training data that wasn't confirmed by tools
- If a tool doesn't return useful information, say so — don't guess
- Be specific and cite the information source (which tool provided it)"""


# =============================================================================
# Grounding Validation — Keywords + Embeddings
# =============================================================================

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "should", "could", "more",
    "very", "also", "been", "have", "has", "was", "were", "are", "its",
    "from", "not", "but", "can", "will", "would", "into", "about", "is",
    "of", "in", "to", "a", "an", "it", "on", "at", "by", "as", "or",
}

CLAIMS_EXTRACTION_PROMPT = """Extract all factual claims from the following answer.
Return ONLY a JSON array of strings. Each string should be one verifiable factual claim.

Rules:
- Only include verifiable facts (numbers, names, dates, measurements)
- Do NOT include opinions, transitions, or filler text
- Each claim should be self-contained (understandable without context)
- If there are no factual claims, return an empty array []

Example input: "Paris, the capital of France, has about 2.1 million residents and it's currently 18 degrees there."
Example output: ["Paris is the capital of France", "Paris has about 2.1 million residents", "The temperature in Paris is 18 degrees"]

Now extract claims from this answer:
"""


def keyword_check(claim: str, observations_text: str) -> bool:
    """Layer 1: Fast keyword matching."""
    keywords = [
        w for w in claim.lower().split()
        if len(w) >= 3 and w not in STOPWORDS
    ]
    if not keywords:
        return False
    matches = sum(1 for kw in keywords if kw in observations_text.lower())
    return (matches / len(keywords)) >= KEYWORD_THRESHOLD


def embedding_check(claim: str, observations: list[str]) -> float:
    """Layer 2: Semantic similarity using local embeddings."""
    if not observations:
        return 0.0
    claim_emb = embedding_model.encode([claim])
    obs_embs = embedding_model.encode(observations)
    similarities = embedding_model.similarity(claim_emb, obs_embs)[0]
    return float(similarities.max())


def validate_claims(claims: list[str], observations: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate each claim against tool observations.
    Returns (grounded_claims, ungrounded_claims).
    """
    observations_text = " ".join(observations)
    grounded = []
    ungrounded = []

    for claim in claims:
        # Layer 1: Keyword check (fast)
        if keyword_check(claim, observations_text):
            grounded.append(claim)
            continue

        # Layer 2: Embedding check (only if keywords failed)
        best_sim = embedding_check(claim, observations)
        if best_sim >= EMBEDDING_THRESHOLD:
            grounded.append(claim)
        else:
            ungrounded.append(claim)

    return grounded, ungrounded


# =============================================================================
# Track state for retry logic
# =============================================================================

retry_count = 0


# =============================================================================
# Graph Nodes
# =============================================================================

def agent(state: MessagesState) -> dict:
    """
    AGENT NODE — The "Thinker + Actor"

    The LLM receives the full conversation (including tool results) and decides:
    - Call another tool (needs more information)
    - Give a final answer (has enough information)
    """
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)


    return {"messages": [response]}


def grounding_check(state: MessagesState) -> dict:
    """
    GROUNDING CHECK NODE — Validates the final answer

    1. Extract the final answer (last AIMessage)
    2. Collect all tool observations (ToolMessages)
    3. Use LLM to extract factual claims from the answer
    4. Validate each claim against observations (keywords + embeddings)
    5. Score = grounded / total claims
    6. PASS if score >= threshold, FAIL otherwise
    """
    messages = state["messages"]

    # Get the final answer
    final_answer = messages[-1].content

    # Collect all tool observations
    observations = [
        msg.content for msg in messages
        if isinstance(msg, ToolMessage)
    ]

    if not observations:
        return {}

    # Extract factual claims from the answer using LLM
    extraction_messages = [
        HumanMessage(content=CLAIMS_EXTRACTION_PROMPT + final_answer)
    ]
    extraction_response = llm.invoke(extraction_messages)

    try:
        claims = json.loads(extraction_response.content)
    except json.JSONDecodeError:
        # Try to find JSON array in the response
        content = extraction_response.content
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            try:
                claims = json.loads(content[start:end])
            except json.JSONDecodeError:
                claims = []
        else:
            claims = []

    if not claims:
        return {}

    # Validate each claim
    grounded, ungrounded = validate_claims(claims, observations)

    # Calculate score
    total = len(grounded) + len(ungrounded)
    score = len(grounded) / total if total > 0 else 1.0

    if score >= GROUNDING_THRESHOLD:
        return {}
    else:
        # Return feedback to the agent about ungrounded claims
        feedback = (
            f"GROUNDING CHECK FAILED (score: {score:.0%}, need: {GROUNDING_THRESHOLD:.0%}).\n\n"
            f"The following claims in your answer are NOT supported by any tool result:\n"
            + "\n".join(f"- {c}" for c in ungrounded)
            + "\n\nPlease either:\n"
            "1. Use a tool to verify these claims, OR\n"
            "2. Remove them from your answer and only state facts from tool results."
        )
        return {"messages": [HumanMessage(content=feedback)]}


# =============================================================================
# Conditional Edges
# =============================================================================

def after_agent(state: MessagesState) -> Literal["tools", "grounding_check"]:
    """
    After agent node: route to tools or grounding check.
    - If the LLM made tool calls → go to tools
    - If the LLM gave a final answer → go to grounding check
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "grounding_check"


def after_grounding(state: MessagesState) -> Literal["agent", "__end__"]:
    """
    After grounding check: pass or retry.
    - If grounding added feedback (HumanMessage) → retry (go back to agent)
    - If no feedback added → passed, end
    """
    global retry_count
    last_message = state["messages"][-1]

    # If the last message is a HumanMessage (feedback), the check failed
    if isinstance(last_message, HumanMessage) and "GROUNDING CHECK FAILED" in last_message.content:
        retry_count += 1
        if retry_count > MAX_RETRIES:
            return "__end__"
        return "agent"

    return "__end__"


# =============================================================================
# Build the LangGraph
# =============================================================================

def build_react_graph():
    """
    Constructs the ReAct graph:

        START → agent → tools → agent → ... → grounding_check → END
                  │                                    │
                  └── final answer ───→ grounding_check │
                                              │        │
                                            FAIL → agent (retry)
                                            PASS → END
    """
    graph = StateGraph(MessagesState)

    # Nodes
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("grounding_check", grounding_check)

    # Edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "grounding_check": "grounding_check"})
    graph.add_edge("tools", "agent")  # After tool execution, always go back to agent
    graph.add_conditional_edges("grounding_check", after_grounding, {"agent": "agent", "__end__": END})

    return graph.compile()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION

    app = build_react_graph()

    initial_input = {
        "messages": [HumanMessage(content=question)]
    }

    result = app.invoke(initial_input, config={"recursion_limit": MAX_ITERATIONS * 2})

    # Print final answer
    all_messages = result.get("messages", [])
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            print("\nFINAL ANSWER:")
            print(msg.content)
            break


if __name__ == "__main__":
    main()
