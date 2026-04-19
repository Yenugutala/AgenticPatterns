"""
Agentic AI Pattern #1: REFLECTION (with Memory Management)
============================================================
Demonstrates the Reflection pattern using LangGraph with:
  - Hybrid stop strategy (LLM-as-Judge + hard max)
  - Structured extraction for memory compression
  - 3-layer validation: Keywords → Embeddings → Fallback
  - SQLite persistence for full message history

The Flow:
  GENERATE → REFLECT → SHOULD_CONTINUE? → COMPRESS → GENERATE (loop)
                                        → END (done)

Usage:
  export ANTHROPIC_API_KEY=your-key-here
  python reflection_langgraph.py
  python reflection_langgraph.py "benefits of open source software"
"""

import sys
import json
import sqlite3
import uuid
from datetime import datetime
from typing import TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

MODEL = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 5
QUALITY_THRESHOLD = 7
DEFAULT_TOPIC = "Why learning to code is valuable in the age of AI"

# Memory compression settings
MESSAGE_BYTE_THRESHOLD = 2000
DB_PATH = "reflection_history.db"
KEYWORD_THRESHOLD = 0.5
EMBEDDING_THRESHOLD = 0.7
VALIDATION_PASS_RATE = 0.5
FALLBACK_MESSAGE_COUNT = 4


# =============================================================================
# Custom State (replaces MessagesState)
# =============================================================================

class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    iteration: int


# =============================================================================
# System Prompts
# =============================================================================

WRITER_SYSTEM_PROMPT = """You are a skilled essay writer. Your job is to write
clear, engaging, and well-structured short essays (3-4 paragraphs).

When you receive feedback on a previous draft, carefully address every point
raised and produce an improved version. Do NOT repeat the feedback — just
write the improved essay directly."""

CRITIC_SYSTEM_PROMPT = """You are a constructive writing critic. Your job is to
review an essay and provide structured feedback.

You MUST respond in this exact JSON format and nothing else:

{
    "quality_score": <number from 1-10>,
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "suggestions": ["suggestion 1", "suggestion 2"]
}

Scoring guide:
  1-3: Poor — major issues with clarity, structure, or accuracy
  4-6: Decent — has potential but needs significant improvement
  7-8: Good — well-written with only minor issues
  9-10: Excellent — publishable quality

Be specific and actionable in your feedback. Be honest — do NOT inflate scores."""

EXTRACTOR_SYSTEM_PROMPT = """You are a precise feedback analyzer. Extract ALL unique
improvement points from the feedback history below.

You MUST respond in this exact JSON format and nothing else:

{
    "addressed": ["points that have been fixed in later drafts"],
    "pending": ["points that still need work"]
}

Rules:
- Only include points that were explicitly mentioned in the feedback
- Remove duplicates
- If a point was raised and then fixed in a later draft, put it in "addressed"
- Be specific — use the exact language from the feedback where possible
- Do NOT invent or add points that were never mentioned"""


# =============================================================================
# Initialize LLM and Embedding Model
# =============================================================================

llm = ChatAnthropic(model=MODEL, max_tokens=1024)

print("Loading embedding model (first run downloads ~80MB)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.\n")


# =============================================================================
# SQLite Helpers
# =============================================================================

def init_db():
    """Create the messages table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_messages_to_db(session_id: str, messages: list, iteration: int):
    """Archive all current messages to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        conn.execute(
            "INSERT INTO messages (session_id, iteration, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, iteration, role, msg.content, now),
        )
    conn.commit()
    conn.close()


# =============================================================================
# 3-Layer Validation: Keywords → Embeddings → Fallback
# =============================================================================

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "should", "could", "more",
    "very", "also", "been", "have", "has", "was", "were", "are", "its",
    "from", "not", "but", "can", "will", "would", "into", "about",
}


def keyword_check(point: str, original_text: str) -> bool:
    """
    LAYER 1: Fast keyword matching (instant, free).
    Returns True if >= 50% of meaningful keywords from the extracted point
    are found in the original messages.
    """
    keywords = [
        w for w in point.lower().split()
        if len(w) >= 3 and w not in STOPWORDS
    ]
    if not keywords:
        return False
    matches = sum(1 for kw in keywords if kw in original_text.lower())
    return (matches / len(keywords)) >= KEYWORD_THRESHOLD


def embedding_check(point: str, original_messages: list[str]) -> float:
    """
    LAYER 2: Semantic similarity using local embeddings.
    Only called when keyword check fails.
    Returns the best cosine similarity score.
    """
    point_emb = embedding_model.encode([point])
    orig_embs = embedding_model.encode(original_messages)
    similarities = embedding_model.similarity(point_emb, orig_embs)[0]
    return float(similarities.max())


def validate_extraction(
    extracted_points: list[str], original_messages: list[str]
) -> list[str]:
    """
    3-layer validation: keywords first, embeddings if needed.

    Layer 1: Keyword match (>= 50% keywords found) → KEEP
    Layer 2: Embedding similarity (>= 0.7 cosine) → KEEP
    Neither: → DROP the point

    Returns only validated points.
    """
    original_text = " ".join(original_messages)
    validated = []

    for point in extracted_points:
        # Layer 1: Keyword check (fast, free)
        if keyword_check(point, original_text):
            validated.append(point)
            print(f"  [VALID-KW]  '{point[:70]}'")
            continue

        # Layer 2: Embedding check (only if keywords failed)
        best_sim = embedding_check(point, original_messages)
        if best_sim >= EMBEDDING_THRESHOLD:
            validated.append(point)
            print(f"  [VALID-EMB] '{point[:70]}' (similarity: {best_sim:.2f})")
        else:
            print(f"  [DROPPED]   '{point[:70]}' (kw: fail, emb: {best_sim:.2f} < {EMBEDDING_THRESHOLD})")

    return validated


# =============================================================================
# Graph Nodes
# =============================================================================

def generate(state: ReflectionState) -> dict:
    """GENERATE NODE — The "Writer". Writes or rewrites the essay."""
    messages = [SystemMessage(content=WRITER_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    iteration = state.get("iteration", 0)
    print("\n" + "=" * 60)
    if iteration == 0:
        print("  INITIAL DRAFT")
    else:
        print(f"  IMPROVED DRAFT (iteration {iteration})")
    print("=" * 60)
    print(response.content)

    return {"messages": [response]}


def reflect(state: ReflectionState) -> dict:
    """REFLECT NODE — The "Critic". Scores and critiques the essay."""
    essay = state["messages"][-1].content
    iteration = state.get("iteration", 0)

    critic_messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=f"Please critique this essay:\n\n{essay}"),
    ]
    response = llm.invoke(critic_messages)

    try:
        feedback = json.loads(response.content)
    except json.JSONDecodeError:
        feedback = {
            "quality_score": 5,
            "strengths": [],
            "weaknesses": ["Could not parse structured feedback"],
            "suggestions": [response.content],
        }

    score = feedback.get("quality_score", 5)

    print("\n" + "-" * 60)
    print(f"  CRITIQUE — Quality Score: {score}/10")
    print("-" * 60)
    print(f"  Strengths:   {', '.join(feedback.get('strengths', []))}")
    print(f"  Weaknesses:  {', '.join(feedback.get('weaknesses', []))}")
    print(f"  Suggestions: {', '.join(feedback.get('suggestions', []))}")

    feedback_text = (
        f"Quality Score: {score}/10\n\n"
        f"Strengths:\n"
        + "\n".join(f"- {s}" for s in feedback.get("strengths", []))
        + "\n\n"
        f"Weaknesses:\n"
        + "\n".join(f"- {w}" for w in feedback.get("weaknesses", []))
        + "\n\n"
        f"Suggestions for improvement:\n"
        + "\n".join(f"- {s}" for s in feedback.get("suggestions", []))
        + "\n\n"
        f"Please rewrite the essay addressing all the feedback above."
    )

    return {
        "messages": [HumanMessage(content=feedback_text)],
        "iteration": iteration + 1,
    }


def compress(state: ReflectionState) -> dict:
    """
    COMPRESS NODE — Memory management.

    1. Archives ALL raw messages to SQLite (never lose data)
    2. If messages exceed byte threshold:
       a. LLM extracts a checklist of pending improvement points
       b. 3-layer validation (keywords → embeddings → fallback)
       c. Replaces message history with compact validated checklist
    """
    messages = state["messages"]
    session_id = state.get("session_id", "unknown")
    iteration = state.get("iteration", 0)

    # Always archive to SQLite
    save_messages_to_db(session_id, messages, iteration)

    # Calculate total bytes
    total_bytes = sum(len(m.content.encode("utf-8")) for m in messages)

    if total_bytes <= MESSAGE_BYTE_THRESHOLD:
        print(f"\n>>> Messages within threshold ({total_bytes} <= {MESSAGE_BYTE_THRESHOLD} bytes), no compression needed")
        return {}

    print(f"\n>>> Messages exceed threshold ({total_bytes} > {MESSAGE_BYTE_THRESHOLD} bytes), compressing...")

    # Collect original feedback messages (HumanMessages, excluding the first topic)
    original_feedback_texts = [
        m.content for m in messages
        if isinstance(m, HumanMessage) and not m.content.startswith("Write a short essay")
    ]

    if not original_feedback_texts:
        print(">>> No feedback messages to compress, skipping")
        return {}

    # Call LLM to extract structured checklist
    all_feedback = "\n---\n".join(original_feedback_texts)
    extractor_messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract improvement points from this feedback history:\n\n{all_feedback}"),
    ]
    response = llm.invoke(extractor_messages)

    try:
        extraction = json.loads(response.content)
    except json.JSONDecodeError:
        print(">>> Could not parse extraction, falling back to last messages")
        trimmed = messages[-FALLBACK_MESSAGE_COUNT:]
        new_bytes = sum(len(m.content.encode("utf-8")) for m in trimmed)
        print(f">>> Fallback: {total_bytes} → {new_bytes} bytes (last {FALLBACK_MESSAGE_COUNT} messages)")
        return {"messages": trimmed}

    pending_points = extraction.get("pending", [])

    if not pending_points:
        print(">>> No pending points extracted, skipping compression")
        return {}

    # 3-layer validation
    print(f"\n>>> Validating {len(pending_points)} extracted points...")
    validated_points = validate_extraction(pending_points, original_feedback_texts)

    pass_rate = len(validated_points) / len(pending_points) if pending_points else 0

    if pass_rate < VALIDATION_PASS_RATE:
        # LAYER 3 FALLBACK: Too many points failed validation
        # Don't trust the checklist — keep last N raw messages instead
        trimmed = messages[-FALLBACK_MESSAGE_COUNT:]
        new_bytes = sum(len(m.content.encode("utf-8")) for m in trimmed)
        print(
            f"\n>>> Validation failed: only {len(validated_points)}/{len(pending_points)} "
            f"points passed ({pass_rate:.0%} < {VALIDATION_PASS_RATE:.0%})"
        )
        print(f">>> Falling back to last {FALLBACK_MESSAGE_COUNT} messages "
              f"({total_bytes} → {new_bytes} bytes)")
        return {"messages": trimmed}

    # Compression successful — build compact message list
    checklist = "Previous feedback summary (validated):\n" + "\n".join(
        f"- {p}" for p in validated_points
    )

    # Keep: validated checklist + latest essay + latest feedback
    latest_essay = None
    latest_feedback = None
    for msg in reversed(messages):
        if latest_feedback is None and isinstance(msg, HumanMessage):
            latest_feedback = msg
        elif latest_essay is None and isinstance(msg, AIMessage):
            latest_essay = msg
        if latest_essay and latest_feedback:
            break

    new_messages = [HumanMessage(content=checklist)]
    if latest_essay:
        new_messages.append(latest_essay)
    if latest_feedback:
        new_messages.append(latest_feedback)

    new_bytes = sum(len(m.content.encode("utf-8")) for m in new_messages)
    print(
        f"\n>>> Compressed: {total_bytes} → {new_bytes} bytes "
        f"({len(validated_points)}/{len(pending_points)} points passed validation)"
    )

    return {"messages": new_messages}


# =============================================================================
# Conditional Edge
# =============================================================================

def should_continue(state: ReflectionState) -> str:
    """
    HYBRID STOP STRATEGY:
    1. LLM-as-Judge: score >= threshold → stop
    2. Safety net: max iterations → stop
    """
    iteration = state.get("iteration", 0)
    last_message = state["messages"][-1].content

    score = QUALITY_THRESHOLD - 1
    for line in last_message.split("\n"):
        if line.startswith("Quality Score:"):
            try:
                score = int(line.split(":")[1].strip().split("/")[0])
            except (ValueError, IndexError):
                pass
            break

    if score >= QUALITY_THRESHOLD:
        print(f"\n>>> Stopping: Quality score {score} >= threshold {QUALITY_THRESHOLD}")
        return "end"

    if iteration >= MAX_ITERATIONS:
        print(f"\n>>> Stopping: Reached max iterations ({MAX_ITERATIONS})")
        return "end"

    print(f"\n>>> Continuing: Score {score} < {QUALITY_THRESHOLD}, "
          f"iteration {iteration}/{MAX_ITERATIONS}")
    return "compress"


# =============================================================================
# Build the LangGraph
# =============================================================================

def build_reflection_graph():
    """
    Constructs the Reflection graph:

        START → generate → reflect → should_continue?
                   ↑                      │
                   │                     STOP → END
                   │                      │
                   └── compress ← ── CONTINUE
    """
    graph = StateGraph(ReflectionState)

    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)
    graph.add_node("compress", compress)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {"compress": "compress", "end": END},
    )
    graph.add_edge("compress", "generate")

    return graph.compile()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC
    session_id = str(uuid.uuid4())[:8]

    init_db()

    print("\n" + "*" * 60)
    print("  REFLECTION PATTERN DEMO (LangGraph)")
    print(f"  Topic: {topic}")
    print(f"  Session: {session_id}")
    print(f"  Quality threshold: {QUALITY_THRESHOLD}/10")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  Compress when messages > {MESSAGE_BYTE_THRESHOLD} bytes")
    print("*" * 60)

    app = build_reflection_graph()

    initial_input = {
        "messages": [HumanMessage(content=f"Write a short essay about: {topic}")],
        "session_id": session_id,
        "iteration": 0,
    }

    result = app.invoke(initial_input)

    # Print final essay
    for msg in reversed(result["messages"]):
        if hasattr(msg, "type") and msg.type == "ai":
            print("\n" + "=" * 60)
            print("  FINAL ESSAY")
            print("=" * 60)
            print(msg.content)
            break

    print(f"\n>>> Total reflection cycles: {result.get('iteration', 0)}")
    print(f">>> Message history saved to: {DB_PATH}")
    print(f">>> Session ID: {session_id}")
    print(">>> Done!\n")


if __name__ == "__main__":
    main()
