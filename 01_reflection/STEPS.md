# Reflection Pattern — Step-by-Step Breakdown

## Overview

The Reflection pattern makes an LLM improve its own output through a loop of
writing, critiquing, and rewriting — with smart memory management.

---

## Step 1: User Provides a Topic

```
Input: "Write a short essay about: benefits of open source software"
```

This becomes the first `HumanMessage` in the state.

```python
state["messages"] = [HumanMessage("Write a short essay about: benefits of open source")]
state["session_id"] = "a1b2c3d4"
state["iteration"] = 0
```

---

## Step 2: GENERATE Node (The Writer)

The Writer LLM receives the messages + a writer system prompt and produces an essay.

```
System Prompt: "You are a skilled essay writer..."

Input:  state["messages"]  (the topic)
Output: AIMessage(essay_v1)
```

```python
state["messages"] = [
    HumanMessage("Write essay about..."),   # original topic
    AIMessage("Open source software...")     # draft v1
]
```

**First call** → writes from scratch.
**Later calls** → rewrites based on feedback from the critic.

---

## Step 3: REFLECT Node (The Critic)

The Critic LLM reads the latest essay and returns structured JSON feedback.

```
System Prompt: "You are a constructive writing critic..."

Input:  The latest essay only (NOT full history — keeps critic focused)
Output: JSON with quality_score + strengths + weaknesses + suggestions
```

```json
{
    "quality_score": 5,
    "strengths": ["Good topic coverage", "Clear language"],
    "weaknesses": ["Opening is generic", "No real-world examples"],
    "suggestions": ["Add a compelling hook", "Include 2-3 specific examples"]
}
```

The feedback is formatted and returned as a `HumanMessage` — this is the
**key trick**. The Generator sees it as "user feedback" and naturally responds to it.

```python
state["messages"] = [
    HumanMessage("Write essay about..."),
    AIMessage("Open source software..."),        # essay v1
    HumanMessage("Quality Score: 5/10\n...")      # formatted feedback
]
state["iteration"] = 1  # incremented by reflect node
```

---

## Step 4: SHOULD_CONTINUE — The Decision

Checks two conditions (Hybrid Stop Strategy):

```
Condition 1: Is quality_score >= 7?
  YES → STOP (essay is good enough)

Condition 2: Has iteration >= 5?
  YES → STOP (safety net — prevent infinite loops + API cost explosion)

Neither → CONTINUE (loop back for another round)
```

**If STOP** → go to END, output the final essay.
**If CONTINUE** → go to COMPRESS node.

---

## Step 5: COMPRESS Node (Memory Management)

### Step 5a: Archive to SQLite (always)

Every message is saved to `reflection_history.db` — you never lose data.

```sql
INSERT INTO messages (session_id, iteration, role, content, timestamp)
VALUES ('a1b2c3d4', 1, 'human', 'Quality Score: 5/10...', '2026-04-19T...');
```

### Step 5b: Check byte size

```python
total_bytes = sum(len(m.content.encode("utf-8")) for m in messages)

if total_bytes <= 2000:
    # No compression needed, pass through to GENERATE
else:
    # Proceed to extraction + validation (Step 5c)
```

### Step 5c: LLM Extracts Checklist

The LLM reads ALL feedback messages and extracts a structured checklist.

```
System Prompt: "Extract ALL unique improvement points..."

Input:  All feedback messages joined together
Output: JSON with "addressed" and "pending" lists
```

```json
{
    "addressed": ["Added statistics (fixed in draft v2)"],
    "pending": ["Opening paragraph still generic", "Needs stronger conclusion"]
}
```

**Why structured extraction, not freeform summary?**
- Checklists are verifiable (you can see exactly what was extracted)
- Addressed items get removed (no contradictions like "add more" + "too many")
- Generator gets clear TODO items, not a vague paragraph

### Step 5d: 3-Layer Validation

Each extracted "pending" point is validated against the original messages
to make sure it's real (not hallucinated by the extractor LLM).

```
LAYER 1: KEYWORD CHECK (instant, free)
─────────────────────────────────────────
For each extracted point:
  1. Split into keywords (skip stopwords like "the", "and", "should")
  2. Check what % of keywords appear in the original feedback text
  3. If >= 50% match → KEEP ✓ (skip Layer 2)
     If < 50% match  → go to Layer 2

Example:
  Extracted:  "Fix the statistics in paragraph 2"
  Keywords:   ["fix", "statistics", "paragraph"]
  Original:   "...the statistics in paragraph 2 are outdated..."
  Matches:    3/3 = 100% → KEEP ✓
  Output:     [VALID-KW] 'Fix the statistics in paragraph 2'


LAYER 2: EMBEDDING CHECK (local model, no API cost)
─────────────────────────────────────────
Only runs for points that failed keyword check.
  1. Convert extracted point → vector (numbers) using local model
  2. Convert each original message → vector
  3. Calculate cosine similarity (how close in meaning)
  4. If best similarity >= 0.7 → KEEP ✓
     If best similarity < 0.7  → DROP ✗ (hallucinated)

Example:
  Extracted:  "Introduction needs a stronger attention grabber"
  Original:   "The opening paragraph lacks a compelling hook"
  Keywords:   0% match (different words!)
  Embedding:  similarity = 0.87 → KEEP ✓ (same meaning!)
  Output:     [VALID-EMB] 'Introduction needs...' (similarity: 0.87)

Example (hallucination caught):
  Extracted:  "Add a bibliography section with citations"
  Original:   (never mentioned anywhere in any feedback)
  Keywords:   0% match
  Embedding:  similarity = 0.23 → DROP ✗
  Output:     [DROPPED] 'Add a bibliography...' (kw: fail, emb: 0.23 < 0.7)


LAYER 3: FALLBACK (if too many points fail)
─────────────────────────────────────────
After all points are checked:
  If >50% of points passed validation:
    → Use the validated checklist (compressed history)

  If >50% of points FAILED:
    → Don't trust the checklist at all
    → Just keep the last 4 raw messages (last 2 rounds)
    → Safe, simple, no bad data risk
```

### Step 5e: Replace Messages with Compact History

```
BEFORE compression:
  state["messages"] = [
    HumanMessage("Write essay..."),              # original topic
    AIMessage("Open source software..."),         # essay v1
    HumanMessage("Quality Score: 5/10\n..."),     # feedback 1
    AIMessage("Open source has transformed..."),  # essay v2
    HumanMessage("Quality Score: 6/10\n..."),     # feedback 2
  ]
  Total: ~4500 bytes

AFTER compression:
  state["messages"] = [
    HumanMessage("Previous feedback summary:\n   # validated checklist
      - Opening paragraph still generic\n
      - Needs stronger conclusion"),
    AIMessage("Open source has transformed..."),  # latest essay (full)
    HumanMessage("Quality Score: 6/10\n..."),     # latest feedback (full)
  ]
  Total: ~1800 bytes
```

---

## Step 6: Back to GENERATE (Improved)

The Generator now receives:
1. A compact, validated checklist of what still needs fixing
2. The latest essay in full
3. The latest feedback in full

It writes an improved essay addressing all the pending points.

---

## Step 7: The Loop Repeats

```
GENERATE → REFLECT → SHOULD_CONTINUE? → COMPRESS → GENERATE → ...
```

Each cycle:
- The essay gets better
- The score goes up
- Old messages get compressed
- New messages get archived to SQLite

Until: score >= 7 OR 5 iterations reached → **END**

---

## Step 8: Final Output

```
FINAL ESSAY: [The polished, improved essay]

Total reflection cycles: 3
Message history saved to: reflection_history.db
Session ID: a1b2c3d4
```

---

## Complete Data Flow Diagram

```
User Topic
    │
    ▼
┌─────────┐    messages = [topic]
│ GENERATE │───────────────────────────────────┐
└────┬────┘                                    │
     │ adds AIMessage(essay)                   │
     ▼                                         │
┌─────────┐    scores essay, returns feedback   │
│ REFLECT  │                                    │
└────┬────┘                                    │
     │ adds HumanMessage(feedback)             │
     │ increments iteration                    │
     ▼                                         │
┌──────────────┐                               │
│SHOULD_CONTINUE│                               │
└──┬────────┬──┘                               │
   │        │                                  │
 STOP    CONTINUE                              │
   │        │                                  │
   ▼        ▼                                  │
 [END]  ┌─────────┐                            │
        │COMPRESS  │                            │
        │          │                            │
        │ 1. Save to SQLite                    │
        │ 2. Check byte size                   │
        │ 3. Extract checklist (LLM)           │
        │ 4. Validate:                         │
        │    Layer 1: Keywords (50%)           │
        │    Layer 2: Embeddings (0.7)         │
        │    Layer 3: Fallback (last 4 msgs)   │
        │ 5. Compact messages                  │
        └────┬────┘                            │
             │                                 │
             └─────────────────────────────────┘
```

---

## Configuration Reference

| Setting | Default | What it does |
|---------|---------|-------------|
| `MODEL` | claude-sonnet-4-20250514 | LLM model for writing + critiquing |
| `MAX_ITERATIONS` | 5 | Safety net — max loops before forced stop |
| `QUALITY_THRESHOLD` | 7 | Score needed to stop (1-10 scale) |
| `MESSAGE_BYTE_THRESHOLD` | 2000 | Compress when messages exceed this size |
| `KEYWORD_THRESHOLD` | 0.5 | Layer 1: % of keywords that must match |
| `EMBEDDING_THRESHOLD` | 0.7 | Layer 2: cosine similarity minimum |
| `VALIDATION_PASS_RATE` | 0.5 | % of points that must pass to use checklist |
| `FALLBACK_MESSAGE_COUNT` | 4 | Layer 3: keep last N messages if validation fails |
| `DB_PATH` | reflection_history.db | SQLite file for message archive |
