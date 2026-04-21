# ReAct Pattern — Step-by-Step Breakdown

## Overview

ReAct (Reasoning + Acting) allows an LLM to think step-by-step and use tools
to gather information, then validates that the final answer only contains
facts from tool results (no hallucinations).

---

## Step 1: User Asks a Question

```
Input: "What is the population of the capital of France and what's the weather there?"
```

This becomes the first `HumanMessage` in the state.

---

## Step 2: AGENT NODE — Think + Decide

The LLM receives the question + tool descriptions and decides what to do.

```
System: "You are a research assistant with access to: web_search, calculator, get_weather"

LLM thinks: "I need to find the capital of France first, then its population,
             then check the weather there."

LLM decides: Call web_search("capital of France population")
```

The LLM uses **function calling** (not text output) — it returns a structured tool call:
```json
{"tool": "web_search", "args": {"query": "capital of France population"}}
```

---

## Step 3: TOOL NODE — Execute

LangGraph's `ToolNode` executes the requested tool and returns the result as a `ToolMessage`.

```
Tool: web_search("capital of France population")
Result: "Paris is the capital of France with a population of approximately 2.1 million
         in the city proper and 12.4 million in the metropolitan area."
```

This observation is added to the state messages.

---

## Step 4: Back to AGENT — Observe + Think Again

```
LLM sees: The tool returned info about Paris population.
LLM thinks: "Good, I have population data. Now I need weather for Paris."
LLM decides: Call get_weather("Paris")
```

---

## Step 5: TOOL NODE Again

```
Tool: get_weather("Paris")
Result: "Paris: 18°C, partly cloudy, humidity 65%, wind 12 km/h NW"
```

---

## Step 6: AGENT — Enough Info, Give Final Answer

```
LLM sees: I have population AND weather for Paris.
LLM thinks: "I have all the information needed to answer."
LLM output: "Paris, the capital of France, has approximately 2.1 million
             residents. The current weather is 18°C and partly cloudy
             with 65% humidity."
```

Since this is a text response (no tool calls), the flow goes to **Grounding Check**.

---

## Step 7: GROUNDING CHECK — Validate the Answer

### Step 7a: Extract Factual Claims

The LLM extracts claims from the final answer:
```json
[
    "Paris is the capital of France",
    "Paris has approximately 2.1 million residents",
    "The current temperature in Paris is 18°C",
    "The weather is partly cloudy",
    "Humidity is 65%"
]
```

### Step 7b: Collect All Tool Observations

```
Observation 1: "Paris is the capital of France with a population of
                approximately 2.1 million..."
Observation 2: "Paris: 18°C, partly cloudy, humidity 65%, wind 12 km/h NW"
```

### Step 7c: Validate Each Claim (Keywords → Embeddings)

```
Claim: "Paris is the capital of France"
  Layer 1 (Keywords): "paris" ✓ "capital" ✓ "france" ✓ → 100% match
  Result: [GROUNDED-KW] ✓

Claim: "Paris has approximately 2.1 million residents"
  Layer 1 (Keywords): "paris" ✓ "2.1" ✓ "million" ✓ → match
  Result: [GROUNDED-KW] ✓

Claim: "The current temperature in Paris is 18°C"
  Layer 1 (Keywords): "temperature" ✗ "paris" ✓ "18°c" ✓ → borderline
  Layer 2 (Embeddings): similarity with "Paris: 18°C, partly cloudy" = 0.83
  Result: [GROUNDED-EMB] ✓

Claim: "The weather is partly cloudy"
  Layer 1 (Keywords): "partly" ✓ "cloudy" ✓ → match
  Result: [GROUNDED-KW] ✓

Claim: "Humidity is 65%"
  Layer 1 (Keywords): "humidity" ✓ "65%" ✓ → match
  Result: [GROUNDED-KW] ✓
```

### Step 7d: Calculate Score

```
Grounded: 5/5 = 100% ≥ 70% threshold
Result: PASS ✓ → Go to END
```

---

## Step 8: What Happens When Grounding FAILS

Example — LLM adds info from its training data:

```
Final Answer: "Paris has 2.1 million people. The Eiffel Tower,
              built in 1889, is the city's most famous landmark."

Claims extracted:
  1. "Paris has 2.1 million people" → GROUNDED (tool confirmed)
  2. "Eiffel Tower built in 1889" → UNGROUNDED (no tool mentioned this)
  3. "Eiffel Tower is city's most famous landmark" → UNGROUNDED

Score: 1/3 = 33% < 70% → FAIL ✗
```

The agent receives feedback:
```
"GROUNDING CHECK FAILED (score: 33%, need: 70%).
The following claims are NOT supported by any tool result:
- Eiffel Tower built in 1889
- Eiffel Tower is city's most famous landmark

Please either:
1. Use a tool to verify these claims, OR
2. Remove them from your answer."
```

The agent then either:
- Searches for "Eiffel Tower facts" to verify, OR
- Removes those claims and gives a shorter, grounded answer

---

## Complete Data Flow

```
User Question
    │
    ▼
┌─────────┐     ┌─────────┐
│  AGENT  │────►│  TOOLS  │
│  Think  │◄────│ Execute │
│  Decide │     │ Return  │
└────┬────┘     └─────────┘
     │            (loops until agent has enough info)
     │
     │ final answer (no tool calls)
     ▼
┌──────────────────┐
│ GROUNDING CHECK  │
│                  │
│ 1. Extract claims│
│ 2. Keyword check │──── match ≥50%? → GROUNDED
│ 3. Embedding     │──── sim ≥0.7?   → GROUNDED
│    check         │──── both fail?  → UNGROUNDED
│ 4. Score claims  │
│                  │
│ Score ≥ 70%? ────┼──── YES → END (safe answer)
│                  │
│ Score < 70%? ────┼──── NO + retries left → AGENT (retry)
│                  │──── NO + max retries  → END (accept as-is)
└──────────────────┘
```

---

## Safety Nets

| Safety Net | What it prevents |
|-----------|-----------------|
| `MAX_ITERATIONS = 10` | Infinite tool-calling loop |
| `MAX_RETRIES = 2` | Infinite grounding check loop |
| `GROUNDING_THRESHOLD = 0.7` | Answers with >30% ungrounded claims |
| `recursion_limit` | LangGraph-level safety against infinite graphs |
| Sandboxed calculator | Code injection via math expressions |

---

## Configuration Reference

| Setting | Default | What it does |
|---------|---------|-------------|
| `MODEL` | claude-sonnet-4-20250514 | LLM model for reasoning |
| `MAX_ITERATIONS` | 10 | Max tool calls before forced stop |
| `GROUNDING_THRESHOLD` | 0.7 | % of claims that must be grounded |
| `MAX_RETRIES` | 2 | Max retries after grounding failure |
| `KEYWORD_THRESHOLD` | 0.5 | Layer 1: % keywords that must match |
| `EMBEDDING_THRESHOLD` | 0.7 | Layer 2: cosine similarity minimum |

---

## Comparison: ReAct vs Reflection

| | Reflection (Pattern #1) | ReAct (Pattern #2) |
|-|------------------------|-------------------|
| **Goal** | Improve quality of generated content | Answer questions using external data |
| **Loop** | Write → Critique → Rewrite | Think → Act → Observe |
| **Tools** | None (self-improvement only) | Web search, calculator, APIs |
| **Validation** | Quality score (1-10) | Grounding check (claims vs observations) |
| **When to use** | Essay writing, code gen, content creation | Research, Q&A, data retrieval |
| **Risk** | Low quality output | Hallucinated facts in answer |
| **Safety** | Score threshold + max iterations | Grounding validation + max retries |
