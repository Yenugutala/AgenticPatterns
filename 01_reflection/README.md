# Pattern #1: Reflection

## What is the Reflection Pattern?

The Reflection pattern is like a writer who drafts, self-reviews, and rewrites — in a loop.

Instead of generating one response and hoping it's good, the LLM:
1. **Generates** an initial output (the "Writer" role)
2. **Critiques** its own output (the "Critic" role)
3. **Rewrites** based on the critique
4. **Repeats** until quality is good enough or max iterations hit

The same LLM plays both roles — just with different system prompts.

## How It Works (LangGraph)

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
   [Generate Node]  Writer writes/rewrites      │
         │                                      │
         ▼                                      │
   [Reflect Node]   Critic scores + feedback    │
         │                                      │
         ▼                                      │
   [should_continue?]                           │
         │          │                           │
       STOP       LOOP                          │
         │          │                           │
         ▼          ▼                           │
       [END]  [Compress Node]                   │
               │  1. Save messages to SQLite    │
               │  2. If messages > 2000 bytes:  │
               │     Extract checklist          │
               │     Validate (KW → Embeddings) │
               │     Compact the history        │
               └────────────────────────────────┘
```

## Key Features

### 1. Hybrid Stop Strategy
- **LLM-as-Judge**: The critic scores output 1-10. Score >= 7 means "good enough"
- **Hard max safety net**: Never loop more than 5 times (prevents infinite loops)

### 2. Memory Compression (Compress Node)
As messages grow, the compress node:
- Archives ALL raw messages to **SQLite** (never lose data)
- Extracts a structured **checklist** of pending improvement points
- Validates each point using **3-layer validation**
- Replaces full history with a compact [checklist + latest essay + latest feedback]

### 3. Three-Layer Validation
Prevents hallucinated or distorted points from entering the compressed checklist:

```
Layer 1: KEYWORDS (instant, free)
  Check if >= 50% of keywords exist in original → KEEP
  Otherwise → go to Layer 2

Layer 2: EMBEDDINGS (local model, no API cost)
  Cosine similarity >= 0.7 → KEEP (same meaning, different words)
  Otherwise → DROP (hallucinated)

Layer 3: FALLBACK (if >50% points fail both layers)
  Skip checklist entirely → keep last 4 raw messages
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your API key:
```bash
export ANTHROPIC_API_KEY=your-key-here
```

Note: The embedding model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first run.

## Run

```bash
# Default topic
python reflection_langgraph.py

# Custom topic
python reflection_langgraph.py "the future of renewable energy"
```

## What You'll See in the Output

```
INITIAL DRAFT              ← First attempt
CRITIQUE (Score: 5/10)     ← Critic finds issues
>>> Compressing...         ← Memory management kicks in
  [VALID-KW]  'point 1'   ← Keyword validation passed
  [VALID-EMB] 'point 2'   ← Embedding validation passed
  [DROPPED]   'point 3'   ← Hallucinated point removed
IMPROVED DRAFT             ← Writer fixes validated issues
CRITIQUE (Score: 7/10)     ← Critic approves
FINAL ESSAY                ← The polished result
```

## Inspect Message History

After running, check the SQLite database:
```bash
sqlite3 reflection_history.db "SELECT iteration, role, substr(content,1,80) FROM messages ORDER BY iteration;"
```

## Learning Exercises

1. **Change the topic** — Try different subjects and see how the critique adapts
2. **Lower the threshold** — Set `QUALITY_THRESHOLD = 9` and watch it loop more times
3. **Force compression** — Set `MESSAGE_BYTE_THRESHOLD = 500` to trigger compression every cycle
4. **Modify the critic prompt** — Make the critic focus on humor, or technical accuracy
5. **Tune validation** — Change `KEYWORD_THRESHOLD` or `EMBEDDING_THRESHOLD` and observe the effect

## Who Uses This in Production?

- **Microsoft (AutoGen)**: Uses reflection for code generation and validation
- **LangChain**: Official `langgraph-reflection` library built on this pattern
- **Reflexion paper**: Achieved 91% on HumanEval (vs 80% baseline) using reflection
