# Pattern #2: ReAct (Reasoning + Acting)

## What is the ReAct Pattern?

ReAct makes an LLM **think and act** in alternating steps — like a detective who investigates one clue at a time, adapting their next move based on what they just found.

Instead of planning everything upfront OR acting blindly:
1. **Think** — What do I need to find out?
2. **Act** — Call a tool (search, calculate, etc.)
3. **Observe** — Read the tool's result
4. **Repeat** — Until enough info is gathered
5. **Answer** — Only using facts from tool results

## How It Works

```
User Question
      │
      ▼
┌────────────┐         ┌────────────┐
│ AGENT NODE │◄────────│ TOOL NODE  │
│ Think +    │────────►│ Execute    │
│ Decide     │ tool    │ Return     │
└─────┬──────┘ call    └────────────┘
      │
      │ final answer
      ▼
┌─────────────────┐
│ GROUNDING CHECK │
│                 │
│ Are all claims  │
│ backed by tool  │──── YES → END (answer is safe)
│ observations?   │
│                 │──── NO  → Retry (re-verify claims)
└─────────────────┘
```

## What Makes This Different from Pattern #1 (Reflection)?

| Aspect | Reflection | ReAct |
|--------|-----------|-------|
| **Purpose** | Improve output quality | Gather information + answer questions |
| **Loop type** | Generate → Critique → Improve | Think → Act → Observe |
| **Tools** | No tools (self-review only) | Uses external tools (search, calc, APIs) |
| **When to use** | Writing, code generation | Research, Q&A, data gathering |

## The Grounding Check (Anti-Hallucination)

After the LLM gives its final answer, we validate every factual claim:

```
Answer: "Paris has 2.1M people and the Eiffel Tower is 330m tall"

Claim 1: "Paris has 2.1M people"
  → Tool returned: "population of Paris: 2.1 million" ✓ GROUNDED

Claim 2: "Eiffel Tower is 330m tall"
  → No tool ever mentioned this ✗ UNGROUNDED (from training data, not verified)

Score: 1/2 = 50% < 70% threshold → FAIL → Retry
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

## Run

```bash
# Default question (population + weather)
python react_langgraph.py

# Custom question
python react_langgraph.py "What is 847 * 293 and what's the weather in Tokyo?"
```

## Available Tools

| Tool | What it does | Example |
|------|-------------|---------|
| `web_search` | Searches the web via DuckDuckGo | "population of Paris 2024" |
| `calculator` | Evaluates math expressions | "847 * 293" |
| `get_weather` | Gets weather for a city (simulated) | "Paris" |

## What You'll See in the Output

```
>>> THOUGHT + ACTION: Calling web_search({'query': 'capital of France'})
>>> THOUGHT + ACTION: Calling web_search({'query': 'population Paris'})
>>> THOUGHT + ACTION: Calling get_weather({'city': 'Paris'})
>>> THOUGHT: I have enough information. Providing final answer.

GROUNDING CHECK
  Extracted 3 factual claims. Validating...
  [GROUNDED-KW]  'Paris is the capital of France'
  [GROUNDED-KW]  'Paris has approximately 2.1 million residents'
  [GROUNDED-EMB] 'Temperature in Paris is 18°C' (sim: 0.82)
  Grounding Score: 3/3 = 100%
  Result: PASS ✓

FINAL ANSWER (GROUNDED)
```

## Learning Exercises

1. **Ask something tools can't answer** — e.g., "What's the meaning of life?" and see how the agent handles it
2. **Force a grounding failure** — ask a complex question where the LLM is tempted to add info from its training data
3. **Add a new tool** — create a `get_stock_price` tool and ask financial questions
4. **Lower grounding threshold** — set `GROUNDING_THRESHOLD = 0.9` and see more retries
5. **Test parallel tool calls** — ask questions that need multiple tools simultaneously

## Who Uses This in Production?

- **LangChain/LangGraph** — Default agent architecture uses ReAct
- **OpenAI Assistants API** — Built on ReAct with function calling
- **Google (original paper)** — ReAct: Synergizing Reasoning and Acting in LLMs (2022)
- **Anthropic Claude** — Tool use follows the ReAct pattern natively
