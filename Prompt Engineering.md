# Prompt Engineering — Practical Reference

> A concise, code-first guide to the core prompting techniques every LLM engineer should have in their toolkit.

---

## Table of Contents

- [Overview](#overview)
- [1. Zero-Shot Prompting](#1-zero-shot-prompting)
- [2. Few-Shot Prompting](#2-few-shot-prompting)
- [3. System Prompts](#3-system-prompts)
- [4. Chain-of-Thought Reasoning](#4-chain-of-thought-reasoning)
- [5. Structured Output (JSON)](#5-structured-output-json)
- [Practice Exercises](#practice-exercises)
- [Quick Reference](#quick-reference)
- [What to Build Next](#what-to-build-next)

---

## Overview

Prompt engineering is the discipline of designing input instructions that reliably elicit high-quality, predictable outputs from large language models. Mastering these techniques directly impacts model accuracy, latency, cost efficiency, and downstream system reliability.

This guide covers five foundational techniques with working Python examples using the OpenAI SDK.

**Prerequisites**

```bash
pip install openai
export OPENAI_API_KEY="your-key-here"
```

```python
from openai import OpenAI
client = OpenAI()
```

---

## 1. Zero-Shot Prompting

### Concept

Zero-shot prompting sends a task to the model with no prior demonstrations or in-context examples. The model relies entirely on its pretrained knowledge and the clarity of the instruction.

This works well for tasks that are well-defined, unambiguous, and within the model's general training distribution — classification, summarization, translation, and simple extraction.

**When to use it:** Rapid prototyping, low-complexity tasks, latency-critical pipelines where token count matters.

**Limitation:** Performance degrades on niche domains, edge cases, or tasks requiring consistent output formatting without examples.

---

### Prompt

```
Classify the sentiment of the following text. 
Reply with a single word: Positive, Negative, or Neutral.

Text: "I love this product!"
```

---

### Code

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": (
                "Classify the sentiment of the following text.\n"
                "Reply with a single word: Positive, Negative, or Neutral.\n\n"
                'Text: "I love this product!"'
            )
        }
    ]
)

print(response.choices[0].message.content)
# Output: Positive
```

> **Pro tip:** Always constrain the output space explicitly (`"Reply with a single word"`). Without constraints, models tend to over-explain, breaking downstream parsers.

---

## 2. Few-Shot Prompting

### Concept

Few-shot prompting provides `k` labeled input-output demonstrations before the actual query. This technique leverages the model's in-context learning capability — the model infers the task pattern from the examples without any weight updates.

**When to use it:** Domain-specific classification, output format enforcement, tasks where zero-shot performance is inconsistent.

**Key considerations:**
- Example quality matters more than quantity. Noisy or contradictory examples actively hurt performance.
- Typically `k = 3–8` examples is optimal. Diminishing returns beyond 10 in most tasks.
- Maintain consistent formatting across all examples — the model is pattern-matching.

---

### Prompt

```
Classify the sentiment of each text as Positive, Negative, or Neutral.

Text: "I hate this movie."
Sentiment: Negative

Text: "This is absolutely amazing."
Sentiment: Positive

Text: "It is okay, nothing special."
Sentiment: Neutral

Text: "The packaging was fine but the product broke."
Sentiment:
```

---

### Code

```python
FEW_SHOT_PROMPT = """Classify the sentiment of each text as Positive, Negative, or Neutral.

Text: "I hate this movie."
Sentiment: Negative

Text: "This is absolutely amazing."
Sentiment: Positive

Text: "It is okay, nothing special."
Sentiment: Neutral

Text: "The packaging was fine but the product broke."
Sentiment:"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": FEW_SHOT_PROMPT}]
)

print(response.choices[0].message.content)
# Output: Negative
```

> **Pro tip:** For production classifiers, use few-shot examples that represent your edge cases, not just the clean happy-path — this is where the technique earns its value.

---

## 3. System Prompts

### Concept

The `system` role in the OpenAI chat format allows you to establish persistent context, persona, behavioral constraints, and response style before the user turn begins. It is processed with elevated priority relative to the user message.

System prompts are the primary mechanism for:
- Defining the model's role and persona
- Enforcing output style and tone
- Setting hard constraints (`"Never mention competitor products"`)
- Injecting domain context without polluting the user turn

**When to use it:** Any production deployment. Every serious LLM application should have a thoughtfully crafted system prompt.

---

### Prompt Structure

```
[SYSTEM]
You are a senior software engineer conducting a technical code review.
Be direct. Prioritize correctness, then performance, then readability.
Do not suggest refactors unless they address a concrete issue.

[USER]
Review this Python function: ...
```

---

### Code

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a senior software engineer conducting a technical code review. "
                "Be direct. Prioritize correctness, then performance, then readability. "
                "Do not suggest refactors unless they address a concrete issue."
            )
        },
        {
            "role": "user",
            "content": "Review this function:\n\ndef divide(a, b):\n    return a / b"
        }
    ]
)

print(response.choices[0].message.content)
```

> **Pro tip:** Treat the system prompt as your application's contract with the model. Version-control it alongside your code, and A/B test changes systematically — even small edits can cause meaningful behavioral drift.

---

## 4. Chain-of-Thought Reasoning

### Concept

Chain-of-Thought (CoT) prompting elicits explicit intermediate reasoning steps before the model produces a final answer. Originally demonstrated in *Wei et al. (2022)*, CoT consistently improves accuracy on tasks involving multi-step logic, arithmetic, commonsense reasoning, and causal inference.

There are two main variants:

| Variant | Mechanism | Best For |
|---------|-----------|----------|
| **Zero-shot CoT** | Append `"Think step by step."` | General reasoning, quick setup |
| **Few-shot CoT** | Provide full reasoning chains as examples | Domain-specific logic, higher reliability |

**Why it works:** Forcing the model to decompose a problem into intermediate steps reduces the probability of the model committing to an incorrect final answer early and confabulating a justification for it.

---

### Prompt

```
A store sells notebooks for $4.50 each and pens for $1.25 each.
A customer buys 3 notebooks and 5 pens. What is the total cost?

Think step by step.
```

---

### Code

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": (
                "A store sells notebooks for $4.50 each and pens for $1.25 each.\n"
                "A customer buys 3 notebooks and 5 pens. What is the total cost?\n\n"
                "Think step by step."
            )
        }
    ]
)

print(response.choices[0].message.content)
```

**Expected output:**
```
Step 1: Cost of notebooks = 3 × $4.50 = $13.50
Step 2: Cost of pens = 5 × $1.25 = $6.25
Step 3: Total = $13.50 + $6.25 = $19.75

Answer: $19.75
```

> **Pro tip:** For production reasoning tasks, consider combining CoT with **self-consistency** — sample multiple reasoning paths at `temperature > 0` and majority-vote the final answer. This significantly reduces variance on complex problems.

---

## 5. Structured Output (JSON)

### Concept

Structured output constrains the model to return machine-parseable data rather than natural language prose. This is essential in agentic systems, data extraction pipelines, and any workflow where the LLM output feeds downstream code.

OpenAI's `response_format` parameter with `json_object` enforces valid JSON at the decoding level. For stricter schema enforcement, the newer **Structured Outputs** API (`json_schema`) allows you to specify an exact JSON Schema the response must conform to.

**When to use it:** Entity extraction, classification with metadata, API integrations, tool/function call preprocessing, any LLM output consumed programmatically.

---

### Prompt

```
Extract the following fields from the text and return as JSON:
- name (string)
- email (string)  
- phone (string)

Text: "Please contact Sarah Chen at sarah.chen@acme.com or call 415-555-0192."
```

---

### Code — `json_object` mode

```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You are a data extraction assistant. Always respond with valid JSON only."
        },
        {
            "role": "user",
            "content": (
                "Extract the following fields and return as JSON:\n"
                "- name (string)\n- email (string)\n- phone (string)\n\n"
                'Text: "Please contact Sarah Chen at sarah.chen@acme.com or call 415-555-0192."'
            )
        }
    ]
)

import json
data = json.loads(response.choices[0].message.content)
print(data)
# Output: {"name": "Sarah Chen", "email": "sarah.chen@acme.com", "phone": "415-555-0192"}
```

---

### Code — Strict JSON Schema mode (recommended for production)

```python
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "Extract contact: Sarah Chen, sarah.chen@acme.com, 415-555-0192"}
    ],
    response_format=ContactInfo,
)

contact = response.choices[0].message.parsed
print(contact.name)   # Sarah Chen
print(contact.email)  # sarah.chen@acme.com
```

> **Pro tip:** Always parse with a schema in production. `json_object` mode guarantees valid JSON but not your expected structure. A missing field or unexpected type will break your pipeline silently unless you validate.

---

## Practice Exercises

The following exercises are designed to reinforce each technique in realistic engineering contexts.

**Exercise 1 — Zero-Shot Summarization**

Build a function that accepts an arbitrarily long document and returns a 3-sentence executive summary. Constrain output length in the prompt. Handle the case where the input is already shorter than 3 sentences.

**Exercise 2 — Few-Shot Entity Extraction**

Write a few-shot prompt that extracts `{company, role, years_experience}` from unstructured resume text. Test on at least 5 different input formats. Measure accuracy and identify failure modes.

**Exercise 3 — System Prompt Engineering**

Design a customer support assistant with the following constraints: responds only in formal English, never acknowledges competitor products, always ends responses with a satisfaction check question. Test how the behavior degrades as you progressively reduce system prompt verbosity.

**Exercise 4 — Chain-of-Thought Classifier**

Build a content moderation classifier that outputs both a label (`safe` / `unsafe`) and a reasoning trace explaining the classification. Evaluate whether the reasoning trace correlates with classification accuracy.

**Exercise 5 — Structured Extraction Pipeline**

Build an end-to-end pipeline that: (1) accepts raw job description text, (2) extracts structured fields via JSON mode, (3) validates against a Pydantic schema, and (4) writes validated records to a SQLite database.

---

## Quick Reference

| Technique | Core Mechanism | Optimal Use Case | Primary Risk |
|-----------|---------------|------------------|-------------|
| **Zero-shot** | Direct instruction, no examples | Simple, well-defined tasks | Inconsistent formatting |
| **Few-shot** | In-context demonstrations | Format enforcement, domain tasks | Example quality sensitivity |
| **System prompt** | Persistent behavioral framing | All production deployments | Prompt injection vulnerability |
| **Chain-of-Thought** | Explicit reasoning decomposition | Multi-step logic, arithmetic | Verbose output, higher latency |
| **Structured Output** | Schema-constrained decoding | Data pipelines, agentic systems | Schema over-specification |

---

## Composing Techniques

These techniques are not mutually exclusive. Production systems typically combine all five:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            # System prompt — defines role and output contract
            "role": "system",
            "content": "You are a financial analyst. Always respond in valid JSON."
        },
        {
            # Few-shot + CoT in the user turn
            "role": "user",
            "content": """Analyze the following earnings statement.
Think step by step, then return your findings as JSON with keys:
revenue_trend, risk_level, recommendation.

Example:
Input: Revenue up 12% YoY, margins compressed.
Output: {"revenue_trend": "positive", "risk_level": "medium", "recommendation": "hold"}

Now analyze:
Input: Revenue down 8% YoY, operating costs rising, new product launch pending."""
        }
    ]
)
```

---

## What to Build Next

With these fundamentals in place, the natural next steps are:

- **RAG pipeline** — Combine structured extraction with a vector store for retrieval-augmented generation
- **Agentic loop** — Chain CoT + structured output to build a reasoning agent with tool use
- **Evaluation harness** — Use few-shot prompts to build an LLM-as-judge evaluator for your own outputs
- **Fine-tuning dataset** — Curate your best few-shot examples as supervised fine-tuning data for a smaller, faster model

---

## Further Reading

- Wei et al. (2022) — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- Brown et al. (2020) — *Language Models are Few-Shot Learners* (GPT-3 paper)
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Anthropic Prompt Engineering Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [DAIR.AI Prompt Engineering Guide](https://www.promptingguide.ai)

---

*Contributions welcome. Open a PR with reproducible examples and benchmark results.*
