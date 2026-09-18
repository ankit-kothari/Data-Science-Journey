# Upsell Recommendations with AI Agents

Agentic recommendation system using LangGraph — an LLM-powered multi-step pipeline that analyzes a product, identifies higher-end alternatives, and generates a structured upsell recommendation.

## Objective

Move beyond static recommendation rules: use an autonomous agent workflow where each step (attribute extraction → alternative identification → comparison → recommendation synthesis) is an LLM-driven node with typed state and validated outputs.

## Architecture

```
Input Product → [Extract Attributes] → [Find Alternatives] → [Compare Features] → [Synthesize Recommendation]
                    (GPT-4o)               (GPT-4o)              (GPT-4o)              (GPT-4o)
                        ↓                      ↓                     ↓                      ↓
                   Pydantic Model         Pydantic Model        Pydantic Model         Final Output
```

## Key Techniques

- **LangGraph** state machine with `TypedDict` for workflow orchestration
- **Pydantic** structured output validation at each agent step
- **Multi-node agentic pipeline** — each node is a specialized LLM call with defined input/output schema
- **Tool use** — agent decides which product attributes matter for the upsell comparison

## Files

| File | Description |
|---|---|
| `upsell_recommendation_using_langraph.ipynb` | Full implementation with worked examples |
| `upsell_recommendations.pdf` | Sample outputs and architecture writeup |
