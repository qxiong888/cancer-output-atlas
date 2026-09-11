# Architecture

Cancer Output Atlas is an agent-managed living graph of public cancer research outputs, updated daily. Find ranks the current baked snapshot. Honesty rules (no invented accessions, abstain when empty, no on-click ingest, no omics download) live in the repository README [Safety](../README.md#safety) section.

![Architecture](architecture.svg)

## Find path

User goal → Strands agent (`find_public_outputs`) → parse → retrieve baked graph → ODS groups, or abstain.

```mermaid
flowchart LR
  U["User goal"] --> A["Strands Agent<br/>find_public_outputs"]
  A --> P["parse_goal"]
  A --> R["retrieve baked graph"]
  A --> X["refuse / abstain"]
  P --> G["link_graph.json<br/>daily-refreshed snapshot"]
  R --> G
  G --> H{hits?}
  H -->|yes| O["ODS groups<br/>data · software · tool<br/>method · model · trial · biospecimen"]
  H -->|no| X
```

## Live demo vs official Strands SDK

The live Cloud Run process is `serve.py`. When `strands-agents` is installed, each `GET /api/find` constructs an official `strands.Agent` via `build_find_agent` with a `find_public_outputs` tool wrapping `find_by_goal`. Official Strands also lives in `runtime.py` and `agent_tools.py` (`coa agent`). The model provider is pluggable: `gemini` | `strands` | `none`. Keys are environment-only and never committed.

```mermaid
flowchart TB
  subgraph live ["Live demo — serve.py"]
    S["GET /api/find?goal=…"] --> AG["build_find_agent<br/>strands.Agent + find_public_outputs"]
    AG --> F["find_by_goal"]
    F --> PG["rank.parse_goal"]
    PG -->|"key present: Gemini slot parse<br/>else lexical"| G2["baked graph → ODS or abstain"]
  end

  subgraph sdk ["Official Strands Agents SDK"]
    RT["runtime.py<br/>from strands import Agent, tool"]
    AT["agent_tools.py @tool<br/>find_public_outputs + helpers"]
    CLI["coa agent → strands.Agent"]
    RT --> AT
    AT --> AG
    AT --> CLI
  end
```

See the repository README for clone/run, Safety, and how judges see Strands.
