# Architecture

Cancer Output Atlas finds **already-public** cancer research outputs for a reuse goal. Find ranks a baked graph. It does not re-ingest, does not download omics, and does not invent accessions.

![Architecture](architecture.svg)

## Find path

User goal → parse → retrieve baked graph → ODS groups, or abstain.

```mermaid
flowchart LR
  U["User goal"] --> A["Agent tools"]
  A --> P["parse_goal"]
  A --> R["retrieve baked graph"]
  A --> X["refuse / abstain"]
  P --> G["link_graph.json<br/>public metadata only"]
  R --> G
  G --> H{hits?}
  H -->|yes| O["ODS groups<br/>data · software · tool<br/>method · model · trial · biospecimen"]
  H -->|no| X
```

## Live demo vs official Strands SDK

The live Cloud Run process is `serve.py`. It does **not** construct `strands.Agent` on each find. Official Strands lives in `runtime.py` and `agent_tools.py` (`coa agent`). The model provider is pluggable: `gemini` | `strands` | `none`. Keys are environment-only and never committed.

```mermaid
flowchart TB
  subgraph live ["Live demo — serve.py"]
    S["GET /api/find?goal=…"] --> F["find_by_goal"]
    F --> PG["rank.parse_goal"]
    PG -->|"key present: Gemini slot parse<br/>else lexical"| G2["baked graph → ODS or abstain"]
  end

  subgraph sdk ["Official Strands Agents SDK"]
    RT["runtime.py<br/>from strands import Agent, tool"]
    AT["agent_tools.py @tool<br/>refuse · classify · link · rank · digest"]
    CLI["coa agent → strands.Agent"]
    RT --> AT --> CLI
  end
```

See the repository README for clone/run, safety rules, and how the official SDK is wired.

