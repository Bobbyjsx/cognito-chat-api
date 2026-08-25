# Smart Model Router Architecture & Specification

## 1. Overview

The **Smart Model Router** is an intelligent, deterministic routing subsystem that automatically selects the optimal AI model for a user's request. It combines:
1. **Lightweight Semantic Request Analysis** (using Gemini Flash-Lite for low-latency JSON classification, with an instant heuristic fallback).
2. **Hard-Constraint Candidate Filtering** (capabilities, context limits, operational status, user plan entitlements).
3. **Deterministic Multi-Dimensional Scoring Engine** (complexity matching curve, capability match, speed, cost, quality, and reliability).
4. **Configurable Routing Policies** (`fast`, `balanced`, `quality`, and custom weights).
5. **Provider-Agnostic Model Metadata Catalog** (the existing model repository and Firestore `AppConfigDB` remain the single source of truth).
6. **Provider Fallback & Resilience** (ranked candidate fallback cascade).
7. **Transparent Explainability & Telemetry** (structured telemetry emission for all decisions without storing sensitive prompts).

---

## 2. System Architecture

```mermaid
flowchart TD
    UserReq[User Request + Context] --> Analyzer[Request Analyzer<br/>Gemini Flash-Lite / Heuristic Fallback]
    Analyzer --> Analysis[RequestAnalysis Signals<br/>task_type, complexity, capabilities]
    
    ModelCatalog[(Canonical Models Repository<br/>AppConfigDB)] --> Filter[Candidate Filter]
    Analysis --> Filter
    Policy[Active Policy<br/>fast | balanced | quality] --> Filter
    
    Filter --> Eligible[Eligible Candidate Models]
    Filter --> FilteredOut[Filtered Models + Reasons]
    
    Eligible --> Scorer[Deterministic Scoring Engine]
    Analysis --> Scorer
    Policy --> Scorer
    
    Scorer --> Ranked[Ranked Scored Candidates<br/>+ ScoreBreakdown]
    Ranked --> Decision[Routing Decision<br/>Selected Model + Fallbacks]
    
    Decision --> Orchestrator[Agent Orchestrator Execution]
    Decision --> Telemetry[Telemetry Emitter<br/>Logging / In-Memory / APM]
```

---

## 3. Metadata & Model Configuration

The existing `TextModelConfig` in [`app/models/config.py`](file:///Users/bobby/Documents/Workspace/project-cognito/cognito-chat-api/app/models/config.py) has been extended with normalized routing metadata:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `complexity_score` | `float [0.0 - 1.0]` | Model capacity to handle intricate, multi-step tasks |
| `reasoning_score` | `float [0.0 - 1.0]` | Analytical and deep reasoning capability |
| `coding_score` | `float [0.0 - 1.0]` | Coding, refactoring, and debugging capability |
| `creative_score` | `float [0.0 - 1.0]` | Stylistic and creative generation capability |
| `context_score` | `float [0.0 - 1.0]` | Long context comprehension and retrieval |
| `vision_score` | `float [0.0 - 1.0]` | Visual understanding and multimodal reasoning |
| `tool_calling_score` | `float [0.0 - 1.0]` | Precision in tool/function calling |
| `speed_score` | `float [0.0 - 1.0]` | Model throughput and response latency (1.0 = fastest) |
| `quality_score` | `float [0.0 - 1.0]` | Overall generation fidelity and output polish |
| `input_cost_per_million` | `float` | Price in USD per 1M input tokens |
| `output_cost_per_million` | `float` | Price in USD per 1M output tokens |
| `context_window_tokens` | `int` | Maximum context limit in tokens |
| `supports_vision` | `bool` | Image and PDF document understanding |
| `supports_tools` | `bool` | Function calling / search tools |
| `supports_structured_output` | `bool` | JSON schema validation |
| `provider` | `str` | Provider identifier (e.g., `"google"`) |
| `status` | `str` | Operational status (`"active"`, `"disabled"`, etc.) |

---

## 4. Request Analysis & Classification

### 4.1 Schema (`RequestAnalysis`)
The analyzer extracts structured routing signals:
- `task_type`: Categorical (`coding`, `general_knowledge`, `creative_writing`, `analysis`, `summarization`, `math_reasoning`, `conversation`, `other`).
- `complexity`: `float (0.0 - 1.0)`
- `reasoning_required`: `float (0.0 - 1.0)`
- `coding_required`: `float (0.0 - 1.0)`
- `creative_required`: `float (0.0 - 1.0)`
- `context_required`: `float (0.0 - 1.0)`
- `vision_required`: `bool`
- `tool_calling_required`: `bool`
- `web_required`: `bool`
- `structured_output_required`: `bool`

### 4.2 Resilient Multi-Tier Analyzer
- **Primary Analyzer (`GeminiFlashLiteAnalyzer`)**: Uses `gemini-3.1-flash-lite` with structured output JSON schema and a concise system prompt to analyze the request in ~150-300ms without solving the prompt.
- **Deterministic Heuristic Fallback (`HeuristicFallbackAnalyzer`)**: Instant (0ms) regex and keyword-based analyzer.
- **Composite Analyzer (`CompositeRequestAnalyzer`)**: Automatically falls back to heuristics if the primary analyzer encounters timeouts, quota limits, or network errors, guaranteeing that chat generation is **never** blocked.

---

## 5. Candidate Filtering

Before scoring, models undergo hard-constraint elimination:
1. **Operational Status**: Model must be `enabled=True` and `status="active"`.
2. **User Entitlement / Policy Allowed Models**: Model must be permitted in `policy.allowed_models`.
3. **Provider Restrictions**: Model must match `policy.preferred_providers` if specified.
4. **Modality & Tools**:
   - Vision required $\implies$ `supports_vision=True`
   - Tool calling / web search required $\implies$ `supports_tools=True`
   - Structured JSON required $\implies$ `supports_structured_output=True`
5. **Context Window**: Request token count $\le$ `model.context_window_tokens`.
6. **Budget Ceiling**: `model.input_cost_per_million` $\le$ `policy.max_cost`.

Models excluded during this phase are tracked with precise rejection reasons for telemetry and diagnostics.

---

## 6. Deterministic Scoring Model

Eligible candidates are scored across multiple dimensions:

$$\text{Total Score} = \sum (\text{Dimension Score} \times \text{Policy Weight})$$

### 6.1 Complexity Matching Curve
The router prefers the **lowest-cost right-sized model** that comfortably satisfies the task complexity:

$$\Delta = \text{Model Complexity} - \text{Request Complexity}$$

- **Sufficient Capacity ($\Delta \ge 0$)**:
  $$\text{Score} = \max(0.50, 1.0 - 0.60 \times \Delta)$$
  *Gently discounts excessive over-provisioning for trivial tasks.*
- **Capacity Deficit ($\Delta < 0$)**:
  $$\text{Score} = \max(0.0, 1.0 - 3.5 \times |\Delta|)$$
  *Steep penalty ensures under-capable models are not chosen for hard reasoning tasks.*

### 6.2 Capability Matching
Measures alignment between request task requirements ($\vec{R}$) and model capabilities ($\vec{M}$):
$$\text{Match} = 1.0 - 2.0 \times \max(0, R_i - M_i)$$

Weighted across coding, reasoning, creative, vision, and tool dimensions.

### 6.3 Cost Normalization
Effective model cost is computed as:
$$\text{Cost}_{\text{eff}} = \text{Price}_{\text{in}} + 2.0 \times \text{Price}_{\text{out}}$$

Normalized across the candidate pool:
$$\text{Cost Score} = 1.0 - 0.85 \times \left( \frac{\text{Cost}_{\text{eff}} - \text{Cost}_{\min}}{\text{Cost}_{\max} - \text{Cost}_{\min}} \right)$$

---

## 7. Routing Policies

| Policy Mode | Description | Weights (Cap / Comp / Qual / Speed / Cost) |
| :--- | :--- | :--- |
| `fast` | Prioritizes minimal latency and lowest cost | $0.15 \ / \ 0.15 \ / \ 0.05 \ / \ 0.35 \ / \ 0.30$ |
| `balanced` *(default)* | Best compromise across all factors | $0.25 \ / \ 0.25 \ / \ 0.20 \ / \ 0.15 \ / \ 0.15$ |
| `quality` | Prioritizes peak reasoning, coding fidelity, and quality | $0.30 \ / \ 0.25 \ / \ 0.35 \ / \ 0.05 \ / \ 0.05$ |

---

## 8. Failover & Resilience

1. When a routing decision is generated, it returns both the top-ranked `selected_model_id` and a prioritized list of `fallback_models`.
2. The orchestrator [`AgentService`](file:///Users/bobby/Documents/Workspace/project-cognito/cognito-chat-api/app/services/chats.py) attempts generation with the primary candidate.
3. If temporary provider errors, rate limits, or context overflow occur, the orchestrator seamlessly attempts the next candidate in the ranked fallback list before reporting a failure.

---

## 9. Observability & Telemetry

Every routing invocation emits a structured `RoutingTelemetry` payload:
- `request_id`: Unique routing decision UUID.
- `timestamp`: UTC timestamp.
- `selected_model` & `provider`: Chosen model and provider.
- `routing_mode`, `task_type`, `complexity`, `score`: Signals and evaluation score.
- `candidate_models`: Ranked candidate IDs.
- `filtered_out_models`: Excluded model IDs paired with rejection reasons.
- `analysis_latency_ms`, `scoring_latency_ms`, `total_latency_ms`: High-precision timing metrics.
- `is_fallback`: Whether fallback analyzers were invoked.

> [!NOTE]
> Sensitive user prompts are never included in the telemetry payload, preserving data privacy and security.
