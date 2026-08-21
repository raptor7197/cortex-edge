# CortexEdge — Detailed Project Plan

> **CortexEdge: Quality-Constrained and Resource-Aware Adaptive Orchestration of Large Language Models Across Mobile, Edge and Cloud Platforms**

Adaptive hybrid edge-AI runtime for LLMs on Raspberry Pi 5 and Android. The core research contribution is a **quality-constrained, resource-aware routing system** that selects the least expensive execution route (small local model / medium local model / local RAG / cloud fallback) capable of meeting a required response-quality target.

---

## 1. Goals & Success Criteria

### Primary goals
- Run open-source LLMs reliably on Raspberry Pi 5 and Android.
- Reduce average latency, energy, and cloud usage vs. fixed-model baselines.
- Maintain acceptable answer quality via quality-aware routing.
- Support local document answering, offline STT, and offline TTS.
- Produce reproducible experiments, ablations, and a documented codebase.

### Research questions
1. Can adaptive routing reduce latency/energy vs. always using the largest local model?
2. Can CortexEdge preserve quality while reducing cloud dependence?
3. How accurately can the router pick the lowest-cost adequate model?
4. When should local RAG be used instead of a larger LLM?
5. How should routing change as memory, temperature, battery, or network degrade?

### Optimization objective
For each candidate route, compute a weighted cost (latency, energy, memory, cloud cost, privacy risk) rewarding response quality. Select the **minimum-cost route** satisfying system constraints and minimum quality requirement.

---

## 2. System Architecture

```
Voice/Text input
  -> Input & privacy policy manager
  -> Query feature & complexity extractor
  -> Adaptive request router
        |-> Small LLM | Medium LLM | Local RAG | Cloud model
  -> Response cache, logger, evaluator
  -> Text output or offline TTS
```

### Routing inputs
| Category | Examples |
|---|---|
| Prompt features | token count, sentence count, task type, code/math indicators, reasoning depth, retrieval need |
| Device state | free RAM, CPU load, temperature, battery, currently loaded model |
| Network state | connectivity, RTT, bandwidth, cloud availability |
| Policy constraints | offline-only mode, privacy level, latency vs. quality preference |

### Execution routes
| Route | Purpose | Typical use |
|---|---|---|
| Small local model | Fast, energy-efficient | Greetings, definitions, short factual queries |
| Medium local model | Higher reasoning | Technical explanations, summarization, moderate reasoning |
| Local RAG | Grounded answers from private docs | Manuals, notes, reports |
| Cloud fallback | Difficult/resource-intensive tasks | Complex reasoning, long-context, current info |

---

## 3. Tech Stack

| Layer | Tool |
|---|---|
| Local LLM runtime | llama.cpp |
| Model format | GGUF |
| API | FastAPI + Uvicorn |
| UI | Streamlit (initial) |
| Android app | Kotlin + Jetpack Compose |
| STT | whisper.cpp |
| TTS | Piper or Android offline TTS |
| Embeddings | sentence-transformers / ONNX Runtime |
| Vector search | FAISS |
| Fine-tuning | Transformers, PEFT, TRL |
| Experiment tracking | MLflow |
| Monitoring | psutil, vcgencmd, INA219/INA226 |
| Data storage | SQLite or DuckDB |
| Testing | pytest |
| VCS | GitHub |

### Hardware
- **Primary:** Raspberry Pi 5, 8 GB RAM, active cooling, USB mic + speaker, 64 GB+ microSD (USB 3 SSD preferred for models), INA219/INA226 for power logging.
- **Secondary:** Android phone, ARM64, Android 10+, 6 GB+ RAM.
- **Dev system:** 16 GB RAM, Python, Git, CMake, Android Studio.

---

## 4. Repository Structure

```
cortexedge/
├── app/
│   ├── api/server.py            # FastAPI orchestrator
│   ├── router/
│   │   ├── features.py          # Query feature/complexity extraction
│   │   ├── rule_router.py       # Transparent rule-based baseline
│   │   └── learned_router.py    # ML classifier router
│   ├── inference/llm_client.py  # local + cloud inference clients
│   ├── rag/
│   │   ├── ingest.py            # PDF extract, chunk, embed, FAISS index
│   │   └── retriever.py         # search + RAG prompt builder
│   ├── cache/semantic_cache.py  # exact + semantic cache
│   ├── monitoring/
│   │   ├── system_state.py      # psutil + thermal + network probes
│   │   └── logger.py            # SQLite experiment logger
│   ├── speech/                  # STT/TTS wrappers
│   ├── config.py                # pydantic-settings central config
│   └── schemas.py               # Pydantic request/result/state models
├── models/{llm,embeddings,stt,tts}/
├── datasets/                    # prompts.txt, documents/, routing_training.csv
├── scripts/
│   ├── benchmark_routes.py      # per-route benchmark -> CSV
│   ├── train_router.py          # train & evaluate learned router
│   └── log_power.py             # INA219 energy logging
├── experiments/                 # results/, router.joblib, cortexedge.db
├── tests/test_router.py
├── .env
└── README.md
```

---

## 5. Phased Execution Plan

Each phase lists: **Goal → Tasks → Deliverable → Exit criteria → Dependencies.**

### Phase 1 — Environment Setup & First Local Model  *(Week 1)*
**Goal:** Prepare the Pi and prove one quantized model runs reliably.
- [ ] Install 64-bit Raspberry Pi OS; full upgrade.
- [ ] Install system packages: git, git-lfs, cmake, build-essential, python3-pip/venv, ffmpeg, libopenblas-dev, sqlite3.
- [ ] Create Python venv; install FastAPI, uvicorn, pydantic(-settings), psutil, pandas, numpy, scikit-learn, sentence-transformers, faiss-cpu, diskcache, mlflow, pytest, streamlit, pymupdf, python-docx, python-multipart.
- [ ] Clone & build llama.cpp with OpenBLAS (`-DGGML_BLAS=ON`).
- [ ] Download one small GGUF model (SmolLM2-360M-Instruct or Qwen2.5-0.5B-Instruct).
- [ ] Run `llama-cli` and record load time, latency, memory, temperature.
**Deliverable:** Working local inference + baseline metrics.
**Exit criteria:** Model answers prompts; metrics captured.
**Dependencies:** None.

### Phase 2 — Two-Model & Quantization Benchmark  *(Week 2)*
**Goal:** Establish small + medium model hierarchy.
- [ ] Select small-tier and medium-tier (1–3B) instruction models. Recommended: **SmolLM2-360M-Instruct** or **Qwen2.5-0.5B-Instruct** (small); a 1–3B instruction model that fits hardware + licence (medium).
- [ ] Test Q8, Q5, Q4_K_M quantization for each.
- [ ] Fix identical prompts & generation settings (temperature, max_tokens).
- [ ] Measure TTFT, tokens/s, total latency, peak RAM, temperature, quality.
**Deliverable:** Comparative benchmark table; choose deployment quantization.
**Exit criteria:** Small + medium quantized models selected.
**Dependencies:** Phase 1.

### Phase 3 — Persistent Inference Service  *(Week 3)*
**Goal:** Avoid reloading models per request.
- [ ] Run persistent `llama-server` for small (port 8101) and medium (port 8102).
- [ ] Implement model manager with memory-safe load/unload rules.
- [ ] Add monitoring (psutil + thermal + network probe).
- [ ] Add SQLite experiment logger schema.
**Deliverable:** Stable local REST endpoints + logger.
**Exit criteria:** `/health` returns system state; endpoints serve both models.
**Dependencies:** Phase 2.

### Phase 4 — Rule-Based Router  *(Week 4)*
**Goal:** First transparent routing baseline.
- [ ] Implement `features.py` (token/sentence counts, code/math indicators, reasoning/multi-part scores, complexity score).
- [ ] Implement `rule_router.py` per rule logic (privacy/offline → local; resource pressure → small/cloud; documents → RAG; low complexity → small; moderate → medium; high → cloud).
- [ ] Wire into FastAPI `/query` orchestrator with controlled fallback (medium→small→cloud).
- [ ] Log route + reason for every decision.
**Deliverable:** Functional multi-route prototype.
**Exit criteria:** End-to-end `/query` returns routed responses with reasons.
**Dependencies:** Phase 3.

### Phase 5 — Routing Dataset  *(Week 5)*
**Goal:** Collect supervised data for learned routing.
- [ ] Prepare 1,000–2,000 prompts across classes: simple, technical, document-grounded, coding, math, multi-step reasoning, safety-critical.
- [ ] Run `benchmark_routes.py`: execute each prompt on every route with identical params.
- [ ] Capture per-route latency, TTFT, token rate, peak RAM, temperature, energy, quality.
- [ ] Label best route: retain routes within quality tolerance of best, pick lowest weighted cost.
**Deliverable:** Structured `routing_training.csv` with best-route labels.
**Exit criteria:** Dataset covers all task classes; labels generated.
**Dependencies:** Phase 4.

**Dataset field schema (per prompt × route):**
| Field group | Examples |
|---|---|
| Prompt metadata | prompt ID, task class, token count, privacy level |
| Route metrics | latency, TTFT, token rate, peak RAM, energy |
| Quality metrics | automatic score, human rating, hallucination, faithfulness |
| System state | temperature, CPU load, free memory, network condition |
| Decision label | best route under quality + cost constraints |

**Best-route labelling rule:** retain routes whose quality is within a small tolerance of the best observed quality; among those, select the one with the lowest weighted system cost.

### Phase 6 — Learned Router  *(Week 6)*
**Goal:** Replace fixed thresholds with lightweight predictive model.
- [ ] Train logistic regression, decision tree, random forest, gradient-boosted models.
- [ ] Select simplest model with strong performance (start: RandomForest, balanced class weights).
- [ ] Evaluate routing accuracy, macro F1, quality regret, avg energy, cloud-use.
- [ ] Persist model with joblib; wire `LearnedRouter` behind a feature flag.
**Deliverable:** Trained classifier + routing metrics report.
**Exit criteria:** Learned router matches or beats rule baseline on F1.
**Dependencies:** Phase 5.

### Phase 7 — Local RAG  *(Week 7)*
**Goal:** Grounded answers from private docs without cloud.
- [ ] `ingest.py`: PDF text extraction (pymupdf), chunking (450 words, 70 overlap), embeddings, FAISS `IndexFlatIP` index.
- [ ] `retriever.py`: top-k search + `build_rag_prompt` with source citations.
- [ ] Integrate RAG route into orchestrator (routes to medium model with retrieved context).
- [ ] Evaluate retrieval recall@k, answer faithfulness, citation correctness.
**Deliverable:** Local document-QA pipeline.
**Exit criteria:** Document questions answered with correct citations.
**Dependencies:** Phase 4 (router), Phase 3 (medium model server).

### Phase 8 — Caching & Cloud Fallback  *(Week 8)*
**Goal:** Reduce repeated computation; robust service under hard tasks.
- [ ] Implement exact cache (diskcache, SHA256 key, 7-day TTL).
- [ ] Implement semantic cache (embeddings, cosine similarity, 0.95 threshold).
- [ ] Disable caching for private/restricted requests.
- [ ] Add cloud fallback with timeout, retry, and circuit-breaker controls.
- [ ] Measure cache hit rate, latency saved, energy saved.
**Deliverable:** Exact + semantic cache; controlled cloud fallback.
**Exit criteria:** Cache hits return ~0 ms; circuit breaker disables failing cloud routes.
**Dependencies:** Phase 4, Phase 6.

### Phase 9 — Offline Speech  *(Week 9)*
**Goal:** Natural voice interaction without internet.
- [ ] Build whisper.cpp; download `tiny.en` STT model.
- [ ] ffmpeg preprocessing (16 kHz mono PCM).
- [ ] Integrate Piper (or Android offline TTS) for spoken output.
- [ ] Wire STT → router → TTS end-to-end.
**Deliverable:** Voice query → local routing → spoken response demo.
**Exit criteria:** End-to-end voice round-trip works offline.
**Dependencies:** Phase 4 (router stable on text first).

### Phase 10 — Android Integration  *(Week 10)*
**Goal:** Extend to mobile + cooperative edge execution.
- [ ] Build Android client (Kotlin + Jetpack Compose) for the Pi FastAPI.
- [ ] Add on-device small model where feasible.
- [ ] Implement phone → Pi → cloud cooperative routing per policy/resource state.
- [ ] Secure API (auth + HTTPS between phone and Pi).
**Deliverable:** Android app with local, Pi-assisted, and cloud execution.
**Exit criteria:** App routes requests across phone/Pi/cloud by policy.
**Dependencies:** Phase 8 (cloud fallback), Phase 9 (optional voice on mobile).

### Phase 11 — Fine-Tuning  *(optional, stretch)*
**Goal:** Improve behavior for one defined domain.
- [ ] LoRA/QLoRA on GPU workstation/cloud notebook.
- [ ] Merge adapter, convert to GGUF, quantize, deploy.
- [ ] Compare: base vs. fine-tuned vs. base+RAG vs. fine-tuned+RAG.
**Deliverable:** Controlled comparison across four configurations.
**Exit criteria:** Fine-tuned model improves over base+RAG baseline.
**Dependencies:** Phase 7 (RAG baseline established first).

### Phase 12 — Full Evaluation  *(Week 11)*
**Goal:** Scientifically demonstrate adaptive routing value.
- [ ] Run all baselines: small-only, medium-only, cloud-only, length-threshold, rule-based, learned, complete CortexEdge.
- [ ] Ablations A1–A9 (semantic cache, RAG, device-state, network-state, cloud fallback, learned router, multiple models, quantization, quality constraint).
- [ ] Energy measurement (INA219/INA226 logging, net energy = query − idle).
- [ ] Quality evaluation (ROUGE, BERTScore, exact match, token F1, code pass rate, RAG faithfulness, human rubric with ≥2 evaluators + inter-rater agreement).
- [ ] Statistical analysis (P50/P95 latency, std dev, ≥5 repeats per config).
**Deliverable:** Reproducible result tables, plots, ablation summaries.
**Exit criteria:** All baselines + ablations documented with stats.
**Dependencies:** All prior phases.

### Phase 13 — Documentation & Manuscript  *(Week 12)*
- [ ] Finalize repository (README, reproducibility instructions, requirements).
- [ ] Final report + demo video.
- [ ] Manuscript draft for publication.
**Deliverable:** Repository, report, demo, manuscript draft.
**Exit criteria:** External user can reproduce results from repo.
**Dependencies:** Phase 12.

---

## 6. Twelve-Week Schedule

| Week | Main work | Output |
|---|---|---|
| 1 | Environment + first model | Working Pi inference + baseline metrics |
| 2 | Two-model + quantization benchmark | Comparative performance table |
| 3 | API + monitoring | Persistent inference service + logger |
| 4 | Rule-based routing | Multi-route prototype |
| 5 | Prompt dataset | 1,000+ prompts + batch runner |
| 6 | Learned router | Trained classifier + routing metrics |
| 7 | Local RAG | Document ingestion + grounded answering |
| 8 | Cache + cloud fallback | Exact/semantic cache + robust fallback |
| 9 | Offline speech | STT→router→TTS demo |
| 10 | Android integration | Mobile client + on-device small model |
| 11 | Full experiments | Baselines, ablations, energy, quality |
| 12 | Documentation | Repo, report, demo, manuscript draft |

---

## 7. Router Design Details

### Task classes
| Class | Description |
|---|---|
| 0 | Greeting / simple command |
| 1 | Simple factual question |
| 2 | Summarization / rewrite / translation |
| 3 | Document-grounded question |
| 4 | Coding / mathematics |
| 5 | Multi-step reasoning |
| 6 | Safety-critical / high-accuracy |

### Rule logic (baseline)
```
if offline_only or private_data:
    use local RAG when documents needed
    else small (complexity <= 0.45) or medium (complexity > 0.45)
elif low RAM (<1500 MB) or high temperature (>75°C):
    small model, or cloud if permitted & complexity > 0.35
elif requires documents:
    local RAG
elif complexity low (<0.35):
    small
elif complexity moderate (<0.70):
    medium
else:
    cloud (if network), else medium
```

### Key configuration values
| Setting | Default |
|---|---|
| Small model server | `http://127.0.0.1:8101/v1/chat/completions` |
| Medium model server | `http://127.0.0.1:8102/v1/chat/completions` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Semantic cache threshold | 0.95 |
| Max CPU temperature | 75.0 °C |
| Minimum free RAM | 1500 MB |
| Request timeout | 90 s |

### Learned router I/O
- **Inputs:** prompt features (length, task class, code/math indicators, reasoning score, retrieval need) + device state (RAM, CPU, temp, battery, loaded model) + network state + policy (privacy, offline, quality/latency priority).
- **Outputs:** `small | medium | local-RAG-small | local-RAG-medium | cloud`.

---

## 8. Evaluation Framework

### Baselines
Small-only · Medium-only · Cloud-only · Length-threshold · Rule-based CortexEdge · Learned CortexEdge · Complete CortexEdge (learned router + RAG + cache + fallback).

### Metrics
- Routing accuracy & macro F1.
- Latency: mean, median, P95.
- TTFT & token generation speed.
- Peak memory & model-switching overhead.
- Net energy per query (query − idle).
- Response quality & **quality regret** (quality lost vs. the best-available route for each query).
- Cloud offloading % & local completion %.
- Failure rate & constraint-violation rate.

### Ablations (A1–A9)
Semantic cache · RAG · device-state features · network-state features · cloud fallback · learned router · multiple models · quantization · quality constraint.

### Quality metrics by task
- Summarization: ROUGE-1/2/L, BERTScore.
- QA: exact match, token F1, semantic similarity.
- Code: compile success, test-case pass rate.
- RAG: recall@k, context precision, faithfulness, citation correctness.
- General: human rubric (correctness, completeness, relevance, clarity, hallucination) with ≥2 evaluators + inter-rater agreement.

---

## 9. Energy & Resource Measurement

- Log CPU %, process memory, temperature, network traffic, generated tokens at fixed intervals.
- USB power meter for initial checks; INA219/INA226 for continuous V/I logging.
- Energy = time integral of power; **net energy = query energy − (idle power × query duration)**.
- Android: Android Studio Profiler / Perfetto + external power measurement.
- **Discipline:** report averages + variability; ≥5 repeats per config after warm-up; identical prompts; controlled thermal conditions.

---

## 10. Privacy, Security & Reliability

| Policy | Behaviour |
|---|---|
| Public | Cloud permitted |
| Private | Local model + local RAG only |
| Restricted | No cloud, no cache, no prompt logging, encrypted local storage |
| Ephemeral | In-memory only; delete temp data immediately |

### Minimum controls
- Bind model servers to localhost unless remote access required.
- API auth + HTTPS between phone and Pi.
- Separate performance logs from raw prompt content.
- Encrypt private document storage; remove sensitive temp files.
- Graceful fallback for OOM, model failures, cloud timeouts, network loss.
- Circuit breaker to disable repeatedly failing cloud routes.

---

## 11. Outcome Levels

| Level | Required components |
|---|---|
| Minimum viable product | Pi, two quantized models, rule router, FastAPI, monitoring, cloud fallback |
| Strong final-year project | + learned router, local RAG, semantic cache, offline speech, Android client, energy measurement |
| Publication-quality system | + quality-constrained optimization, online adaptation, cooperative phone–Pi execution, complete ablations, reproducible dataset |

---

## 12. Recommended Execution Sequence

1. Build llama.cpp; run small model manually.
2. Start small model as persistent server; test with curl.
3. Add medium model; memory-safe loading rules.
4. Run FastAPI with rule-based router.
5. Add logging; collect benchmark data (identical prompts).
6. Build local RAG index; test document questions.
7. Add exact caching, then cautiously enable semantic caching.
8. Generate route labels via quality-constrained cost selection.
9. Train & compare lightweight learned routers.
10. Add STT/TTS only after text inference is stable.
11. Android client; later on-device inference.
12. Complete baselines, ablations, energy measurements, human evaluation.

---

## 13. Common Mistakes to Avoid

- Starting Android before proving runtime on Linux.
- Using too many models before validating a simple two-model hierarchy.
- Claiming novelty merely from combining LLM + RAG + STT + TTS.
- Fine-tuning before measuring base model + RAG baselines.
- Evaluating only latency (ignore quality, energy, memory).
- Using prompt length as the sole complexity indicator.
- Comparing models with different prompts/generation params.
- Reporting only average latency (no P50/P95/std dev).
- Treating estimated power as measured energy.
- Sending private prompts to cloud without explicit policy control.

---

## 14. Final Implementation Checklist

- [ ] Run one local model; record baseline metrics.
- [ ] Benchmark quantization; select deployment formats.
- [ ] Run small + medium local models via persistent APIs.
- [ ] Add monitoring + rule-based router.
- [ ] Build routing dataset; train learned router.
- [ ] Integrate local RAG, caching, cloud fallback.
- [ ] Add offline STT + TTS.
- [ ] Develop Android client + cooperative routing.
- [ ] Instrument energy; run controlled experiments.
- [ ] Complete baselines, ablations, quality evaluation, documentation.

**Best final outcome:** A reproducible edge-AI runtime demonstrating measurable reductions in latency, energy, and cloud dependence while maintaining a clearly defined response-quality target.

---

## 15. Reference Implementation (from doc)

Runnable reference implementation. Copy each listing into the indicated path; adapt model filenames and cloud credentials.

### 15.1 Project setup
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y git git-lfs cmake build-essential python3 python3-pip \
    python3-venv ffmpeg libopenblas-dev sqlite3

mkdir -p ~/cortexedge/{app/{api,router,inference,rag,cache,monitoring,speech},models/{llm,embeddings,stt,tts},datasets,experiments,tests}
cd ~/cortexedge
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install fastapi uvicorn pydantic pydantic-settings requests psutil pandas numpy \
    scikit-learn sentence-transformers faiss-cpu diskcache mlflow pytest \
    pymupdf python-docx python-multipart

# Build llama.cpp with OpenBLAS
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

### 15.2 `app/config.py`
```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    small_model_url: str = "http://127.0.0.1:8101/v1/chat/completions"
    medium_model_url: str = "http://127.0.0.1:8102/v1/chat/completions"
    cloud_model_url: str | None = None
    cloud_api_key: str | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: Path = ROOT / "datasets/faiss.index"
    chunk_store_path: Path = ROOT / "datasets/chunks.json"
    database_path: Path = ROOT / "experiments/cortexedge.db"
    cache_path: Path = ROOT / ".cache"
    semantic_cache_threshold: float = 0.95
    max_cpu_temperature_c: float = 75.0
    minimum_free_ram_mb: int = 1500
    request_timeout_s: int = 90
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
```

### 15.3 `app/schemas.py`
```python
from typing import Literal, Any
from pydantic import BaseModel, Field

Route = Literal["small", "medium", "local_rag", "cloud", "cache"]

class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    offline_only: bool = False
    private: bool = False
    use_documents: bool = False
    response_mode: Literal["text", "voice"] = "text"
    quality_priority: float = Field(0.5, ge=0.0, le=1.0)
    latency_priority: float = Field(0.5, ge=0.0, le=1.0)
    energy_priority: float = Field(0.5, ge=0.0, le=1.0)

class InferenceResult(BaseModel):
    text: str
    route: Route
    latency_ms: float
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    metadata: dict[str, Any] = {}

class SystemState(BaseModel):
    free_ram_mb: float
    cpu_percent: float
    cpu_temperature_c: float | None
    network_available: bool
    loaded_model: str | None = None
```

### 15.4 `app/monitoring/system_state.py`
```python
import socket
from pathlib import Path
import psutil
from app.schemas import SystemState

THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")

def read_temperature() -> float | None:
    try:
        return float(THERMAL_PATH.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None

def network_available(host: str = "1.1.1.1", port: int = 53) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False

def read_system_state(loaded_model: str | None = None) -> SystemState:
    vm = psutil.virtual_memory()
    return SystemState(
        free_ram_mb=vm.available / (1024 * 1024),
        cpu_percent=psutil.cpu_percent(interval=0.1),
        cpu_temperature_c=read_temperature(),
        network_available=network_available(),
        loaded_model=loaded_model,
    )
```

### 15.5 `app/router/features.py`
```python
import math
import re
from dataclasses import dataclass, asdict

REASONING_WORDS = {"derive", "prove", "compare", "analyse", "analyze", "why", "steps"}
CODE_MARKERS = {"python", "java", "c++", "code", "function", "algorithm", "debug"}
MATH_PATTERN = re.compile(r"[=+\-*/^]|\b(sin|cos|log|integral|matrix)\b", re.I)

@dataclass
class QueryFeatures:
    token_count: int
    sentence_count: int
    has_code: int
    has_math: int
    reasoning_score: float
    multi_part_score: float
    requires_documents: int
    complexity_score: float

    def as_dict(self):
        return asdict(self)

def extract_features(text: str, requires_documents: bool = False) -> QueryFeatures:
    words = re.findall(r"\b\w+\b", text.lower())
    token_count = len(words)
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    has_code = int(any(w in CODE_MARKERS for w in words) or "```" in text)
    has_math = int(bool(MATH_PATTERN.search(text)))
    reasoning_hits = sum(w in REASONING_WORDS for w in words)
    reasoning_score = min(1.0, reasoning_hits / 3.0)
    multi_part_score = min(1.0, (text.count("?") + text.count(";") + text.count("\n")) / 4.0)
    length_score = min(1.0, math.log1p(token_count) / math.log(513))
    complexity = (
        0.20 * length_score + 0.20 * reasoning_score + 0.15 * has_code +
        0.15 * has_math + 0.15 * multi_part_score + 0.15 * int(requires_documents)
    )
    return QueryFeatures(token_count, sentence_count, has_code, has_math,
                         reasoning_score, multi_part_score,
                         int(requires_documents), round(complexity, 4))
```

### 15.6 `app/router/rule_router.py`
```python
from dataclasses import dataclass
from app.config import settings
from app.router.features import QueryFeatures
from app.schemas import QueryRequest, SystemState, Route

@dataclass
class RoutingDecision:
    route: Route
    reason: str

def select_route(features: QueryFeatures, state: SystemState,
                 request: QueryRequest) -> RoutingDecision:
    private_mode = request.private or request.offline_only

    if request.use_documents or features.requires_documents:
        return RoutingDecision("local_rag", "Document-grounded request")

    if private_mode:
        route = "small" if features.complexity_score <= 0.45 else "medium"
        return RoutingDecision(route, "Private/offline policy prohibits cloud")

    too_hot = state.cpu_temperature_c is not None and \
        state.cpu_temperature_c > settings.max_cpu_temperature_c
    low_memory = state.free_ram_mb < settings.minimum_free_ram_mb

    if too_hot or low_memory:
        if state.network_available and features.complexity_score > 0.35:
            return RoutingDecision("cloud", "Local resource pressure")
        return RoutingDecision("small", "Resource pressure with no cloud")

    if features.complexity_score < 0.35:
        return RoutingDecision("small", "Low predicted complexity")
    if features.complexity_score < 0.70:
        return RoutingDecision("medium", "Moderate predicted complexity")
    if state.network_available:
        return RoutingDecision("cloud", "High complexity and network available")
    return RoutingDecision("medium", "High complexity but offline")
```

### 15.7 `app/router/learned_router.py`
```python
from pathlib import Path
import joblib
import numpy as np
from app.router.features import QueryFeatures
from app.schemas import SystemState

FEATURE_ORDER = [
    "token_count", "sentence_count", "has_code", "has_math",
    "reasoning_score", "multi_part_score", "requires_documents",
    "complexity_score", "free_ram_mb", "cpu_percent",
    "cpu_temperature_c", "network_available"
]

class LearnedRouter:
    def __init__(self, model_path: str | Path):
        self.model = joblib.load(model_path)

    def predict(self, q: QueryFeatures, s: SystemState) -> str:
        row = q.as_dict() | {
            "free_ram_mb": s.free_ram_mb,
            "cpu_percent": s.cpu_percent,
            "cpu_temperature_c": s.cpu_temperature_c or 0.0,
            "network_available": int(s.network_available),
        }
        x = np.array([[row[name] for name in FEATURE_ORDER]], dtype=float)
        return str(self.model.predict(x)[0])
```

### 15.8 `app/inference/llm_client.py`
```python
import time
import requests
from app.config import settings
from app.schemas import InferenceResult

URLS = {"small": settings.small_model_url, "medium": settings.medium_model_url}

def run_local(route: str, prompt: str, max_tokens: int = 256) -> InferenceResult:
    start = time.perf_counter()
    response = requests.post(
        URLS[route],
        json={"model": route, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.2, "max_tokens": max_tokens, "stream": False},
        timeout=settings.request_timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    elapsed_ms = (time.perf_counter() - start) * 1000
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    completion = usage.get("completion_tokens")
    tps = completion / (elapsed_ms / 1000) if completion and elapsed_ms else None
    return InferenceResult(
        text=text, route=route, latency_ms=elapsed_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion, tokens_per_second=tps,
        metadata={"backend": "llama.cpp"},
    )

def run_cloud(prompt: str, max_tokens: int = 256) -> InferenceResult:
    if not settings.cloud_model_url or not settings.cloud_api_key:
        raise RuntimeError("Cloud endpoint is not configured")
    start = time.perf_counter()
    r = requests.post(
        settings.cloud_model_url,
        headers={"Authorization": f"Bearer {settings.cloud_api_key}"},
        json={"messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens, "temperature": 0.2},
        timeout=settings.request_timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    return InferenceResult(
        text=data["choices"][0]["message"]["content"], route="cloud",
        latency_ms=(time.perf_counter() - start) * 1000,
        metadata={"backend": "cloud"},
    )
```

### 15.9 `app/cache/semantic_cache.py`
```python
import hashlib
import json
from pathlib import Path
import numpy as np
from diskcache import Cache
from sentence_transformers import SentenceTransformer
from app.config import settings

class SemanticCache:
    def __init__(self):
        Path(settings.cache_path).mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(settings.cache_path / "exact"))
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.records_path = settings.cache_path / "semantic.jsonl"
        self.records = self._load_records()

    def _load_records(self):
        if not self.records_path.exists(): return []
        return [json.loads(line) for line in self.records_path.read_text().splitlines() if line]

    @staticmethod
    def exact_key(text: str, route: str = "auto") -> str:
        return hashlib.sha256(f"{route}|{text.strip()}".encode()).hexdigest()

    def exact_get(self, text: str):
        return self.cache.get(self.exact_key(text))

    def exact_put(self, text: str, response: str):
        self.cache.set(self.exact_key(text), response, expire=7 * 86400)

    def semantic_get(self, text: str):
        if not self.records: return None
        q = self.embedder.encode([text], normalize_embeddings=True)[0]
        matrix = np.asarray([r["embedding"] for r in self.records], dtype="float32")
        scores = matrix @ q.astype("float32")
        idx = int(np.argmax(scores))
        return self.records[idx]["response"] if scores[idx] >= settings.semantic_cache_threshold else None

    def semantic_put(self, text: str, response: str):
        emb = self.embedder.encode([text], normalize_embeddings=True)[0].tolist()
        record = {"query": text, "response": response, "embedding": emb}
        self.records.append(record)
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
```

### 15.10 `app/rag/ingest.py` & `app/rag/retriever.py`
```python
# ingest.py
import json
from pathlib import Path
import faiss
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

def extract_pdf(path: Path) -> str:
    with fitz.open(path) as pdf:
        return "\n".join(page.get_text("text") for page in pdf)

def chunk_text(text: str, size: int = 450, overlap: int = 70):
    words = text.split()
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size]).strip()
        if len(chunk.split()) >= 40: chunks.append(chunk)
    return chunks

def build_index(document_dir: str):
    embedder = SentenceTransformer(settings.embedding_model)
    records = []
    for path in Path(document_dir).glob("*.pdf"):
        for number, chunk in enumerate(chunk_text(extract_pdf(path))):
            records.append({"source": path.name, "chunk": number, "text": chunk})
    vectors = embedder.encode([r["text"] for r in records],
                              normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(settings.faiss_index_path))
    settings.chunk_store_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(records)} chunks")

if __name__ == "__main__":
    build_index("datasets/documents")
```
```python
# retriever.py
import json
import faiss
from sentence_transformers import SentenceTransformer
from app.config import settings

class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.index = faiss.read_index(str(settings.faiss_index_path))
        self.records = json.loads(settings.chunk_store_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 4):
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, top_k)
        return [self.records[i] | {"score": float(scores[0][rank])}
                for rank, i in enumerate(ids[0]) if i >= 0]

def build_rag_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {p['source']}, chunk {p['chunk']}]\n{p['text']}" for p in passages
    )
    return f"""Answer using only the supplied context. If the answer is absent, say so.
Cite sources using [filename, chunk number].

CONTEXT
{context}

QUESTION
{question}

ANSWER"""
```

### 15.11 `app/monitoring/logger.py`
```python
import json
import sqlite3
from datetime import datetime, timezone
from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
 id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, prompt_hash TEXT,
 route TEXT, reason TEXT, latency_ms REAL, ttft_ms REAL, tokens_per_second REAL,
 free_ram_mb REAL, cpu_percent REAL, temperature_c REAL, network_available INTEGER,
 quality_score REAL, metadata_json TEXT
);"""

def initialize():
    with sqlite3.connect(settings.database_path) as con:
        con.execute(SCHEMA)

def log_request(prompt_hash, route, reason, result, state, quality_score=None):
    initialize()
    row = (
        datetime.now(timezone.utc).isoformat(), prompt_hash, route, reason,
        result.latency_ms, result.ttft_ms, result.tokens_per_second,
        state.free_ram_mb, state.cpu_percent, state.cpu_temperature_c,
        int(state.network_available), quality_score, json.dumps(result.metadata),
    )
    with sqlite3.connect(settings.database_path) as con:
        con.execute("""INSERT INTO requests
        (timestamp,prompt_hash,route,reason,latency_ms,ttft_ms,tokens_per_second,
         free_ram_mb,cpu_percent,temperature_c,network_available,quality_score,metadata_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
```

### 15.12 `app/api/server.py` (orchestrator)
```python
import hashlib
from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest
from app.router.features import extract_features
from app.router.rule_router import select_route
from app.monitoring.system_state import read_system_state
from app.monitoring.logger import log_request
from app.inference.llm_client import run_local, run_cloud
from app.cache.semantic_cache import SemanticCache
from app.rag.retriever import Retriever, build_rag_prompt

app = FastAPI(title="CortexEdge", version="0.1.0")
cache = SemanticCache()
retriever = None

@app.on_event("startup")
def startup():
    global retriever
    try: retriever = Retriever()
    except Exception: retriever = None

@app.get("/health")
def health():
    return {"status": "ok", "system": read_system_state().model_dump()}

@app.post("/query")
def query(request: QueryRequest):
    cached = cache.exact_get(request.text)
    if cached and not request.private:
        return {"route": "cache", "response": cached, "latency_ms": 0.0}

    state = read_system_state()
    features = extract_features(request.text, request.use_documents)
    decision = select_route(features, state, request)
    prompt, route = request.text, decision.route

    try:
        if route == "local_rag":
            if retriever is None: raise RuntimeError("RAG index is not available")
            passages = retriever.search(request.text)
            prompt = build_rag_prompt(request.text, passages)
            result = run_local("medium", prompt)
            result.route = "local_rag"
            result.metadata["sources"] = passages
        elif route in {"small", "medium"}:
            result = run_local(route, prompt)
        elif route == "cloud":
            result = run_cloud(prompt)
        else:
            raise RuntimeError(f"Unsupported route: {route}")
    except Exception as exc:
        if route == "medium": result = run_local("small", prompt)
        elif not request.offline_only and not request.private and state.network_available:
            result = run_cloud(prompt)
        else: raise HTTPException(status_code=503, detail=str(exc))

    if not request.private:
        cache.exact_put(request.text, result.text)
    prompt_hash = hashlib.sha256(request.text.encode()).hexdigest()
    log_request(prompt_hash, result.route, decision.reason, result, state)
    return {"route": result.route, "reason": decision.reason,
            "response": result.text, "latency_ms": result.latency_ms,
            "tokens_per_second": result.tokens_per_second,
            "system": state.model_dump()}
```

### 15.13 `scripts/benchmark_routes.py` & `scripts/train_router.py`
```python
# benchmark_routes.py
import csv, time
from pathlib import Path
import psutil
from app.inference.llm_client import run_local, run_cloud
from app.monitoring.system_state import read_system_state

ROUTES = ["small", "medium", "cloud"]

def execute(route, prompt):
    return run_cloud(prompt) if route == "cloud" else run_local(route, prompt)

def main():
    prompts = [l.strip() for l in Path("datasets/prompts.txt").read_text().splitlines() if l.strip()]
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open("experiments/results/route_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["prompt_id","route","latency_ms","tokens_per_second",
                  "rss_mb","free_ram_mb","temperature_c","response"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        process = psutil.Process()
        for pid, prompt in enumerate(prompts):
            for route in ROUTES:
                try:
                    result = execute(route, prompt)
                    state = read_system_state()
                    writer.writerow({
                        "prompt_id": pid, "route": route,
                        "latency_ms": result.latency_ms,
                        "tokens_per_second": result.tokens_per_second,
                        "rss_mb": process.memory_info().rss/(1024*1024),
                        "free_ram_mb": state.free_ram_mb,
                        "temperature_c": state.cpu_temperature_c,
                        "response": result.text.replace("\n", " "),
                    })
                    f.flush(); time.sleep(2)
                except Exception as exc:
                    print(pid, route, exc)

if __name__ == "__main__": main()
```
```python
# train_router.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["token_count","sentence_count","has_code","has_math",
            "reasoning_score","multi_part_score","requires_documents",
            "complexity_score","free_ram_mb","cpu_percent",
            "cpu_temperature_c","network_available"]

df = pd.read_csv("datasets/routing_training.csv").dropna(subset=FEATURES + ["best_route"])
X, y = df[FEATURES], df["best_route"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=250, max_depth=12, class_weight="balanced", random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
joblib.dump(model, "experiments/router.joblib")
```

### 15.14 `scripts/log_power.py` (INA219 energy)
```python
import csv, time
import board, busio
from adafruit_ina219 import INA219

i2c = busio.I2C(board.SCL, board.SDA)
sensor = INA219(i2c)
interval, energy_j = 0.1, 0.0

with open("experiments/results/power.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time_s","voltage_v","current_a","power_w","energy_j"])
    start = time.perf_counter(); previous = start
    while True:
        now = time.perf_counter(); dt = now - previous; previous = now
        voltage = sensor.bus_voltage + sensor.shunt_voltage
        current = sensor.current / 1000.0
        power = voltage * current
        energy_j += power * dt
        w.writerow([now-start, voltage, current, power, energy_j]); f.flush()
        time.sleep(interval)
```
> **Net energy** = measured query energy − (idle power × query duration). ≥5 repeats per config after warm-up.

### 15.15 `tests/test_router.py`
```python
from app.router.features import extract_features
from app.router.rule_router import select_route
from app.schemas import QueryRequest, SystemState

def state(ram=4000, temp=55, network=True):
    return SystemState(free_ram_mb=ram, cpu_percent=15,
                       cpu_temperature_c=temp, network_available=network)

def test_simple_prompt_uses_small_model():
    q = QueryRequest(text="Hello")
    assert select_route(extract_features(q.text), state(), q).route == "small"

def test_private_complex_prompt_never_uses_cloud():
    text = "Derive and compare three algorithms with code and mathematical analysis."
    q = QueryRequest(text=text, private=True)
    assert select_route(extract_features(text), state(), q).route in {"small", "medium"}

def test_document_request_uses_rag():
    q = QueryRequest(text="Summarize the uploaded paper", use_documents=True)
    assert select_route(extract_features(q.text, True), state(), q).route == "local_rag"

def test_resource_pressure_can_offload():
    q = QueryRequest(text="Analyse this difficult multi-step problem and provide code.")
    assert select_route(extract_features(q.text), state(ram=700, temp=80), q).route in {"small", "cloud"}
```

---

## 16. Improvements & Recommendations

The doc is strong but has gaps between its stated research contribution and the reference code. The following improvements close those gaps and harden the project for reproducibility.

### 16.1 Core research gap — implement the actual cost function
The doc's headline contribution is a *quality-constrained, resource-aware* router, but the reference code uses hardcoded thresholds (rule router) or a classifier (learned router). Neither implements the weighted cost minimization described in §1.3. Add `app/router/cost.py`:

```python
from dataclasses import dataclass

@dataclass
class RouteCost:
    latency_ms: float
    energy_j: float
    memory_mb: float
    cloud_cost: float
    privacy_risk: float
    quality: float

def weighted_cost(c: RouteCost, w: dict, quality_target: float) -> float:
    """Cost = weighted resource sum, penalised if quality below target."""
    cost = (w["latency"] * c.latency_ms + w["energy"] * c.energy_j +
            w["memory"] * c.memory_mb + w["cloud"] * c.cloud_cost +
            w["privacy"] * c.privacy_risk)
    if c.quality < quality_target:
        cost += w["quality_penalty"] * (quality_target - c.quality)
    return cost

def select_min_cost(routes: list[RouteCost], w: dict, quality_target: float):
    feasible = [r for r in routes if r.quality >= quality_target]
    pool = feasible or routes  # fall back to best-quality if none feasible
    return min(pool, key=lambda r: weighted_cost(r, w, quality_target))
```
This makes the "quality-constrained, resource-aware" claim concrete and testable, and it is what generates the `best_route` labels in Phase 5.

### 16.2 TTFT is never measured (bug)
`run_local` uses `stream=False`, so `ttft_ms` is always `None` and the benchmark cannot report time-to-first-token. Add a streaming path:
```python
def run_local_stream(route, prompt, max_tokens=256):
    start = time.perf_counter(); ttft = None; text = ""
    with requests.post(URLS[route], json={..., "stream": True}, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line: continue
            if line.startswith(b"data: "):
                payload = line[6:]
                if payload == b"[DONE]": break
                if ttft is None: ttft = (time.perf_counter() - start) * 1000
                text += json.loads(payload)["choices"][0]["delta"].get("content", "")
    return text, ttft, (time.perf_counter() - start) * 1000
```

### 16.3 Model manager is specified but not implemented
Phase 3 requires a memory-safe model manager; the doc only shows manual `llama-server` commands. Add `app/inference/model_manager.py` that starts/stops `llama-server` subprocesses based on free RAM and unloads the medium model under pressure.

### 16.4 Semantic cache does not scale
`semantic_get` loads all embeddings into a Python list and does a linear scan. For 1,000+ records, back it with a FAISS `IndexFlatIP` (mirroring the RAG retriever) instead of `np.asarray(...) @ q`.

### 16.5 Learned router output classes are incomplete
The doc lists `local-RAG-small` and `local-RAG-medium` as output classes, but the reference `LearnedRouter` and `train_router.py` only predict `small/medium/cloud`. Extend the label set and the RAG route to select the generation model (small vs. medium) by complexity.

### 16.6 Reproducibility tooling
- **`requirements.txt`** (pin versions) instead of ad-hoc `pip install`.
- **`Makefile`** for `setup`, `run`, `test`, `benchmark`, `ingest`.
- **Dockerfile** for a reproducible Pi/ARM64 environment.
- **GitHub Actions** CI running `pytest` + `compileall` on push.
- **DVC** (or a `datasets/README.md` with hashes) to version the routing dataset and models.

### 16.7 Privacy hardening
- The `Restricted` policy requires "no prompt logging", but `log_request` always stores `prompt_hash` and `metadata_json`. Gate logging on policy level and store only aggregate metrics for `Restricted`/`Ephemeral`.
- `Ephemeral` policy should bypass the exact cache entirely (currently only `private` is checked).

### 16.8 Evaluation additions
- Add **quality regret** as a first-class metric (already noted in §8) and report it per baseline.
- Add **model-switching overhead** measurement (load/unload time) since it directly affects the "multiple models" ablation (A7).
- Report **inter-rater agreement** (Cohen's κ) for human evaluation, not just "where feasible".

### 16.9 Schedule risk buffer
The 12-week plan is tight. Recommend a 1-week buffer (Week 13) for fine-tuning (Phase 11) and Android on-device inference, which are the two highest-risk items. Mark them explicitly as *stretch* so the MVP/strong-project levels remain achievable if they slip.
