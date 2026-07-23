# MindStack — Multi-Agent Financial Research System

<p align="center">
  <strong>输入一家公司名称，自动生成 8 章节专业研报</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Orchestration-orange" alt="LangGraph">
  <img src="https://img.shields.io/badge/Data-A%20Share%20%7C%20HK%20%7C%20US-brightgreen" alt="Markets">
  <img src="https://img.shields.io/badge/Deploy-Railway-purple" alt="Railway">
</p>

---

## Overview

MindStack is a multi-agent system that generates institutional-grade financial research reports. Given a company name or ticker, it orchestrates a team of specialized agents to collect financial data, conduct web research, cross-validate findings, and produce a structured 8-section report — all streamed in real time via WebSocket.

**Key design choice:** Each agent has a single responsibility. The Chief Editor manages the workflow, Researchers fetch data, Writers draft chapters, and Reviewers validate — no single agent does everything, which keeps prompts focused and outputs consistent.

---

## Report Sections

| # | Section | Data Sources |
|---|---------|-------------|
| 1 | Company Overview | Web search + financial APIs |
| 2 | Core Financial Analysis | Xueqiu / FMP financial statements |
| 3 | Valuation Assessment | PE, PB, PS ratios + industry benchmarks |
| 4 | Industry & Competition | Peer comparison via LLM-generated peer lists |
| 5 | Risk Analysis | Web research + financial health indicators |
| 6 | ESG Evaluation | Web search (governance, environmental) |
| 7 | Technical Analysis | Price trends, moving averages |
| 8 | Investment Thesis | Synthesized from all sections above |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Frontend (Next.js)                  │
│              WebSocket ↔ real-time progress           │
└──────────────────────┬──────────────────────────────┘
                       │ wss://
┌──────────────────────▼──────────────────────────────┐
│              FastAPI Backend (main.py)                │
│  ┌────────────────────────────────────────────────┐  │
│  │       ChiefEditorAgent (LangGraph)              │  │
│  │                                                │  │
│  │  browser ──► planner ──► human ──► researcher  │  │
│  │                                    │            │  │
│  │       ┌────────────────────────────┘            │  │
│  │       ▼                                         │  │
│  │  writer ──► reviewer ──► reviser                │  │
│  │       │         │          │                    │  │
│  │       └────┬────┘          │                    │  │
│  │            ▼  reject       │                    │  │
│  │       publisher ◄── approve │                   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Role | Key Logic |
|-------|------|-----------|
| **ChiefEditorAgent** | LangGraph orchestrator | Manages state graph, routes between nodes, handles retry logic |
| **BrowserAgent** | Web search | Tavily API for multi-source web retrieval |
| **PlannerAgent** | Report structuring | Generates 8-section outline with sub-questions per section |
| **ResearcherAgent** | Data collection | Dual-source routing: A-share/HK → Xueqiu API, US → FMP API |
| **WriterAgent** | Report drafting | Financial-mode prompt with data citation requirements |
| **ReviewerAgent** | Quality assurance | Cross-validates financial numbers against source data |
| **ReviserAgent** | Revision | Addresses reviewer feedback, fixes data discrepancies |
| **PublisherAgent** | Final assembly | Merges chapters, strips duplicates, formats Markdown output |

### Data Pipeline

```
Query → Ticker Recognition (3-tier)
           ├── Regex: 6-digit (A-share), 5-digit (HK), letter codes (US)
           ├── Hardcoded map: 22 Chinese company names → tickers
           └── LLM fallback: extract ticker from natural language query
                     │
                     ▼
           Dual-Source Router
           ├── isdigit() && len ∈ {5,6} → XueqiuDataTool
           │     ├── get_stock_overview()      → 10+ financial indicators
           │     ├── get_financial_statements() → income/balance/cash flow
           │     └── get_industry_peers()       → peer comparison data
           └── else → FinancialDataTool (FMP)
                     │
                     ▼
           Concurrency Control (asyncio.Semaphore)
           └── max_parallel=1 (respects LLM API rate limits)
```

### Ticker Recognition

Three-tier fallback ensures robust stock identification:

1. **Regex patterns** — matches `600519`, `000001`, `00700`, `AAPL`, `TSLA`
2. **Hardcoded map** — 22 Chinese company names (贵州茅台, 宁德时代, etc.)
3. **LLM extraction** — calls `call_model()` for ambiguous natural language queries

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- API keys (see below)

### Installation

```bash
# Clone
git clone https://github.com/Xavier7776/multi-agent-financial-research-platform.git
cd multi-agent-financial-research-platform

# Backend
pip install -r requirements.txt
pip install -r multi_agents/requirements.txt

# Frontend
cd frontend/nextjs
npm install
```

### Environment Variables

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | LLM API key |
| `OPENAI_BASE_URL` | Yes | LLM endpoint URL |
| `TAVILY_API_KEY` | Yes | Web search |
| `JINA_API_KEY` | Yes | Embeddings (Jina AI, free tier available) |
| `MAX_PARALLEL_RESEARCH` | No | Concurrent research limit (default: 1) |
| `MAX_SCRAPER_CONTENT_LENGTH` | No | Content truncation threshold (default: 150000) |

### Running Locally

```bash
# Terminal 1 — Backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend/nextjs
npm run dev
```

Open `http://localhost:3000`, navigate to the research panel, enter a company name, and watch the agents work in real time.

---

## Deployment

Deployed on [Railway](https://railway.app/) with 1GB RAM. Configuration:

```toml
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"

[service]
healthcheckPath = "/health"
```

### Production Considerations

- **Embeddings:** Jina AI API (zero local memory) — avoids OOM on 512MB-1GB instances
- **Concurrency:** `asyncio.Semaphore(1)` — respects LLM API 2-concurrent rate limit
- **WebSocket:** 20s heartbeat ping to prevent Railway's 30s idle timeout
- **Scraper:** `MAX_SCRAPER_CONTENT_LENGTH=50000` truncation prevents memory blow-up from large PDFs

---

## WebSocket Protocol

The frontend communicates with the backend exclusively over WebSocket (`/ws`).

### Client → Server

```json
{
  "type": "start",
  "task": "贵州茅台 600519 全面分析",
  "report_type": "multi_agents",
  "tone": "Objective"
}
```

### Server → Client (progress)

```json
{
  "type": "progress",
  "agent": "Researcher",
  "message": "Fetching financial data for 600519...",
  "data": {}
}
```

### Server → Client (complete)

```json
{
  "type": "complete",
  "report": "# 贵州茅台 (600519) 深度研究报告\n\n## 1. 公司概览\n..."
}
```

Heartbeat: client sends `"ping"` every 20 seconds, server responds `"pong"`.

---

## Project Structure

```
├── backend/                     # FastAPI server
│   └── server/
│       ├── app.py               # Application entry, routes, WebSocket handler
│       ├── server_utils.py      # Config, file ops, multi-agent execution
│       └── websocket_manager.py # Connection management
├── multi_agents/                # Multi-Agent engine
│   ├── agents/
│   │   ├── chief_editor.py      # LangGraph state machine
│   │   ├── researcher.py        # Data collection + dual-source routing
│   │   ├── writer.py            # Section drafting with financial prompts
│   │   ├── reviewer.py          # Data accuracy validation
│   │   ├── reviser.py           # Feedback-driven revision
│   │   ├── publisher.py         # Report assembly + formatting
│   │   ├── editor.py            # Task decomposition
│   │   └── utils/               # File I/O, Markdown → PDF/DOCX
│   ├── components/
│   │   ├── xueqiu_finance.py    # A-share / HK stock data (pysnowball)
│   │   └── financial_data.py    # US stock data (FMP) + ticker extraction
│   ├── main.py                  # CLI entry for standalone execution
│   └── task.json                # Task definition template
├── gpt_researcher/              # Research engine (web scraping, retrieval)
├── frontend/nextjs/             # Next.js frontend
├── outputs/                     # Generated reports
├── requirements.txt
├── Procfile
└── railway.toml
```

---

## Output Example

A completed report is structured as follows:

```
outputs/
└── 贵州茅台-600519-20260708/
    ├── report.md           # Full Markdown report
    ├── report.pdf          # PDF export
    ├── report.docx         # Word export
    └── financial_data.json # Raw financial data snapshot
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph over custom orchestration | Native retry, state persistence, and visualization |
| Dual-source routing (not unified API) | No single free API covers A-share + US data with sufficient depth |
| LLM ticker fallback (not embedding similarity) | Embedding models lack financial domain training for ticker disambiguation |
| Jina over local embeddings | Eliminates 200MB+ model memory on constrained deployment instances |
| `Semaphore(1)` default | LLM API typically enforces 2 concurrent requests; single worker avoids 429 errors |
| Financial data JSON export | Enables post-hoc validation and debugging of generated reports |

---

## License

MIT
