from datetime import datetime
import asyncio
import logging
import os
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END

from .utils.views import print_agent_output
from .utils.llms import call_model
from ..memory.draft import DraftState
from . import ResearchAgent, ReviewerAgent, ReviserAgent

logger = logging.getLogger(__name__)


class EditorAgent:
    """Agent responsible for editing and managing code."""

    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers or {}

    async def plan_research(self, research_state: Dict[str, any]) -> Dict[str, any]:
        """
        Plan the research outline based on initial research and task parameters.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with title, date, and planned sections
        """
        initial_research = research_state.get("initial_research")
        task = research_state.get("task")
        include_human_feedback = task.get("include_human_feedback")
        human_feedback = research_state.get("human_feedback")
        max_sections = task.get("max_sections")

        # --- Phase 2: Financial report mode ---
        financial_data = research_state.get("financial_data") or {}
        industry_peers = research_state.get("industry_peers") or []
        is_financial = bool(financial_data)
        logger.info(
            f"[Editor DEBUG] is_financial={is_financial}, "
            f"fd_keys={list(financial_data.keys()) if financial_data else 'None'}, "
            f"ticker={financial_data.get('ticker','?')}, "
            f"peers={len(industry_peers)}"
        )
        if is_financial:
            max_sections = 8  # Force 8-section structure for financial reports
        # ------------------------------------

        prompt = self._create_planning_prompt(
            initial_research, include_human_feedback, human_feedback, max_sections,
            financial_data, industry_peers, is_financial,
        )

        # Send explicit planner stage signal for frontend node tracking
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "planner_start",
                "Planning an outline layout based on initial research...",
                self.websocket,
            )

        print_agent_output(
            "Planning an outline layout based on initial research...", agent="EDITOR")
        plan = await call_model(
            prompt=prompt,
            model=task.get("model"),
            response_format="json",
        )

        # Send research plan content for human review display
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "plan",
                f"Research plan: title={plan.get('title')}, sections={plan.get('sections')}",
                self.websocket,
                True,
                plan,
            )

        return {
            "title": plan.get("title"),
            "date": plan.get("date"),
            "sections": plan.get("sections"),
        }

    async def run_parallel_research(self, research_state: Dict[str, any]) -> Dict[str, List[str]]:
        """
        Execute parallel research tasks for each section.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with research results
        """
        agents = self._initialize_agents()
        workflow = self._create_workflow()
        chain = workflow.compile()

        queries = research_state.get("sections")
        title = research_state.get("title")

        self._log_parallel_research(queries)

        # 并行数控制：默认 1，适配 LLM API 2 并发限制
        max_parallel = int(os.environ.get("MAX_PARALLEL_RESEARCH", "1"))
        semaphore = asyncio.Semaphore(max_parallel)

        async def _limited_invoke(query):
            async with semaphore:
                task_input = self._create_task_input(research_state, query, title)
                return await chain.ainvoke(task_input, config={"tags": ["gpt-researcher"]})

        final_drafts = [_limited_invoke(query) for query in queries]
        research_results = [
            result["draft"] for result in await asyncio.gather(*final_drafts)
        ]

        return {"research_data": research_results}

    def _create_planning_prompt(self, initial_research: str, include_human_feedback: bool,
                                human_feedback: Optional[str], max_sections: int,
                                financial_data: dict = None, industry_peers: list = None,
                                is_financial: bool = False) -> List[Dict[str, str]]:
        """Create the prompt for research planning."""
        return [
            {
                "role": "system",
                "content": "You are a research editor. Your goal is to oversee the research project "
                           "from inception to completion. Your main task is to plan the article section "
                           "layout based on an initial research summary.\n ",
            },
            {
                "role": "user",
                "content": self._format_planning_instructions(
                    initial_research, include_human_feedback, human_feedback, max_sections,
                    financial_data, industry_peers, is_financial,
                ),
            },
        ]

    def _format_planning_instructions(self, initial_research: str, include_human_feedback: bool,
                                      human_feedback: Optional[str], max_sections: int,
                                      financial_data: dict = None, industry_peers: list = None,
                                      is_financial: bool = False) -> str:
        """Format the instructions for research planning.

        When financial_data is present, uses a specialized 8-section financial
        research report structure. Otherwise falls back to the generic layout.
        """
        today = datetime.now().strftime('%d/%m/%Y')
        feedback_instruction = (
            f"Human feedback: {human_feedback}. You must plan the sections based on the human feedback."
            if include_human_feedback and human_feedback and human_feedback != 'no'
            else ''
        )
        language = self.headers.get("language", "chinese") if self.headers else "chinese"
        language_name = {
            "chinese": "Chinese (中文)",
            "english": "English",
            "japanese": "Japanese (日本語)",
            "korean": "Korean (한국어)",
        }.get(language, "Chinese (中文)")

        # --- Phase 2: Financial report 8-section structure ---
        if is_financial and financial_data:
            ticker = financial_data.get("ticker", "Unknown")
            overview = financial_data.get("overview", {})
            company_name = overview.get("name", ticker)
            sector = overview.get("sector", "")
            industry = overview.get("industry", "")

            # Build peer context for the prompt
            peer_context = ""
            if industry_peers:
                peer_lines = [f"  {p.get('ticker','?')} ({p.get('name','')}) "
                              f"| PE: {p.get('pe','-')} "
                              f"| PB: {p.get('pb','-')} "
                              f"| ROE: {p.get('roe','-')}%"
                              for p in industry_peers[:6]]
                peer_context = "Comparable companies:\n" + "\n".join(peer_lines)

            return f"""Today's date is {today}
Research summary report: '{initial_research}'
{feedback_instruction}

【Financial Report Mode】
You are the chief editor of a financial research report. Your task is to plan a professional
financial analysis report outline for {company_name} ({ticker}).

Company Sector: {sector} / {industry}
{peer_context}

【MANDATORY 8-Section Structure】
You MUST generate exactly 8 section headers. Use these EXACT Chinese titles —
do NOT rephrase, combine, skip, or change the order:

1. 投资摘要 — Core thesis, rating (Buy/Hold/Sell), key catalysts for {company_name}
2. 宏观经济环境 — GDP growth, interest rates, policy impacts on {sector} sector
3. 行业竞争格局 — Porter's Five Forces, market concentration, {ticker}'s position vs peers
4. 公司深度分析 — Business model, competitive moat, revenue drivers for {company_name}
5. 财务数据分析 — Revenue trends, profit margins, cash flow, balance sheet health
6. 估值分析 — PE/PB/ROE comparison with peers, DCF discussion, target price range
7. 风险提示 — Macro risks, industry risks, company-specific risks for {company_name}
8. 投资建议 — Target price, investment strategy, position sizing recommendation

You must write the title in {language_name}.
Section headers MUST be exactly the 8 titles listed above — do not add company name, ticker, or any prefix/suffix.
You must return nothing but a JSON with the fields 'title' (str) and
'sections' (exactly {max_sections} section headers) with the following structure:
'{{"title": "string research title", "date": "today's date",
"sections": ["section header 1", "section header 2", ... "section header 8"]}}'."""

        # --- Original generic mode (no financial data) ---
        return f"""Today's date is {today}
                   Research summary report: '{initial_research}'
                   {feedback_instruction}
                   \nYour task is to generate an outline of sections headers for the research project
                   based on the research summary report above.
                   You must generate a maximum of {max_sections} section headers.
                   You must focus ONLY on related research topics for subheaders and do NOT include introduction, conclusion and references.
                   You MUST write the title and all section headers in {language_name}.
                   You must return nothing but a JSON with the fields 'title' (str) and 
                   'sections' (maximum {max_sections} section headers) with the following structure:
                   '{{title: string research title, date: today's date, 
                   sections: ['section header 1', 'section header 2', 'section header 3' ...]}}'."""

    def _initialize_agents(self) -> Dict[str, any]:
        """Initialize the research, reviewer, and reviser skills."""
        return {
            "research": ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "reviewer": ReviewerAgent(self.websocket, self.stream_output, self.headers),
            "reviser": ReviserAgent(self.websocket, self.stream_output, self.headers),
        }

    def _create_workflow(self) -> StateGraph:
        """Create the workflow for the research process."""
        agents = self._initialize_agents()
        workflow = StateGraph(DraftState)

        workflow.add_node("researcher", agents["research"].run_depth_research)
        workflow.add_node("reviewer", agents["reviewer"].run)
        workflow.add_node("reviser", agents["reviser"].run)

        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "reviewer")
        workflow.add_edge("reviser", "reviewer")
        workflow.add_conditional_edges(
            "reviewer",
            lambda draft: "accept" if draft["review"] is None else "revise",
            {"accept": END, "revise": "reviser"},
        )

        return workflow

    def _log_parallel_research(self, queries: List[str]) -> None:
        """Log the start of parallel research tasks."""
        if self.websocket and self.stream_output:
            asyncio.create_task(self.stream_output(
                "logs",
                "parallel_research",
                f"Running parallel research for the following queries: {queries}",
                self.websocket,
            ))
        else:
            print_agent_output(
                f"Running the following research tasks in parallel: {queries}...",
                agent="EDITOR",
            )

    def _create_task_input(self, research_state: Dict[str, any], query: str, title: str) -> Dict[str, any]:
        """Create the input for a single research task.

        Passes financial context from the main ResearchState into each subgraph's
        DraftState so the Reviewer and Reviser can access financial data for validation.
        """
        # --- Phase 2: Pass financial context to subgraph ---
        financial_data = research_state.get("financial_data") or {}
        industry_peers = research_state.get("industry_peers") or []
        financial_context = {}
        if financial_data:
            overview = financial_data.get("overview", {})
            financial_context = {
                "ticker": financial_data.get("ticker", ""),
                "company_name": overview.get("name", ""),
                "sector": overview.get("sector", ""),
                "industry": overview.get("industry", ""),
                "pe_ratio": overview.get("pe_ratio"),
                "pb_ratio": overview.get("pb_ratio"),
                "roe": overview.get("roe"),
                "revenue_growth": overview.get("revenue_growth"),
                "profit_margin": overview.get("profit_margin"),
                "market_cap": overview.get("market_cap"),
                "dividend_yield": overview.get("dividend_yield"),
                "peers": industry_peers,
            }
        # ---------------------------------------------------

        return {
            "task": research_state.get("task"),
            "topic": query,
            "title": title,
            "headers": self.headers,
            "financial_context": financial_context,
        }
