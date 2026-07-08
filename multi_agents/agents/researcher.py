import asyncio
import logging
from typing import Optional
from gpt_researcher import GPTResearcher
from colorama import Fore, Style
from .utils.views import print_agent_output
from .utils.llms import call_model
from ..components.financial_data import FinancialDataTool, extract_ticker_from_query
from ..components.xueqiu_finance import XueqiuDataTool

logger = logging.getLogger(__name__)


class ResearchAgent:
    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone

    async def research(self, query: str, research_report: str = "research_report",
                       parent_query: str = "", verbose=True, source="web", tone=None, headers=None):
        # Initialize the researcher
        researcher = GPTResearcher(query=query, report_type=research_report, parent_query=parent_query,
                                   verbose=verbose, report_source=source, tone=tone, websocket=self.websocket, headers=self.headers)
        # 设置报告语言：优先 headers.language，默认中文
        # GPTResearcher.cfg.language 控制 generate_prompt / generate_report_introduction
        # / generate_report_conclusion 等所有报告生成 prompt 的语言
        language = self.headers.get("language") or "chinese"
        researcher.cfg.language = language
        # Conduct research on the given query
        # 获取上下文
        await researcher.conduct_research()
        # Write the report
        # 根据上下文给LLM生成报告
        report = await researcher.write_report()

        return report

    async def run_subtopic_research(self, parent_query: str, subtopic: str, verbose: bool = True, source="web", headers=None):
        try:
            report = await self.research(parent_query=parent_query, query=subtopic,
                                         research_report="subtopic_report", verbose=verbose, source=source, tone=self.tone, headers=None)
        except Exception as e:
            print(f"{Fore.RED}Error in researching topic {subtopic}: {e}{Style.RESET_ALL}")
            report = None
        return {subtopic: report}

    async def run_initial_research(self, research_state: dict):
        task = research_state.get("task")
        query = task.get("query")
        source = task.get("source", "web")

        # --- Phase 2: 金融数据注入 ---
        ticker = extract_ticker_from_query(query)

        # LLM fallback：正则/硬编码未命中时，用大模型判断
        if not ticker:
            ticker = await self._extract_ticker_via_llm(query, task)
        financial_data = {}
        industry_peers = []

        if ticker:
            print_agent_output(
                f"检测到股票代码 {ticker}，开始获取金融数据...", agent="RESEARCHER")
            try:
                # --- Phase 2: 双源路由（格式验证 + 路由）---
                # 5-6位纯数字 → 雪球（A股6位/HK5位），其他 → FMP（美股）
                is_ashare = (
                    ticker.isdigit()
                    and len(ticker) in (5, 6)
                )
                if is_ashare:
                    tool = XueqiuDataTool(ticker)
                else:
                    tool = FinancialDataTool(ticker)
                overview, statements, peers = await asyncio.gather(
                    tool.get_stock_overview(),
                    tool.get_financial_statements(),
                    tool.get_industry_peers(),
                )
                # Only build financial_data if overview has actual data.
                # An empty dict means yfinance failed (rate-limit, timeout, etc.)
                # — don't activate financial mode downstream with N/A values.
                if overview:
                    # DEBUG: check what overview actually contains
                    logger.info(
                        f"[Researcher DEBUG] overview keys={list(overview.keys())}, "
                        f"name={overview.get('name')!r}, pe={overview.get('pe_ratio')}"
                    )
                    financial_data = {
                        "ticker": ticker,
                        "overview": overview,
                        "statements": statements,
                    }
                    industry_peers = peers

                    # --- LLM 生成同行列表（peers 为空时）---
                    if not industry_peers and is_ashare:
                        industry_peers = await self._get_peers_via_llm(
                            overview, task, XueqiuDataTool
                        )
                    # ----------------------------------------

                    # 追加金融提示到 query
                    name = overview.get("name", ticker)
                    query = (
                        f"{query}\n\n"
                        f"请重点关注 {ticker}（{name}）的财务数据和行业竞争格局。"
                    )
                    print_agent_output(
                        f"金融数据获取完成: {ticker} | 同行 {len(industry_peers)} 家",
                        agent="RESEARCHER",
                    )
            except Exception as e:
                logger.warning(f"金融数据获取失败 ({ticker}): {e}")
                print_agent_output(
                    f"金融数据获取失败 ({ticker}): {e}，回退到通用模式",
                    agent="RESEARCHER",
                )
        # --- 金融数据注入完毕 ---

        if self.websocket and self.stream_output:
            await self.stream_output("logs", "initial_research", f"Running initial research on the following query: {query}", self.websocket)
        else:
            print_agent_output(f"Running initial research on the following query: {query}", agent="RESEARCHER")

        result = {
            "task": task,
            "initial_research": await self.research(
                query=query,
                verbose=task.get("verbose"),
                source=source,
                tone=self.tone,
                headers=self.headers,
            ),
            "financial_data": financial_data,
            "industry_peers": industry_peers,
        }
        return result

    async def _extract_ticker_via_llm(self, query: str, task: dict) -> Optional[str]:
        """正则未命中时，用 LLM 判断是否为金融查询并提取股票代码。

        返回 str（股票代码）或 None（非金融查询）。
        成本 ~$0.001，仅在前置正则失败时触发。
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个金融查询分类器。判断用户的查询是否涉及某家上市公司的"
                    "财务分析或投资研究。\n"
                    "- 如果是：返回 JSON {\"ticker\": \"股票代码\"}，代码必须是标准格式"
                    "（美股如AAPL/TSLA，A股6位数字如600519/000001，港股5位数字如00700/00939）\n"
                    "- 如果否：返回 JSON {\"ticker\": null}\n"
                    "不要解释，只返回 JSON。"
                ),
            },
            {
                "role": "user",
                "content": f"查询：{query}",
            },
        ]
        try:
            response = await call_model(prompt, task.get("model"), response_format="json")
            import json
            data = json.loads(response) if isinstance(response, str) else response
            ticker = data.get("ticker") if isinstance(data, dict) else None
            if ticker:
                print_agent_output(
                    f"LLM 识别到金融查询，股票代码 {ticker}",
                    agent="RESEARCHER",
                )
            return ticker
        except Exception as e:
            logger.debug(f"LLM ticker extraction failed: {e}")
            return None

    async def _get_peers_via_llm(self, overview: dict, task: dict, ToolClass) -> list[dict]:
        """LLM 生成 2-3 个同行 ticker，然后查雪球拿 PE/PB/ROE。

        Args:
            overview: get_stock_overview 返回的 dict
            task: 任务配置 dict
            ToolClass: XueqiuDataTool 或 FinancialDataTool 类

        Returns:
            [{"ticker":"000858","name":"五粮液","pe":15.2,...}, ...]
        """
        name = overview.get("name", "")
        sector = overview.get("sector", "")
        industry = overview.get("industry", "")
        ticker = overview.get("ticker", "")

        prompt = [
            {
                "role": "system",
                "content": (
                    "你是一个金融行业分析师。根据公司信息，列出 2-3 家同行业可比公司"
                    "的股票代码。只返回 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"公司：{name}（{ticker}）\n"
                    f"行业：{sector} / {industry}\n"
                    f"请返回 2-3 家同行业对手的股票代码。"
                    f"返回 JSON：{{\"peers\": [\"600519\", \"000858\", ...]}}"
                    f"代码必须是标准的 6 位 A 股数字或美股字母代码。"
                    f"只返回 JSON，不要解释。"
                ),
            },
        ]
        try:
            response = await call_model(prompt, task.get("model"), response_format="json")
            import json
            data = json.loads(response) if isinstance(response, str) else response
            peer_tickers = data.get("peers", []) if isinstance(data, dict) else []

            # 逐个查雪球
            peers = []
            for pt in peer_tickers[:3]:
                try:
                    p_tool = ToolClass(str(pt))
                    p_ov = await p_tool.get_stock_overview()
                    if p_ov and p_ov.get("name"):
                        peers.append({
                            "ticker": str(pt),
                            "name": p_ov.get("name", ""),
                            "pe": p_ov.get("pe_ratio"),
                            "pb": p_ov.get("pb_ratio"),
                            "roe": p_ov.get("roe"),
                            "revenue_growth": p_ov.get("revenue_growth"),
                        })
                except Exception:
                    continue

            if peers:
                print_agent_output(
                    f"LLM 生成同行列表: {[p['ticker'] for p in peers]} | "
                    f"PE: {[p['pe'] for p in peers]}",
                    agent="RESEARCHER",
                )
            return peers
        except Exception as e:
            logger.debug(f"LLM peer generation failed: {e}")
            return []

    async def run_depth_research(self, draft_state: dict):
        task = draft_state.get("task")
        topic = draft_state.get("topic")
        parent_query = task.get("query")
        source = task.get("source", "web")
        verbose = task.get("verbose")
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "depth_research", f"Running in depth research on the following report topic: {topic}", self.websocket)
        else:
            print_agent_output(f"Running in depth research on the following report topic: {topic}", agent="RESEARCHER")
        research_draft = await self.run_subtopic_research(parent_query=parent_query, subtopic=topic,
                                                          verbose=verbose, source=source, headers=self.headers)
        return {"draft": research_draft}