"""Financial Data Retriever for GPT Researcher.

This module provides the FinancialDataRetriever class that integrates
Yahoo Finance data into the GPT Researcher retrieval pipeline.
It wraps FinancialDataTool as a first-class retriever alongside
Tavily, Google, and other search backends.

Architecture:
    get_retrievers() → FinancialDataRetriever → FinancialDataTool → FMP API
                     → TavilyRetriever           → Tavily API
                     → GoogleRetriever           → Google API

Usage:
    retriever = FinancialDataRetriever(query="分析苹果公司(AAPL)")
    results = retriever.search(max_results=5)
    # Returns structured financial data in search result format

Note:
    search() is synchronous (matches existing retriever protocol).
    FinancialDataTool's async methods are bridged via asyncio.run()
    in a thread executor to avoid event-loop conflicts.
"""

import asyncio
import concurrent.futures
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class FinancialDataRetriever:
    """Financial data retriever that fetches stock fundamentals via Yahoo Finance.

    Implements the standard retriever protocol expected by ResearchConductor:
    - __init__(query, headers, query_domains, **kwargs)
    - search(max_results) → List[dict]

    When a ticker is detected in the query, this retriever fetches:
    1. Stock overview (price, PE, PB, ROE, market cap, etc.)
    2. Financial statements (quarterly/annual revenue & earnings)
    3. Industry peers (comparable companies with key metrics)

    If no ticker is found, returns an empty list (no-op for non-financial queries).
    """

    def __init__(
        self,
        query: str,
        headers: Optional[dict] = None,
        query_domains: Optional[list] = None,
        **kwargs,
    ):
        """Initialize the FinancialDataRetriever.

        Args:
            query: The search query string (may contain a ticker symbol).
            headers: Additional HTTP headers (unused by this retriever).
            query_domains: Domains to filter (unused by this retriever).
            **kwargs: Additional keyword arguments (for compatibility).
        """
        self.query = query
        self.headers = headers or {}
        self.query_domains = query_domains
        self._ticker: Optional[str] = None
        self._results_cache: Optional[List[dict]] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, max_results: int = 5) -> List[dict]:
        """Execute financial data search (synchronous).

        Args:
            max_results: Maximum number of result items to return.

        Returns:
            List of search result dicts with keys:
                - href: Yahoo Finance page URL
                - title: Descriptive label (e.g. "公司概览: AAPL")
                - body: Formatted financial data text
            Returns empty list if no ticker detected or data unavailable.
        """
        if self._results_cache is not None:
            return self._results_cache

        ticker = self._extract_ticker()
        if not ticker:
            self._results_cache = []
            return []

        # A 股/港股纯数字 ticker → 跳过，FMP 不支持（每股数据走 XueqiuDataTool）
        if ticker.isdigit() and len(ticker) in (5, 6):
            self._results_cache = []
            return []

        try:
            from multi_agents.components.financial_data import FinancialDataTool

            tool = FinancialDataTool(ticker)

            # Bridge async FinancialDataTool to sync retriever protocol.
            # Runs in a thread executor to avoid event-loop nesting issues.
            overview, statements, peers = self._run_async(
                self._fetch_all(tool)
            )

            results = self._format_results(ticker, overview, statements, peers)
            self._results_cache = results
            return results

        except ImportError:
            logger.warning(
                "[FinancialDataRetriever] financial_data module not found — "
                "financial data unavailable."
            )
            self._results_cache = []
            return []
        except Exception as e:
            logger.warning(
                f"[FinancialDataRetriever] Failed to fetch data for {ticker}: {e}"
            )
            self._results_cache = []
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_all(self, tool) -> tuple:
        """Fetch all financial data from FinancialDataTool in parallel."""
        return await asyncio.gather(
            tool.get_stock_overview(),
            tool.get_financial_statements(),
            tool.get_industry_peers(),
        )

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from within a synchronous context.

        Uses a thread executor when called from an already-running
        event loop (which is the case in ResearchConductor._search).
        Falls back to asyncio.run() when no loop is running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running — safe to use asyncio.run()
            return asyncio.run(coro)

        # Event loop is already running — delegate to a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=60)

    def _extract_ticker(self) -> Optional[str]:
        """Extract ticker symbol from the query string.

        Uses the same extract_ticker_from_query() logic as the
        multi_agents component, imported lazily.
        """
        if self._ticker is not None:
            return self._ticker

        try:
            from multi_agents.components.financial_data import extract_ticker_from_query

            self._ticker = extract_ticker_from_query(self.query)
        except ImportError:
            self._ticker = None

        return self._ticker

    def _format_results(
        self,
        ticker: str,
        overview: dict,
        statements: dict,
        peers: list,
    ) -> List[dict]:
        """Format raw Yahoo Finance data into search result format.

        Each result dict follows the standard retriever format:
        {"href": str, "title": str, "body": str}

        The href field points to the stock's Yahoo Finance page for
        traceability and source verification.
        """
        yf_url = f"https://finance.yahoo.com/quote/{ticker}"
        results = []

        # ---- Result 1: Stock Overview ----
        if overview:
            name = overview.get("name", ticker)
            price = overview.get("price", "N/A")
            currency = overview.get("currency", "USD")
            market_cap = overview.get("market_cap", "N/A")
            pe = overview.get("pe_ratio", "N/A")
            pb = overview.get("pb_ratio", "N/A")
            roe = overview.get("roe", "N/A")
            rev_growth = overview.get("revenue_growth", "N/A")
            profit_margin = overview.get("profit_margin", "N/A")
            debt_equity = overview.get("debt_to_equity", "N/A")
            dividend = overview.get("dividend_yield", "N/A")
            beta = overview.get("beta", "N/A")
            sector = overview.get("sector", "")
            industry = overview.get("industry", "")
            summary = overview.get("summary", "")
            high_52w = overview.get("fifty_two_week_high", "N/A")
            low_52w = overview.get("fifty_two_week_low", "N/A")

            body_parts = [
                f"公司名称: {name}",
                f"股票代码: {ticker}",
                f"最新股价: {price} {currency}",
                f"市值: {self._fmt_num(market_cap)}",
                f"52周最高: {high_52w} / 52周最低: {low_52w}",
                f"行业: {sector} / {industry}",
                "",
                "【估值指标】",
                f"市盈率(PE): {pe}",
                f"市净率(PB): {pb}",
                f"净资产收益率(ROE): {roe}%",
                f"营收增长率: {rev_growth}%",
                f"净利润率: {profit_margin}%",
                f"资产负债率: {debt_equity}%",
                f"股息率: {dividend}%",
                f"Beta系数: {beta}",
            ]

            if summary:
                body_parts.append("")
                body_parts.append(f"【业务概要】{summary[:500]}")

            results.append({
                "href": yf_url,
                "title": f"公司概览: {name} ({ticker})",
                "body": "\n".join(body_parts),
            })

        # ---- Result 2: Financial Statements ----
        if statements:
            stmt_body = self._format_statements(statements)
            if stmt_body:
                results.append({
                    "href": f"{yf_url}/financials",
                    "title": f"财务报表: {ticker}",
                    "body": stmt_body,
                })

        # ---- Result 3: Industry Peers ----
        if peers:
            peer_body = self._format_peers(peers, ticker, overview)
            if peer_body:
                results.append({
                    "href": yf_url,
                    "title": f"行业对比: {ticker} 可比公司",
                    "body": peer_body,
                })

        logger.info(
            f"[FinancialDataRetriever] Formatted {len(results)} results for {ticker}"
        )
        return results

    def _format_statements(self, statements: dict) -> str:
        """Format financial statement data into readable text."""
        lines = []

        # Quarterly revenue
        quarterly_rev = statements.get("quarterly_revenue", [])
        if quarterly_rev:
            lines.append("【季度营收】")
            for q in quarterly_rev:
                rev = self._fmt_num(q.get("revenue"))
                lines.append(f"  {q.get('date', '?')}: {rev}")
            lines.append("")

        # Quarterly earnings
        quarterly_earn = statements.get("quarterly_earnings", [])
        if quarterly_earn:
            lines.append("【季度净利润】")
            for q in quarterly_earn:
                earn = self._fmt_num(q.get("earnings"))
                lines.append(f"  {q.get('date', '?')}: {earn}")
            lines.append("")

        # Annual revenue
        annual_rev = statements.get("annual_revenue", [])
        if annual_rev:
            lines.append("【年度营收】")
            for a in annual_rev:
                rev = self._fmt_num(a.get("revenue"))
                lines.append(f"  {a.get('date', '?')}: {rev}")
            lines.append("")

        # Annual earnings
        annual_earn = statements.get("annual_earnings", [])
        if annual_earn:
            lines.append("【年度净利润】")
            for a in annual_earn:
                earn = self._fmt_num(a.get("earnings"))
                lines.append(f"  {a.get('date', '?')}: {earn}")
            lines.append("")

        # Cash flow
        cf = statements.get("cash_flow", {})
        if cf:
            lines.append("【现金流】")
            if cf.get("operating_cash_flow"):
                lines.append(f"  经营活动现金流: {self._fmt_num(cf['operating_cash_flow'])}")
            if cf.get("free_cash_flow"):
                lines.append(f"  自由现金流: {self._fmt_num(cf['free_cash_flow'])}")
            lines.append("")

        # Balance sheet
        bs = statements.get("balance_sheet", {})
        if bs:
            lines.append("【资产负债表】")
            if bs.get("total_assets"):
                lines.append(f"  总资产: {self._fmt_num(bs['total_assets'])}")
            if bs.get("total_debt"):
                lines.append(f"  总负债: {self._fmt_num(bs['total_debt'])}")
            if bs.get("total_equity"):
                lines.append(f"  股东权益: {self._fmt_num(bs['total_equity'])}")

        return "\n".join(lines) if lines else ""

    def _format_peers(
        self, peers: list, ticker: str, overview: dict
    ) -> str:
        """Format peer comparison data into a comparison table.

        Includes the target company as the first row for easy comparison.
        """
        lines = ["【可比公司估值对比】", ""]

        # Include target company as first row
        if overview:
            name = overview.get("name", ticker)
            pe = overview.get("pe_ratio", "-")
            pb = overview.get("pb_ratio", "-")
            roe = overview.get("roe", "-")
            rev_growth = overview.get("revenue_growth", "-")
            market_cap = self._fmt_num(overview.get("market_cap"))
            lines.append(
                f"  {ticker} ({name})"
                f" | 市值: {market_cap}"
                f" | PE: {pe}"
                f" | PB: {pb}"
                f" | ROE: {roe}%"
                f" | 营收增速: {rev_growth}%"
            )
            lines.append("  " + "-" * 60)

        for p in peers:
            p_ticker = p.get("ticker", "?")
            p_name = p.get("name", "")
            p_pe = p.get("pe", "-")
            p_pb = p.get("pb", "-")
            p_roe = p.get("roe", "-")
            p_growth = p.get("revenue_growth", "-")
            p_mcap = self._fmt_num(p.get("market_cap"))
            lines.append(
                f"  {p_ticker} ({p_name})"
                f" | 市值: {p_mcap}"
                f" | PE: {p_pe}"
                f" | PB: {p_pb}"
                f" | ROE: {p_roe}%"
                f" | 营收增速: {p_growth}%"
            )

        return "\n".join(lines)

    @staticmethod
    def _fmt_num(value) -> str:
        """Format a numeric value for display.

        Large numbers are converted to billions (B) or millions (M).
        """
        if value is None:
            return "N/A"
        try:
            num = float(value)
            if abs(num) >= 1e12:
                return f"{num / 1e12:.2f}T"
            elif abs(num) >= 1e9:
                return f"{num / 1e9:.2f}B"
            elif abs(num) >= 1e6:
                return f"{num / 1e6:.2f}M"
            elif abs(num) >= 1e3:
                return f"{num / 1e3:.2f}K"
            return f"{num:.2f}"
        except (TypeError, ValueError):
            return str(value)
