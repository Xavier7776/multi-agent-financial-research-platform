"""
Financial Modeling Prep (FMP) 数据获取模块
--------------------------------------------
独立的金融数据获取工具，通过 FMP REST API 获取美股基本面数据。
从财报数据（利润表、资产负债表）自行计算 PE/PB/ROE 等比率，
避免依赖 FMP 免费版预计算字段不稳定的问题。

使用方式:
    tool = FinancialDataTool("AAPL")
    overview = await tool.get_stock_overview()
    statements = await tool.get_financial_statements()
    peers = await tool.get_industry_peers()
"""

import asyncio
import logging
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
# FMP API 配置
_FMP_API_KEY = os.getenv("FMP_API_KEY")
_FMP_BASE = "https://financialmodelingprep.com/stable"


def _fmp_get(endpoint: str, params: dict = None) -> dict | list | None:
    """同步 GET 请求 FMP API。"""
    params = params or {}
    params["apikey"] = _FMP_API_KEY
    try:
        url = f"{_FMP_BASE}/{endpoint.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(
            f"[FMP] {endpoint} returned {resp.status_code}: {resp.text[:200]}"
        )
        return None
    except requests.RequestException as e:
        logger.warning(f"[FMP] {endpoint} request failed: {e}")
        return None


# --- 纯函数：从 query 文本中提取 ticker ---

_TICKER_PATTERNS: dict[str, str] = {
    "AAPL": "苹果", "GOOGL": "谷歌", "GOOG": "谷歌",
    "MSFT": "微软", "AMZN": "亚马逊", "META": "Meta",
    "NVDA": "英伟达", "TSLA": "特斯拉", "NFLX": "奈飞",
    "AMD": "AMD", "INTC": "英特尔",
    "BABA": "阿里巴巴", "JD": "京东", "PDD": "拼多多",
    "NIO": "蔚来", "BIDU": "百度", "TSM": "台积电",
    "DIS": "迪士尼", "BA": "波音",
    "JPM": "摩根大通", "GS": "高盛", "V": "Visa", "MA": "万事达",
    "WMT": "沃尔玛", "COST": "好市多",
    "KO": "可口可乐", "PEP": "百事", "JNJ": "强生", "PFE": "辉瑞",
}


def extract_ticker_from_query(query: str) -> Optional[str]:
    """从用户查询文本中提取股票代码（美股 + A 股）。"""
    import re

    # --- A 股/港股：数字代码 ---
    ashare_match = re.search(r'(?<![0-9])([63]\d{5}|0\d{5}|688\d{3}|0\d{4})(?![0-9])', query)
    if ashare_match:
        return ashare_match.group(1)

    # --- A 股：中文公司名 → 代码映射 ---
    _ASHARE_NAMES = {
        "贵州茅台": "600519", "茅台": "600519",
        "五粮液": "000858",
        "宁德时代": "300750", "宁德": "300750",
        "比亚迪": "002594",
        "招商银行": "600036", "招行": "600036",
        "中国平安": "601318", "平安": "601318",
        "美的集团": "000333", "美的": "000333",
        "格力电器": "000651", "格力": "000651",
        "海康威视": "002415", "海康": "002415",
        "伊利股份": "600887", "伊利": "600887",
        "恒瑞医药": "600276", "恒瑞": "600276",
        "万科A": "000002", "万科": "000002",
        "中国石油": "601857",
        "兴业银行": "601166", "兴业": "601166",
        "中信证券": "600030", "中信": "600030",
        "立讯精密": "002475", "立讯": "002475",
        "迈瑞医疗": "300760", "迈瑞": "300760",
        "海天味业": "603288", "海天": "603288",
        "长江电力": "600900", "长江": "600900",
        "京东方A": "000725", "京东方": "000725",
        "中芯国际": "688981", "中芯": "688981",
        "腾讯控股": "00700", "腾讯": "00700",
        "美团": "03690", "阿里巴巴": "09988",
        "小米集团": "01810", "小米": "01810",
    }
    name_lower = query.lower()
    for name, tkr in _ASHARE_NAMES.items():
        if name in query or name.lower() in name_lower:
            return tkr

    # --- 美股：括号内大写 ticker ---
    bracket_match = re.search(r'\(([A-Z]{1,5})\)', query)
    if bracket_match:
        ticker = bracket_match.group(1)
        if ticker in _TICKER_PATTERNS or len(ticker) <= 5:
            return ticker

    standalone_match = re.search(r'\b([A-Z]{1,5})\b', query)
    if standalone_match:
        ticker = standalone_match.group(1)
        if ticker in _TICKER_PATTERNS:
            return ticker

    for ticker, name in _TICKER_PATTERNS.items():
        if name in query and ticker not in {"GOOG", "GOOGL"}:
            return ticker

    name_lower = query.lower()
    _NAME_TO_TICKER = {
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
        "amazon": "AMZN", "meta": "META", "nvidia": "NVDA",
        "tesla": "TSLA", "netflix": "NFLX", "amd": "AMD",
        "intel": "INTC",
    }
    for name, ticker in _NAME_TO_TICKER.items():
        if name in name_lower:
            return ticker

    return None


# --- 数据类：FMP 工具封装 ---

class FinancialDataTool:
    """金融数据获取工具（FMP API 后端）。

    从 profile + income-statement + balance-sheet 获取原始数据，
    自行计算 PE/PB/ROE/营收增速/利润率/负债率，保证数据一致性。
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # 内部：请求 + 重试
    # ------------------------------------------------------------------

    def _fetch(self, endpoint: str, params: dict = None, max_retries=2) -> dict | list | None:
        """HTTP GET 带重试。"""
        for attempt in range(max_retries):
            result = _fmp_get(endpoint, params)
            if result is not None and (not isinstance(result, list) or result):
                return result
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
        return None

    def _fetch_profile(self) -> dict:
        data = self._fetch("/profile", {"symbol": self.ticker})
        if data is None:
            logger.warning(f"[FinancialData] _fetch_profile({self.ticker}) returned None (API failure)")
            return {}
        if isinstance(data, list) and data:
            return data[0]
        logger.warning(f"[FinancialData] _fetch_profile({self.ticker}) unexpected result: {type(data).__name__}")
        return {}

    def _fetch_income(self, limit=4) -> list:
        data = self._fetch(
            "/income-statement",
            {"symbol": self.ticker, "period": "quarter", "limit": limit},
        )
        return data if isinstance(data, list) else []

    def _fetch_balance_sheet(self, limit=1) -> list:
        data = self._fetch(
            "/balance-sheet-statement",
            {"symbol": self.ticker, "period": "quarter", "limit": limit},
        )
        return data if isinstance(data, list) else []

    def _fetch_cash_flow(self, limit=1) -> list:
        data = self._fetch(
            "/cashflow-statement",
            {"symbol": self.ticker, "period": "quarter", "limit": limit},
        )
        return data if isinstance(data, list) else []

    def _fetch_peer_profile(self, symbol: str) -> dict:
        data = self._fetch("/profile", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return {}

    # ------------------------------------------------------------------
    # 1. 股票基本面概览（从财报计算比率）
    # ------------------------------------------------------------------

    async def get_stock_overview(self) -> dict:
        cache_key = "stock_overview"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            #balance sheet
            profile, income_q, bs_q = await asyncio.gather(
                asyncio.to_thread(self._fetch_profile),
                asyncio.to_thread(self._fetch_income, 5),
                asyncio.to_thread(self._fetch_balance_sheet, 1),
            )

            # 52 周范围: FMP 格式 "199.26-317.4"
            range_str = profile.get("range", "")

            # 如果 profile 没拿到公司名，说明 FMP 返回了空数据 → 不进入金融模式
            if not profile.get("companyName"):
                self._cache[cache_key] = {}
                return {}

            low_52, high_52 = None, None

            # 从财报计算比率
            mkt_cap = profile.get("marketCap")
            latest_income = income_q[0] if income_q else {}
            latest_bs = bs_q[0] if bs_q else {}

            net_income = _safe_float(latest_income.get("netIncome"))
            revenue = _safe_float(latest_income.get("revenue"))
            total_equity = _safe_float(latest_bs.get("totalStockholdersEquity"))
            total_liab = _safe_float(latest_bs.get("totalLiabilities"))

            # TTM 净利润 = 最近 4 个季度净利润之和
            ttm_ni = None
            if len(income_q) >= 4:
                ni_list = [_safe_float(q.get("netIncome")) for q in income_q[:4]]
                if all(ni is not None for ni in ni_list):
                    ttm_ni = sum(ni_list)

            # PE = 市值 / TTM 净利润
            pe = None
            if mkt_cap and ttm_ni and ttm_ni > 0:
                pe = round(mkt_cap / ttm_ni, 2)

            # PB = 市值 / 净资产
            pb = None
            if mkt_cap and total_equity and total_equity > 0:
                pb = round(mkt_cap / total_equity, 2)

            # ROE = TTM 净利润 / 净资产
            roe = None
            if ttm_ni and total_equity and total_equity > 0:
                roe = round(ttm_ni / total_equity * 100, 2)

            # 营收增速 = (当前季营收 - 上年同期营收) / 上年同期
            rev_growth = None
            prev_year_rev = None
            if len(income_q) >= 5:
                prev_year_rev = _safe_float(income_q[4].get("revenue"))
            if revenue and prev_year_rev and prev_year_rev > 0:
                rev_growth = round((revenue - prev_year_rev) / prev_year_rev * 100, 2)

            # 净利润率
            profit_margin = None
            if net_income and revenue and revenue > 0:
                profit_margin = round(net_income / revenue * 100, 2)

            # 负债率
            debt_equity = None
            if total_liab is not None and total_equity and total_equity > 0:
                debt_equity = round(total_liab / total_equity * 100, 2)

            result = {
                "ticker": self.ticker,
                "name": profile.get("companyName") or profile.get("symbol", ""),
                "price": profile.get("price"),
                "market_cap": mkt_cap,
                "pe_ratio": pe,
                "pb_ratio": pb,
                "roe": roe,
                "revenue_growth": rev_growth,
                "debt_to_equity": debt_equity,
                "profit_margin": profit_margin,
                "dividend_yield": None,  # FMP 免费版不提供股息率
                "beta": profile.get("beta"),
                "sector": profile.get("sector", ""),
                "industry": profile.get("industry", ""),
                "summary": profile.get("description", ""),
                "currency": profile.get("currency", "USD"),
                "exchange": profile.get("exchangeShortName") or profile.get("exchange", ""),
                "fifty_two_week_high": high_52,
                "fifty_two_week_low": low_52,
            }

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"[FinancialData] get_stock_overview({self.ticker}) 失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # 2. 财务报表数据
    # ------------------------------------------------------------------

    async def get_financial_statements(self) -> dict:
        cache_key = "financial_statements"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            income_q, cf_q, bs_q = await asyncio.gather(
                asyncio.to_thread(self._fetch_income, 4),
                asyncio.to_thread(self._fetch_cash_flow, 1),
                asyncio.to_thread(self._fetch_balance_sheet, 1),
            )

            q_rev, q_earn = [], []
            for stmt in income_q:
                q_rev.append({"date": stmt.get("date", ""), "revenue": _safe_float(stmt.get("revenue"))})
                q_earn.append({"date": stmt.get("date", ""), "earnings": _safe_float(stmt.get("netIncome"))})

            cash_flow = {}
            if cf_q:
                cf = cf_q[0]
                cash_flow = {
                    "operating_cash_flow": _safe_float(cf.get("operatingCashFlow")),
                    "free_cash_flow": _safe_float(cf.get("freeCashFlow")),
                    "capital_expenditure": _safe_float(cf.get("capitalExpenditure")),
                }

            balance_sheet = {}
            if bs_q:
                bs = bs_q[0]
                balance_sheet = {
                    "total_assets": _safe_float(bs.get("totalAssets")),
                    "total_debt": _safe_float(bs.get("totalDebt")),
                    "total_equity": _safe_float(bs.get("totalStockholdersEquity")),
                    "current_assets": _safe_float(bs.get("totalCurrentAssets")),
                    "current_liabilities": _safe_float(bs.get("totalCurrentLiabilities")),
                }

            result = {
                "quarterly_revenue": q_rev,
                "quarterly_earnings": q_earn,
                "annual_revenue": [],
                "annual_earnings": [],
                "cash_flow": cash_flow,
                "balance_sheet": balance_sheet,
            }
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.debug(f"[FinancialData] get_financial_statements({self.ticker}) 失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # 3. 同行业可比公司
    # ------------------------------------------------------------------

    async def get_industry_peers(self) -> list[dict]:
        cache_key = "industry_peers"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # 确保 overview 已缓存（避免并发调 _fetch_profile 两次）
            if "stock_overview" not in self._cache:
                await self.get_stock_overview()

            overview_cache = self._cache.get("stock_overview", {})
            industry = overview_cache.get("industry", "")
            if not industry:
                profile = await asyncio.to_thread(self._fetch_profile)
                industry = profile.get("industry", "") if profile else ""

            symbol_list = _get_fallback_peers(self.ticker, industry)
            # DEBUG
            logger.info(
                f"[DEBUG peers] ticker={self.ticker!r}, industry={industry!r}, "
                f"found={len(symbol_list)} peers"
            )
            if not symbol_list:
                self._cache[cache_key] = []
                return []

            limited = symbol_list[:6]
            tasks = [self._fetch_peer_info(s) for s in limited]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            peers = [r for r in results if isinstance(r, dict) and r]

            self._cache[cache_key] = peers
            return peers

        except Exception as e:
            logger.warning(f"[FinancialData] get_industry_peers({self.ticker}) 失败: {e}")
            return []

    async def _fetch_peer_info(self, symbol: str) -> dict:
        """获取同行的完整估值指标（profile + income + balance sheet）。"""
        try:
            profile, income_q, bs_q = await asyncio.gather(
                asyncio.to_thread(self._fetch_peer_profile, symbol),
                asyncio.to_thread(
                    lambda s=symbol: self._fetch(f"/income-statement", {"symbol": s, "period": "quarter", "limit": 5}) or [],
                ),
                asyncio.to_thread(
                    lambda s=symbol: self._fetch(f"/balance-sheet-statement", {"symbol": s, "period": "quarter", "limit": 1}) or [],
                ),
            )

            if not profile or not profile.get("companyName"):
                return {}

            mkt_cap = profile.get("marketCap")
            latest_bs = bs_q[0] if isinstance(bs_q, list) and bs_q else {}
            total_equity = _safe_float(latest_bs.get("totalStockholdersEquity"))

            # TTM 净利润
            ttm_ni = None
            if isinstance(income_q, list) and len(income_q) >= 4:
                ni_list = [_safe_float(q.get("netIncome")) for q in income_q[:4]]
                if all(ni is not None for ni in ni_list):
                    ttm_ni = sum(ni_list)

            # 营收增速
            rev_growth = None
            if isinstance(income_q, list) and len(income_q) >= 5:
                rev = _safe_float(income_q[0].get("revenue"))
                prev_rev = _safe_float(income_q[4].get("revenue"))
                if rev and prev_rev and prev_rev > 0:
                    rev_growth = round((rev - prev_rev) / prev_rev * 100, 2)

            pe = round(mkt_cap / ttm_ni, 2) if mkt_cap and ttm_ni and ttm_ni > 0 else None
            pb = round(mkt_cap / total_equity, 2) if mkt_cap and total_equity and total_equity > 0 else None
            roe = round(ttm_ni / total_equity * 100, 2) if ttm_ni and total_equity and total_equity > 0 else None

            return {
                "ticker": symbol,
                "name": profile.get("companyName", ""),
                "market_cap": mkt_cap,
                "pe": pe,
                "pb": pb,
                "roe": roe,
                "revenue_growth": rev_growth,
            }
        except Exception:
            return {}


# --- 辅助函数 ---

def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        return round(v, 2) if v == v else None
    except (TypeError, ValueError):
        return None


# 行业 → 可比公司映射（按 FMP profile.industry 字段分组）
# 比 ticker→peers 更灵活：同一行业的任何股票自动获得同行。
_INDUSTRY_PEERS: dict[str, list[str]] = {
    # 消费电子
    "Consumer Electronics": ["AAPL", "SONO", "HEAR", "VUZI", "GPRO"],
    # 软件
    "Software—Infrastructure": ["MSFT", "ORCL", "CRM", "ADBE", "PANW"],
    "Software—Application": ["ADBE", "CRM", "INTU", "NOW", "WDAY"],
    # 互联网
    "Internet Content & Information": ["GOOGL", "META", "SNAP", "PINS", "BIDU"],
    "Internet Retail": ["AMZN", "JD", "BABA", "ETSY", "W"],
    # 半导体
    "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO"],
    "Semiconductor Equipment & Materials": ["ASML", "AMAT", "LRCX", "KLAC", "TSM"],
    # 汽车
    "Auto Manufacturers": ["TSLA", "F", "GM", "TM", "RIVN"],
    # 娱乐/传媒
    "Entertainment": ["NFLX", "DIS", "WBD", "CMCSA", "SPOT"],
    # 银行
    "Banks—Diversified": ["JPM", "BAC", "WFC", "C", "GS"],
    # 医药
    "Drug Manufacturers—General": ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
    # 零售
    "Discount Stores": ["WMT", "COST", "TGT", "DG", "DLTR"],
    # 支付
    "Credit Services": ["V", "MA", "AXP", "PYPL", "SQ"],
    "Financial Data & Stock Exchanges": ["SPGI", "MCO", "MSCI", "NDAQ", "ICE"],
    # 云计算/IT服务
    "Information Technology Services": ["IBM", "ACN", "INFY", "CTSH", "DXC"],
    # 电子制造/硬件
    "Computer Hardware": ["DELL", "HPQ", "NTAP", "SMCI", "PSTG"],
    # 饮料
    "Beverages—Non-Alcoholic": ["KO", "PEP", "MNST", "KDP", "FIZZ"],
    # 航空航天
    "Aerospace & Defense": ["BA", "LMT", "RTX", "NOC", "GD"],
    # 游戏
    "Electronic Gaming & Multimedia": ["EA", "TTWO", "U", "RBLX", "PLTK"],
}


def _get_fallback_peers(ticker: str, industry: str = "") -> list[str]:
    """获取可比公司列表：行业映射优先 → ticker 兜底。

    Args:
        ticker: 目标股票代码
        industry: FMP profile 返回的 industry 字段（如 "Consumer Electronics"）

    Returns:
        同行 ticker 列表（已过滤自身）
    """
    # 优先：按行业匹配（覆盖 20 个行业）
    if industry:
        peers = _INDUSTRY_PEERS.get(industry)
        if peers:
            return [p for p in peers if p.upper() != ticker.upper()]

    # 兜底：尝试 ticker 本身所属的行业
    ticker_upper = ticker.upper()
    for ind_peers in _INDUSTRY_PEERS.values():
        if ticker_upper in ind_peers:
            return [p for p in ind_peers if p.upper() != ticker_upper]

    # 最终兜底：返回空
    return []
