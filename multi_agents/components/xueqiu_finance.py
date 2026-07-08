"""
雪球 (pysnowball) A 股数据获取模块
---------------------------------
通过 pysnowball 库调用雪球 API，获取 A 股实时行情、PE/PB/ROE 和财务报表。
接口与 FinancialDataTool 完全对齐。

使用方式:
    tool = XueqiuDataTool("600519")
    overview = await tool.get_stock_overview()
    statements = await tool.get_financial_statements()
    peers = await tool.get_industry_peers()
"""

import asyncio
import functools
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# 雪球 token（优先读环境变量，兜底硬编码）
_XQ_TOKEN = os.getenv("XUEQIU_TOKEN")


def _init_token():
    """确保 pysnowball token 已设置。"""
    import pysnowball as ball
    ball.set_token(_XQ_TOKEN)


def _ticker_to_xq(ticker: str) -> str:
    """A股: 600519→SH600519, 港股: 00700→00700"""
    code = ticker.strip()
    if not code.isdigit():
        return code
    if code.startswith(("6", "688")) and len(code) == 6:
        return f"SH{code}"
    if len(code) == 6 and code[0] in ("0", "3"):
        return f"SZ{code}"
    if len(code) == 5:
        return code            # 港股直接用5位数字，不加前缀
    return code


class XueqiuDataTool:
    """A 股金融数据获取工具（雪球 pysnowball 后端）。"""

    _token_set = False

    def __init__(self, ticker: str):
        self.ticker = ticker.strip()
        self.xq_symbol = _ticker_to_xq(self.ticker)
        self._cache: dict = {}
        self._api_delay = 0.2  # 雪球 API 调用间隔（秒），防止限流

    @classmethod
    def _ensure_token(cls):
        if not cls._token_set:
            _init_token()
            cls._token_set = True

    @staticmethod
    def _safe(v):
        """安全转 float。处理雪球 [value, rate] 数组格式。"""
        if v is None:
            return None
        try:
            import math
            # 雪球 [value, change_rate] 数组
            if isinstance(v, (list, tuple)) and len(v) > 0:
                v = v[0]
            f = float(v)
            return round(f, 2) if not math.isnan(f) and math.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    #None为时候默认的线程池
    def _run(self, func, *args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    # ------------------------------------------------------------------
    # 1. 股票基本面概览（quote_detail + main_indicator）
    # ------------------------------------------------------------------

    async def get_stock_overview(self) -> dict:
        cache_key = "stock_overview"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import pysnowball as ball

            self._ensure_token()
            quote_data = await self._run(ball.quote_detail, self.xq_symbol)
            await asyncio.sleep(self._api_delay)
            main_data = await self._run(ball.main_indicator, self.xq_symbol)
            await asyncio.sleep(self._api_delay)
            industry_data = await self._run(ball.industry, self.xq_symbol)
            await asyncio.sleep(self._api_delay)
            biz_data = await self._run(ball.business, self.xq_symbol)

            q = quote_data.get("data", {}).get("quote", {}) if isinstance(quote_data, dict) else {}
            if not q:
                logger.warning(f"[XueqiuData] quote_detail({self.xq_symbol}) 返回空，可能不支持的股票代码")
                return {}  # fall through to generic mode
            items = main_data.get("data", {}).get("items", []) if isinstance(main_data, dict) else []
            metrics = items[0] if items else {}

            # 行业分类 (industry 返回嵌套结构)
            ind_dict = industry_data.get("data", {}) if isinstance(industry_data, dict) else {}
            ind_items = ind_dict.get("industry", [])
            industry = ind_items[0].get("ind_name", "") if isinstance(ind_items, list) and ind_items else ""
            biz_dict = biz_data.get("data", {}) if isinstance(biz_data, dict) else {}
            sector_val = biz_dict.get("industry", "")
            sector = sector_val if isinstance(sector_val, str) else ""

            # 公司简介
            summary = biz_dict.get("main_operation_business", "") or ""

            result = {
                "ticker": self.ticker,
                "name": q.get("name", self.ticker),
                "price": self._safe(q.get("current")),
                "market_cap": self._safe(q.get("market_capital")),
                "pe_ratio": self._safe(q.get("pe_ttm")),
                "pb_ratio": self._safe(q.get("pb")),
                "roe": self._safe(metrics.get("avg_roe")),
                "revenue_growth": None,
                "debt_to_equity": self._safe(metrics.get("asset_liab_ratio")),
                "profit_margin": self._safe(metrics.get("net_selling_rate")),
                "dividend_yield": self._safe(q.get("dividend_yield")),
                "beta": self._safe(q.get("beta")),
                "sector": sector,
                "industry": industry,
                "summary": summary,
                "currency": "CNY",
                "exchange": "上海" if "SH" in self.xq_symbol else "深圳",
                "fifty_two_week_high": self._safe(q.get("high52w")),
                "fifty_two_week_low": self._safe(q.get("low52w")),
            }

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"[XueqiuData] get_stock_overview({self.ticker}) 失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # 2. 财务报表（income/balance/cash_flow + indicator）
    # ------------------------------------------------------------------

    async def get_financial_statements(self) -> dict:
        cache_key = "financial_statements"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import pysnowball as ball

            self._ensure_token()
            bs_data = await self._run(ball.balance_v2, symbol=self.xq_symbol, type='all', is_detail=True)
            await asyncio.sleep(self._api_delay)
            indicator_data = await self._run(ball.income, symbol=self.xq_symbol, is_annals=1, count=5)
            await asyncio.sleep(self._api_delay)
            cf_data = await self._run(ball.cash_flow, self.xq_symbol)

            # 利润表 (income) — 年度数据
            items = indicator_data.get("data", {}).get("list", []) if isinstance(indicator_data, dict) else []
            q_rev, q_earn = [], []
            for row in items[:4]:
                q_rev.append({
                    "date": str(row.get("report_name", "")),
                    "revenue": self._safe(row.get("total_revenue")),
                })
                q_earn.append({
                    "date": str(row.get("report_name", "")),
                    "earnings": self._safe(row.get("net_profit")),
                })

            # 年度营收和净利润（season=1 income 直接返回年度值）
            a_rev, a_earn = [], []
            revenue_growth = None
            for row in items[:5]:
                rev = self._safe(row.get("total_revenue"))
                earn = self._safe(row.get("net_profit"))
                if rev is not None:
                    a_rev.append({"date": str(row.get("report_name", "")), "revenue": rev})
                if earn is not None:
                    a_earn.append({"date": str(row.get("report_name", "")), "earnings": earn})
            if len(a_rev) >= 2:
                this_year, last_year = a_rev[0]["revenue"], a_rev[1]["revenue"]
                if last_year and last_year > 0:
                    revenue_growth = round((this_year - last_year) / last_year * 100, 2)

            # 资产负债表 (balance_v2) — 雪球返回 [value, change_rate] 数组
            bs_list = bs_data.get("data", {}).get("list", []) if isinstance(bs_data, dict) else []
            balance_sheet = {}
            if bs_list:
                bs = bs_list[0]
                total_assets = self._safe(bs.get("total_assets"))  # [val, rate]
                total_liab = self._safe(bs.get("total_liab"))
                # 净资产 = 总资产 - 总负债
                total_equity = round(total_assets - total_liab, 2) if total_assets and total_liab else None
                balance_sheet = {
                    "total_assets": total_assets,
                    "total_debt": total_liab,
                    "total_equity": total_equity,
                    "current_assets": self._safe(bs.get("total_current_assets")),
                    "current_liabilities": self._safe(bs.get("total_current_liab")),
                }

            # 现金流表
            cf_list = cf_data.get("data", {}).get("list", []) if isinstance(cf_data, dict) else []
            cash_flow = {}
            if cf_list:
                cf = cf_list[0]
                cash_flow = {
                    "operating_cash_flow": self._safe(cf.get("ncf_from_oa")),
                    "investing_cash_flow": self._safe(cf.get("ncf_from_ia")),
                    "financing_cash_flow": self._safe(cf.get("ncf_from_fa")),
                    "free_cash_flow": None,
                }

            result = {
                "quarterly_revenue": q_rev,
                "quarterly_earnings": q_earn,
                "annual_revenue": a_rev,
                "annual_earnings": a_earn,
                "cash_flow": cash_flow,
                "balance_sheet": balance_sheet,
            }
            self._cache[cache_key] = result

            # 营收增速共享到 overview 缓存，避免重复计算
            if revenue_growth is not None and "stock_overview" in self._cache:
                self._cache["stock_overview"]["revenue_growth"] = revenue_growth

            return result

        except Exception as e:
            logger.warning(f"[XueqiuData] get_financial_statements({self.ticker}) 失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # 3. 同行业可比公司
    # ------------------------------------------------------------------

    async def get_industry_peers(self) -> list[dict]:
        """雪球无同行列表接口。同行对比由 Editor 通过 Web 搜索自行获取。"""
        return []
