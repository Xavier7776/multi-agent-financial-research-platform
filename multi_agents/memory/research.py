from typing import TypedDict, List, Annotated, Optional
import operator


class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: List[str]
    research_data: List[dict]
    human_feedback: str
    plan_revision_count: int
    # Report layout
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: List[str]
    report: str
    # --- 金融研报扩展字段 (Phase 2) ---
    financial_data: dict            # Yahoo Finance 获取的结构化金融数据
                                    # {ticker, stock_price, market_cap, key_ratios, financial_statements}
    industry_peers: List[dict]      # 同行业可比公司数据
                                    # [{ticker, name, market_cap, pe, pb, roe, revenue_growth}, ...]
    valuation_metrics: dict         # 估值指标汇总
                                    # {pe, pb, roe, dividend_yield, peg, ev_ebitda, sector_avg_pe, sector_avg_pb}
    risk_flags: List[str]           # 风险提示标记
                                    # ["营收下滑", "高负债率", "行业政策风险", ...]
    report_type: str               # 报告类型标识
                                    # "financial_research_report" | "general_research_report"

