from typing import TypedDict, List, Annotated, Optional
import operator


class DraftState(TypedDict):
    task: dict
    topic: str
    draft: dict
    review: str
    revision_notes: str
    # --- 金融研报扩展字段 (Phase 2) ---
    financial_context: dict         # 当前章节相关的金融数据上下文
                                    # {section_topic, related_metrics, peer_comparison, source_urls}
    accuracy_checks: List[dict]     # 金融数据准确性校验结果
                                    # [{section, metric, draft_value, actual_value, pass, note}, ...]
    data_sources: List[str]         # 数据来源追溯
                                    # ["Yahoo Finance: AAPL 2024Q3 10-Q", "Reuters: Industry Report", ...]