"""Phase 4 judge — 用 LLM 做语义级匹配，避免字面命中要求。

判定流程：
1. 拿到 mutation.ground_truth_desc（"应该抓到什么"）和 reviewer 输出（revision_notes）
2. 让 LLM 判断 reviewer 的输出是否抓到了 ground_truth_desc 描述的问题
3. 返回 "CAUGHT" / "MISSED" / "PARTIAL"

关键设计：
- 独立 LLM 调用，与 reviewer 用同一模型但不共享上下文
- prompt 明确给出 ground truth 和 reviewer 输出，避免主观判断
- 对干净样本：reviewer 返回 None = PASS（不误报）；返回非 None = FP
"""

from typing import Optional


JUDGE_PROMPT_TEMPLATE = """你是一个公正的金融报告审查评测员。

任务：判断 reviewer 的输出是否成功抓到了 ground truth 描述的问题。

判断规则：
- CAUGHT: reviewer 的输出明确指出了 ground truth 描述的问题（不要求字面命中，语义匹配即可）
- PARTIAL: reviewer 部分抓到（如指出了同类问题但描述不准确，或抓到了相关但不完全对应的点）
- MISSED: reviewer 完全没抓到 ground truth 描述的问题

ground truth:
{ground_truth}

reviewer 输出:
{reviewer_output}

只返回一个单词：CAUGHT 或 PARTIAL 或 MISSED"""


async def judge_single(ground_truth_desc: str, reviewer_output: Optional[str],
                       call_model_fn, judge_model: str = "mimo") -> str:
    """对单条评测调用 LLM judge。

    Args:
        ground_truth_desc: mutation 的 ground_truth_desc 字段
        reviewer_output: reviewer 的 revision_notes（None 表示 PASS）
        call_model_fn: async callable(prompt, model) -> str
        judge_model: judge 用的 LLM 模型名

    Returns:
        "CAUGHT" / "PARTIAL" / "MISSED"
    """
    if reviewer_output is None or not reviewer_output.strip():
        # reviewer 没报任何问题 → 注入的错误没被抓到
        return "MISSED"

    prompt = [
        {"role": "system", "content": "你是一个公正的金融报告审查评测员。"},
        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(
            ground_truth=ground_truth_desc,
            reviewer_output=reviewer_output,
        )},
    ]
    try:
        response = await call_model_fn(prompt, model=judge_model)
        if not response:
            return "MISSED"
        # 提取返回的第一个关键词
        upper = response.upper().strip()
        for keyword in ("CAUGHT", "PARTIAL", "MISSED"):
            if keyword in upper:
                return keyword
        return "MISSED"
    except Exception as e:
        # judge LLM 调用失败时，用规则兜底：检查 reviewer 输出是否包含 ground_truth 中的关键词
        return _rule_based_judge(ground_truth_desc, reviewer_output)


def _rule_based_judge(ground_truth_desc: str, reviewer_output: str) -> str:
    """judge LLM 失败时的规则兜底。"""
    import re
    # 提取 ground_truth 中的数字（如 PE 偏差值）
    gt_numbers = re.findall(r"\d+\.?\d*", ground_truth_desc)
    # 提取 ground_truth 中的关键短语
    gt_keywords = []
    for phrase in ["保证上涨", "零风险", "下行风险", "ROE 行业第一", "绝对优势", "强烈建议",
                   "缺失", "删除", "章节标题", "数据来源超链接"]:
        if phrase in ground_truth_desc:
            gt_keywords.append(phrase)

    output_lower = reviewer_output.lower()
    matched = 0
    for n in gt_numbers:
        if n in reviewer_output:
            matched += 1
    for kw in gt_keywords:
        if kw in reviewer_output:
            matched += 1

    total = len(gt_numbers) + len(gt_keywords)
    if total == 0:
        return "PARTIAL"  # 没有可提取的关键词，保守判 PARTIAL
    if matched >= max(1, total // 2):
        return "CAUGHT"
    if matched > 0:
        return "PARTIAL"
    return "MISSED"


def judge_clean_sample(reviewer_output: Optional[str]) -> bool:
    """对干净样本的判定（不需要 LLM）。

    Returns:
        True = 误报 (FP), False = 正确通过 (TN)
    """
    # 干净样本：reviewer 返回 None 或空 = 正确通过
    # 返回非空 = 误报
    if reviewer_output is None or not reviewer_output.strip():
        return False  # TN, 不是 FP
    return True  # FP
