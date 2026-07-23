"""Phase 4 评测的 mutation 生成器 — 在真实段落上注入可定位的错误。

设计要点：
- 4 个维度 (data/logic/compliance/format) × 多种 mutation 类型
- 每条 mutation 携带 (dimension, type, ground_truth_desc) 三元组
- ground_truth_desc 是 judge 用于语义匹配的"答案"
- mutation 必须是确定性的（同一输入产生同一错误），便于复现
"""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class Mutation:
    """一条注入的错误。"""
    dimension: str           # "data_accuracy" / "logic" / "compliance" / "format"
    mutation_type: str       # 该维度下的具体错误类型
    ground_truth_desc: str   # judge 用于判定的语义描述（"应该抓到什么"）
    mutated_content: str     # 注入错误后的段落
    inject_succeeded: bool   # 是否成功注入（有些段落没数值，注入失败）


# ============================================================================
# 数据准确性 mutations
# ============================================================================

def _find_number_near_alias(content: str, aliases: list[str]) -> Optional[tuple[str, float, str, int, int]]:
    """在 content 中找到 alias 附近的数字，返回 (alias, 数值, 数值字符串, 起始位置, 结束位置)。"""
    import re
    for alias in aliases:
        pattern = re.escape(alias) + r"[^\d]{0,20}(\d+\.\d+|\d+)"
        m = re.search(pattern, content)
        if m:
            n_str = m.group(1)
            try:
                n = float(n_str)
                return (alias, n, n_str, m.start(1), m.end(1))
            except ValueError:
                continue
    return None


def mutate_data_small_bias(content: str, fc: dict) -> Optional[Mutation]:
    """数据准确性 - 小偏差（~1%）：替换 PE/PB/ROE 附近的数字为略偏的值。"""
    for field, aliases in [
        ("pe_ratio", ["PE", "市盈率", "P/E"]),
        ("pb_ratio", ["PB", "市净率", "P/B"]),
        ("roe", ["ROE", "净资产收益率"]),
    ]:
        true_val = fc.get(field)
        if not isinstance(true_val, (int, float)) or true_val is None or abs(true_val) < 1e-6:
            continue
        found = _find_number_near_alias(content, aliases)
        if not found:
            continue
        alias, n, n_str, start, end = found
        if abs(n - true_val) > 1e-6:
            continue  # 已经是错的，跳过
        # 注入 ~1% 偏差
        new_val = true_val * 0.99 if true_val > 0 else true_val * 1.01
        new_val = round(new_val, 2)
        new_str = f"{new_val}"
        mutated = content[:start] + new_str + content[end:]
        return Mutation(
            dimension="data_accuracy",
            mutation_type="numeric_small_bias_1pct",
            ground_truth_desc=f"草稿在 '{alias}' 附近引用的数值 {new_str} 与参考数据 {field}={true_val} 不符（约 1% 偏差）",
            mutated_content=mutated,
            inject_succeeded=True,
        )
    return None


def mutate_data_large_bias(content: str, fc: dict) -> Optional[Mutation]:
    """数据准确性 - 大偏差（×1.5~×2.0）：把数值翻倍或减半。"""
    for field, aliases in [
        ("pe_ratio", ["PE", "市盈率", "P/E"]),
        ("pb_ratio", ["PB", "市净率", "P/B"]),
        ("roe", ["ROE", "净资产收益率"]),
        ("profit_margin", ["净利率", "净利润率"]),
    ]:
        true_val = fc.get(field)
        if not isinstance(true_val, (int, float)) or true_val is None or abs(true_val) < 1e-6:
            continue
        found = _find_number_near_alias(content, aliases)
        if not found:
            continue
        alias, n, n_str, start, end = found
        if abs(n - true_val) > 1e-6:
            continue
        # 注入 ×1.5 或 ×2.0 偏差
        multiplier = 2.0
        new_val = round(true_val * multiplier, 2)
        new_str = f"{new_val}"
        mutated = content[:start] + new_str + content[end:]
        return Mutation(
            dimension="data_accuracy",
            mutation_type="numeric_large_bias_2x",
            ground_truth_desc=f"草稿在 '{alias}' 附近引用的数值 {new_str} 与参考数据 {field}={true_val} 严重不符（×2 偏差）",
            mutated_content=mutated,
            inject_succeeded=True,
        )
    return None


# ============================================================================
# 逻辑完整性 mutations
# ============================================================================

def mutate_logic_unsupported_claim(content: str, fc: dict) -> Optional[Mutation]:
    """逻辑完整性 - 注入与参考数据矛盾的结论（如 ROE 行业第一但 ROE 实际很低）。"""
    roe = fc.get("roe")
    if not isinstance(roe, (int, float)) or roe is None:
        return None
    # ROE 低于 5% 的公司，注入"行业第一"是逻辑错误
    if roe < 5.0:
        claim = "公司净资产收益率行业第一，盈利能力远超同业。"
        injected = content + "\n\n" + claim
        return Mutation(
            dimension="logic",
            mutation_type="unsupported_industry_leader_claim",
            ground_truth_desc=f"草稿声称 ROE 行业第一，但参考数据 ROE={roe}%（数值偏低），结论缺乏数据支撑",
            mutated_content=injected,
            inject_succeeded=True,
        )
    # ROE 高的公司，注入"风险极低"也是逻辑错误
    elif roe > 15.0:
        claim = "公司盈利能力已无提升空间，未来下行风险极低，股价上涨确定性高。"
        injected = content + "\n\n" + claim
        return Mutation(
            dimension="logic",
            mutation_type="unsupported_low_risk_claim",
            ground_truth_desc=f"草稿声称'下行风险极低'+'上涨确定性高'，与参考数据 ROE={roe}% 的高基数不符，缺乏数据支撑",
            mutated_content=injected,
            inject_succeeded=True,
        )
    return None


# ============================================================================
# 合规性 mutations
# ============================================================================

COMPLIANCE_VIOLATIONS = [
    ("guaranteed_return", "本股票保证上涨，未来 12 个月年化收益率不低于 30%。"),
    ("risk_free", "当前估值下行风险几乎为零，是罕见的零风险投资机会。"),
    ("absolute_leader", "公司绝对垄断行业，毫无竞争对手，可闭眼买入。"),
    ("missing_risk_disclosure", "强烈建议买入，目标价翻倍，无需考虑任何风险因素。"),
]


def mutate_compliance(content: str, fc: dict) -> Optional[Mutation]:
    """合规性 - 注入保证上涨/零风险等违规表述。"""
    violation_type, phrase = random.choice(COMPLIANCE_VIOLATIONS)
    # 插到段落末尾
    injected = content.rstrip() + "\n\n" + phrase + "\n"
    return Mutation(
        dimension="compliance",
        mutation_type=violation_type,
        ground_truth_desc=f"草稿包含违规表述：'{phrase}'（属于 {violation_type} 类型，违反金融研报合规要求）",
        mutated_content=injected,
        inject_succeeded=True,
    )


# ============================================================================
# 格式规范 mutations
# ============================================================================

def mutate_format_remove_headers(content: str, fc: dict) -> Optional[Mutation]:
    """格式规范 - 删除所有 markdown 标题。"""
    import re
    # 删除 ## 和 ### 开头的行
    lines = content.split("\n")
    new_lines = []
    removed_count = 0
    for line in lines:
        if re.match(r"^#{2,3}\s+\S", line):
            removed_count += 1
            continue
        new_lines.append(line)
    if removed_count == 0:
        return None
    mutated = "\n".join(new_lines)
    return Mutation(
        dimension="format",
        mutation_type="missing_section_headers",
        ground_truth_desc=f"草稿删除了 {removed_count} 个 markdown 章节标题（## / ###），专业研报应有清晰的小节结构",
        mutated_content=mutated,
        inject_succeeded=True,
    )


def mutate_format_remove_links(content: str, fc: dict) -> Optional[Mutation]:
    """格式规范 - 删除所有 markdown 超链接。"""
    import re
    # [text](url) → text
    matches = re.findall(r"\[([^\]]+)\]\(https?://[^)]+\)", content)
    if not matches:
        return None
    mutated = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", content)
    return Mutation(
        dimension="format",
        mutation_type="removed_source_links",
        ground_truth_desc=f"草稿删除了 {len(matches)} 个 markdown 数据来源超链接 [text](url)，专业研报应标注关键数据来源",
        mutated_content=mutated,
        inject_succeeded=True,
    )


# ============================================================================
# 编排：对每段生成所有 mutation
# ============================================================================

MUTATORS = [
    ("data_small_bias", mutate_data_small_bias),
    ("data_large_bias", mutate_data_large_bias),
    ("logic_unsupported", mutate_logic_unsupported_claim),
    ("compliance", mutate_compliance),
    ("format_remove_headers", mutate_format_remove_headers),
    ("format_remove_links", mutate_format_remove_links),
]


def generate_mutations(section, seed: int = 42) -> list[Mutation]:
    """对一段内容生成所有 mutation。返回 list（部分 mutation 可能因段落不适用而返回 None）。"""
    random.seed(seed + hash(section.section_id) % 10000)
    mutations = []
    for name, fn in MUTATORS:
        try:
            m = fn(section.content, section.financial_context)
            if m is not None:
                mutations.append(m)
        except Exception:
            continue
    return mutations


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from multi_agents.evals.fixtures import build_sections
    sections = build_sections()
    total = 0
    by_dim = {"data_accuracy": 0, "logic": 0, "compliance": 0, "format": 0}
    by_split = {"in_sample": 0, "held_out": 0}
    for s in sections:
        ms = generate_mutations(s)
        total += len(ms)
        for m in ms:
            by_dim[m.dimension] += 1
            if s.in_sample:
                by_split["in_sample"] += 1
            else:
                by_split["held_out"] += 1
        print(f"{s.section_id}: {len(ms)} mutations")
    print(f"\nTotal mutations: {total}")
    print(f"By dimension: {by_dim}")
    print(f"By split: {by_split}")
    # 加上干净样本数
    print(f"\nFinal sample = {total} injected + {len(sections)} clean = {total + len(sections)}")
