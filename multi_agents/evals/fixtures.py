"""Phase 4 评测样本构建 — 从真实研报自动切分段落 + 提取 ground truth 真值。

样本设计：
    6 个 ticker (4 in-sample + 2 held-out)
    × 每个 ticker 切 4 段 (按 markdown ## 标题切分)
    × 每段多种 mutation
    = 70+ 注入样本 + 12 干净样本

Held-out 设计：
    In-sample (参与 Phase 3): 京东方 000725, AAPL, 比亚迪 002594, 恒瑞医药 600276
    Held-out (Phase 4 新加入): 茅台 600519, 招商银行 600036
    对比 in-sample vs held-out 的 recall，验证不是过拟合。
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

# 6 个 ticker 配置
TICKERS = [
    # (run_dir_name, ticker, in_sample)
    ("run_1782359024_京东方(000725)全面分析", "000725", True),
    ("run_1784263242_全面分析一下苹果(AAPL)的财务与投资价值", "AAPL", True),
    ("run_1784631970_全面分析一下002594 比亚迪", "002594", True),
    ("run_1784636317_全面分析一下600276 恒瑞医药", "600276", True),
    # Held-out: 不参与 Phase 3 评测
    ("run_1784629416_全面分析一下600519 贵州茅台", "600519", False),
    ("run_1784634179_全面分析一下600036 招商银行", "600036", False),
]

# 每个 ticker 取 4 段，每段 500-1500 字符
SECTIONS_PER_TICKER = 4
MIN_SECTION_LEN = 400
MAX_SECTION_LEN = 1500


@dataclass
class Section:
    """一段真实研报内容 + 其对应真值。"""
    ticker: str
    section_id: str          # e.g. "000725_sec0"
    content: str             # 原始段落文本
    financial_context: dict  # 该 ticker 的真值
    in_sample: bool          # 是否在 in-sample 集合


def _split_into_sections(md_text: str) -> list[str]:
    """按 ## / ### 标题切分 markdown，返回长度适中的段落列表。"""
    # 用 ## 或 ### 作为切分点
    parts = re.split(r"(?=^#{2,3}\s+\S)", md_text, flags=re.MULTILINE)
    sections = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 太长的段落再切分（按段落换行）
        if len(p) > MAX_SECTION_LEN:
            chunks = []
            buf = ""
            for para in p.split("\n\n"):
                if len(buf) + len(para) > MAX_SECTION_LEN and buf:
                    chunks.append(buf.strip())
                    buf = para
                else:
                    buf = buf + "\n\n" + para if buf else para
            if buf:
                chunks.append(buf.strip())
            for c in chunks:
                if MIN_SECTION_LEN <= len(c) <= MAX_SECTION_LEN * 2:
                    sections.append(c)
        elif len(p) >= MIN_SECTION_LEN:
            sections.append(p)
    return sections


def _load_fc(run_dir: Path, ticker: str) -> Optional[dict]:
    """从 financial_data.json 加载 financial_context。"""
    json_files = list(run_dir.glob("*.json"))
    if not json_files:
        return None
    with open(json_files[0], encoding="utf-8") as f:
        data = json.load(f)
    ov = data.get("overview", {}) or {}
    peers = data.get("industry_peers", []) or data.get("peers", []) or []
    return {
        "ticker": ticker,
        "company_name": ov.get("name", ticker),
        "pe_ratio": ov.get("pe_ratio"),
        "pb_ratio": ov.get("pb_ratio"),
        "roe": ov.get("roe"),
        "profit_margin": ov.get("profit_margin"),
        "revenue_growth": ov.get("revenue_growth"),
        "market_cap": ov.get("market_cap"),
        "sector": ov.get("sector", ""),
        "industry": ov.get("industry", ""),
        "peers": peers,
    }


def build_sections() -> list[Section]:
    """构建所有 ticker 的段落样本。"""
    sections = []
    for run_name, ticker, in_sample in TICKERS:
        run_dir = OUTPUTS_DIR / run_name
        if not run_dir.exists():
            print(f"[SKIP] run dir not found: {run_dir}")
            continue
        md_files = list(run_dir.glob("*.md"))
        if not md_files:
            print(f"[SKIP] no .md in {run_dir}")
            continue
        md_text = md_files[0].read_text(encoding="utf-8")
        fc = _load_fc(run_dir, ticker)
        if not fc:
            print(f"[SKIP] no fc for {ticker}")
            continue

        parts = _split_into_sections(md_text)
        # 取前 N 段（确保不同 ticker 段落多样）
        selected = parts[:SECTIONS_PER_TICKER]
        # 段落不够则复制（保证样本数）
        while len(selected) < SECTIONS_PER_TICKER and parts:
            selected.append(parts[len(selected) % len(parts)])

        for i, content in enumerate(selected):
            sections.append(Section(
                ticker=ticker,
                section_id=f"{ticker}_sec{i}",
                content=content,
                financial_context=fc,
                in_sample=in_sample,
            ))
        print(f"[OK] {ticker}: {len(selected)} sections (in_sample={in_sample})")
    return sections


if __name__ == "__main__":
    sections = build_sections()
    print(f"\nTotal sections: {len(sections)}")
    in_sample_count = sum(1 for s in sections if s.in_sample)
    held_out_count = sum(1 for s in sections if not s.in_sample)
    print(f"  in-sample: {in_sample_count}")
    print(f"  held-out:  {held_out_count}")
    print(f"\nSample lengths:")
    for s in sections:
        print(f"  {s.section_id} ({'IN' if s.in_sample else 'OUT'}): {len(s.content)} chars")
