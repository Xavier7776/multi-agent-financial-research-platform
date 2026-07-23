"""Phase 4 评测执行器 — 跑 reviewer + judge，输出 results JSONL。

用法：
    python -m multi_agents.evals.run_eval --results multi_agents/evals/results_phase4.jsonl --concurrency 5
    python multi_agents/evals/run_eval.py --results multi_agents/evals/results_phase4.jsonl --limit 5  # smoke test

输出 JSONL 格式：
    {"section_id":"AAPL_sec0","in_sample":true,"dimension":"data_accuracy","mutation_type":"numeric_large_bias_2x","ground_truth":"...","reviewer_output":"...","judge":"CAUGHT","mode":"fin_on"}
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# 让脚本可独立运行
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from multi_agents.agents.reviewer import ReviewerAgent
from multi_agents.agents.utils.llms import call_model
from multi_agents.evals.fixtures import build_sections
from multi_agents.evals.mutations import generate_mutations
from multi_agents.evals.judge import judge_single, judge_clean_sample


async def run_one_review(draft_content: str, fc: dict, mode: str, model: str,
                         timeout: float = 300.0) -> str | None:
    """跑一次 reviewer，返回 revision_notes (None = PASS)。

    Args:
        draft_content: 草稿内容（可能注入了 mutation）
        fc: financial_context (mode=fin_on 时用)
        mode: "fin_on" 用四维校验；"fin_off" 用 baseline review_draft
        model: LLM 模型名
        timeout: 单次 review 超时（秒）
    """
    reviewer = ReviewerAgent()
    task = {
        "model": model,
        "guidelines": ["Write a professional financial research report."],
        "verbose": False,
    }
    draft_state = {
        "task": task,
        "draft": draft_content,
        "revision_notes": None,
        "financial_context": fc if mode == "fin_on" else {},
    }
    try:
        result = await asyncio.wait_for(
            reviewer.review_draft(draft_state),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        return "__TIMEOUT__"
    except Exception as e:
        return f"__ERROR__: {e}"


async def evaluate_one(section, mutation, mode: str, model: str,
                       call_model_fn, judge_model: str = "mimo") -> dict:
    """跑一条评测：review → judge。"""
    # 1. Reviewer
    reviewer_output = await run_one_review(
        draft_content=mutation.mutated_content,
        fc=section.financial_context,
        mode=mode,
        model=model,
    )

    # 2. Judge (只对非超时/非错误的样本)
    if reviewer_output in ("__TIMEOUT__", None) or \
       (reviewer_output or "").startswith("__ERROR__"):
        judge_result = "MISSED" if reviewer_output != "__TIMEOUT__" else "TIMEOUT"
        if (reviewer_output or "").startswith("__ERROR__"):
            judge_result = "ERROR"
    else:
        judge_result = await judge_single(
            ground_truth_desc=mutation.ground_truth_desc,
            reviewer_output=reviewer_output,
            call_model_fn=call_model_fn,
            judge_model=judge_model,
        )

    return {
        "section_id": section.section_id,
        "ticker": section.ticker,
        "in_sample": section.in_sample,
        "dimension": mutation.dimension,
        "mutation_type": mutation.mutation_type,
        "ground_truth": mutation.ground_truth_desc,
        "mode": mode,
        "reviewer_output": reviewer_output,
        "judge": judge_result,
        "timestamp": time.time(),
    }


async def evaluate_clean(section, mode: str, model: str) -> dict:
    """跑一条干净样本评测（无 mutation，测 FP）。"""
    reviewer_output = await run_one_review(
        draft_content=section.content,
        fc=section.financial_context,
        mode=mode,
        model=model,
    )
    is_fp = judge_clean_sample(reviewer_output)
    return {
        "section_id": section.section_id,
        "ticker": section.ticker,
        "in_sample": section.in_sample,
        "dimension": "clean",
        "mutation_type": "none",
        "ground_truth": "无注入错误，reviewer 应返回 None",
        "mode": mode,
        "reviewer_output": reviewer_output,
        "judge": "FP" if is_fp else "TN",
        "timestamp": time.time(),
    }


async def run_eval(results_path: str, model: str, modes: list[str],
                   concurrency: int = 5, limit: int | None = None,
                   timeout: float = 300.0, judge_model: str = "mimo"):
    """主入口：构建所有评测任务 → 并发跑 → 写 JSONL。"""
    print(f"[Phase 4] 构建 sections...")
    sections = build_sections()
    print(f"  sections: {len(sections)} (in-sample: "
          f"{sum(1 for s in sections if s.in_sample)}, "
          f"held-out: {sum(1 for s in sections if not s.in_sample)})")

    print(f"[Phase 4] 构建 mutations...")
    all_tasks = []  # list of coroutines
    for s in sections:
        mutations = generate_mutations(s)
        for mode in modes:
            # 注入样本
            for m in mutations:
                all_tasks.append(evaluate_one(s, m, mode, model, call_model,
                                              judge_model=judge_model))
            # 干净样本（每段一条）
            all_tasks.append(evaluate_clean(s, mode, model))

    if limit:
        all_tasks = all_tasks[:limit]
    print(f"[Phase 4] total tasks: {len(all_tasks)}")

    # 并发执行
    sem = asyncio.Semaphore(concurrency)
    async def _run_with_sem(coro):
        async with sem:
            return await coro

    results = []
    start = time.time()
    completed = 0
    with open(results_path, "w", encoding="utf-8") as f:
        for coro in asyncio.as_completed([_run_with_sem(t) for t in all_tasks]):
            try:
                r = await coro
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                results.append(r)
                completed += 1
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(all_tasks) - completed) / rate if rate > 0 else 0
                judge = r.get("judge", "?")
                sid = r.get("section_id", "?")
                dim = r.get("dimension", "?")
                mode = r.get("mode", "?")
                msg = (f"  [{completed}/{len(all_tasks)}] {sid} {dim} {mode} = {judge}"
                       f"  ({elapsed:.0f}s, ETA {eta:.0f}s)")
                print(msg, flush=True)
            except Exception as e:
                print(f"  [ERROR] {e}", flush=True)

    print(f"\n[Phase 4] 完成: {len(results)} 条结果")
    print(f"  耗时: {time.time() - start:.1f}s")
    print(f"  写入: {results_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="multi_agents/evals/results_phase4.jsonl",
                        help="结果 JSONL 输出路径")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="reviewer 和 judge 用的 LLM 模型")
    parser.add_argument("--modes", nargs="+", default=["fin_on"],
                        choices=["fin_on", "fin_off"],
                        help="评测模式（fin_on=四维校验, fin_off=baseline）")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="并发数")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个任务（smoke test）")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="单次 review 超时秒数")
    parser.add_argument("--judge_model", default="mimo",
                        help="judge 用的 LLM 模型名（默认 mimo）")
    args = parser.parse_args()

    asyncio.run(run_eval(
        results_path=args.results,
        model=args.model,
        modes=args.modes,
        concurrency=args.concurrency,
        limit=args.limit,
        timeout=args.timeout,
        judge_model=args.judge_model,
    ))


if __name__ == "__main__":
    main()
