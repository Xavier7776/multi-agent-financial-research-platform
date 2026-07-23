import asyncio
import re
from .utils.views import print_agent_output
from .utils.llms import call_model

TEMPLATE = """You are an expert research article reviewer. \
Your goal is to review research drafts and provide feedback to the reviser only based on specific guidelines. \
"""


class ReviewerAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def review_draft(self, draft_state: dict):
        """
        Review a draft article
        :param draft_state:
        :return:
        """
        task = draft_state.get("task")
        guidelines = "- ".join(guideline for guideline in task.get("guidelines"))
        revision_notes = draft_state.get("revision_notes")

        # --- Phase 2: Financial review ---
        financial_context = draft_state.get("financial_context") or {}
        is_financial = bool(financial_context)

        if is_financial:
            return await self._review_financial_draft(
                draft_state, task, financial_context, revision_notes
            )
        # ---------------------------------

        revise_prompt = f"""The reviser has already revised the draft based on your previous review notes with the following feedback:
{revision_notes}\n
Please provide additional feedback ONLY if critical since the reviser has already made changes based on your previous feedback.
If you think the article is sufficient or that non critical revisions are required, please aim to return None.
"""

        review_prompt = f"""You have been tasked with reviewing the draft which was written by a non-expert based on specific guidelines.
Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the revision.
If not all of the guideline criteria are met, you should send appropriate revision notes.
If the draft meets all the guidelines, please return None.
{revise_prompt if revision_notes else ""}

Guidelines: {guidelines}\nDraft: {draft_state.get("draft")}\n
"""
        prompt = [
            {"role": "system", "content": TEMPLATE},
            {"role": "user", "content": review_prompt},
        ]

        response = await call_model(prompt, model=task.get("model"))

        if task.get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs", "review_feedback",
                    f"Review feedback is: {response}...",
                    self.websocket,
                )
            else:
                print_agent_output(f"Review feedback is: {response}...", agent="REVIEWER")

        if "None" in response:
            return None
        return response

    # ======================================================================
    # Phase 3: 四维并行 reviewer（拆分架构，解决注意力稀释 + 超时连坐）
    # ======================================================================
    async def _review_financial_draft(self, draft_state, task, financial_context, revision_notes):
        """金融研报四维校验 — 拆分为 4 个独立并行调用 + 程序化预检。

        架构改进（vs Phase 2 单调用包办四维）：
        1. 数据准确性：程序化数值提取 + 精确比对（解决 ~1% 小偏差失明） + LLM 语义兜底
        2. 逻辑完整性：独立 LLM 调用，强制对比草稿主张与参考数据
        3. 合规性：独立短 prompt，无数据比对负担（解决注意力稀释）
        4. 格式规范：程序化预检（标题/链接）+ LLM 兜底（解决几乎失明）

        四维并行（asyncio.gather），总时长 ≈ 单维度时长，避免串行累加。
        每个维度独立返回 findings 列表，最终合并；任一维度超时不连坐其它维度。
        """
        draft = draft_state.get("draft") or ""
        ticker = financial_context.get("ticker", "")
        company_name = financial_context.get("company_name", ticker)

        # 并行跑四个维度
        dim_tasks = [
            self._review_data_accuracy(draft, financial_context, task),
            self._review_logic(draft, financial_context, task),
            self._review_compliance(draft, task),
            self._review_format(draft, task),
        ]
        try:
            results = await asyncio.gather(*dim_tasks, return_exceptions=True)
        except Exception as e:
            print_agent_output(f"Financial review parallel failed: {e}", agent="REVIEWER")
            results = [[], [], [], []]

        dim_names = ["DATA ACCURACY", "LOGICAL INTEGRITY", "COMPLIANCE", "FORMAT"]
        all_findings: list[str] = []
        per_dim_status: list[str] = []
        for name, res in zip(dim_names, results):
            if isinstance(res, Exception):
                per_dim_status.append(f"{name}: ERROR ({res})")
                # 单维度出错不影响其它维度的发现
                continue
            if res:
                all_findings.extend(res)
                per_dim_status.append(f"{name}: {len(res)} issue(s)")
            else:
                per_dim_status.append(f"{name}: PASS")

        if task.get("verbose"):
            status_summary = " | ".join(per_dim_status)
            print_agent_output(
                f"Financial review per-dim: {status_summary}", agent="REVIEWER"
            )
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs", "financial_review_feedback",
                    f"Financial review: {status_summary}",
                    self.websocket,
                )

        if not all_findings:
            return None

        # 合并四维发现为统一修订意见
        revision_notes_block = ""
        if revision_notes:
            revision_notes_block = (
                f"\n[Previous revision notes already addressed]:\n{revision_notes}\n"
            )

        header = (
            f"Financial review findings for {company_name} ({ticker}):\n"
            f"Per-dimension status: {' | '.join(per_dim_status)}\n"
            f"{revision_notes_block}\n"
            f"Required revisions:\n"
        )
        body = "\n".join(f"- [{dim}] {f}" for f, dim in _pair_finding_with_dim(results, dim_names))
        return header + body

    # -------------------- 维度 1: 数据准确性 --------------------
    def _programmatic_data_check(self, draft: str, fc: dict) -> list[str]:
        """程序化数值比对：从草稿中提取参考数据中已知字段的数字，精确比较。

        解决 LLM 对 ~1% 小偏差不敏感的问题。LLM 数值敏感度有限，
        但程序可以精确到浮点比对，捕捉任意小偏差。
        """
        findings: list[str] = []
        # 字段 → (在 fc 中的键, 在草稿中可能的别名)
        field_specs = [
            ("pe_ratio", ["PE", "市盈率", "P/E"]),
            ("pb_ratio", ["PB", "市净率", "P/B"]),
            ("roe", ["ROE", "净资产收益率"]),
            ("profit_margin", ["净利率", "净利润率"]),
        ]
        for field, aliases in field_specs:
            true_val = fc.get(field)
            if not isinstance(true_val, (int, float)) or true_val is None:
                continue
            field_issue_found = False  # 同一字段最多报一次
            for alias in aliases:
                if field_issue_found:
                    break
                # 精确匹配：alias 后允许 0-20 个非数字字符，然后捕获一个数字
                # 这样不会跨字段抓数字（如 PE 的窗口不会卷入 PB 的值）
                pattern = re.escape(alias) + r"[^\d]{0,20}(\d+\.\d+|\d+)"
                for m in re.finditer(pattern, draft):
                    if field_issue_found:
                        break
                    n_str = m.group(1)
                    try:
                        n = float(n_str)
                    except ValueError:
                        continue
                    if abs(n - true_val) < 1e-6:
                        continue  # 与真值一致
                    # 偏差阈值：>0.5% 才 flag（避免四舍五入噪声）
                    if abs(true_val) > 1e-6:
                        rel_delta = abs(n - true_val) / abs(true_val)
                        if rel_delta > 0.005:
                            findings.append(
                                f"{alias} 数据可疑：草稿在 '{alias}' 附近出现 {n}，"
                                f"但参考数据中 {field}={true_val}（相对偏差 {rel_delta:.1%}）。"
                            )
                            field_issue_found = True
                            break
        return findings

    async def _review_data_accuracy(self, draft: str, fc: dict, task: dict) -> list[str]:
        """数据准确性维度：程序化预检 + LLM 语义兜底。"""
        # Phase 1: 程序化精确比对 — 抓小偏差
        prog_findings = self._programmatic_data_check(draft, fc)

        # Phase 2: LLM 语义层兜底 — 抓程序抓不到的（如单位错误、张冠李戴）
        ticker = fc.get("ticker", "")
        company_name = fc.get("company_name", ticker)
        fin_ref_parts = [
            f"Ticker: {ticker}",
            f"Company: {company_name}",
            f"PE Ratio (actual): {fc.get('pe_ratio', 'N/A')}",
            f"PB Ratio (actual): {fc.get('pb_ratio', 'N/A')}",
            f"ROE (actual): {fc.get('roe', 'N/A')}%",
            f"Revenue Growth (actual): {fc.get('revenue_growth', 'N/A')}%",
            f"Profit Margin (actual): {fc.get('profit_margin', 'N/A')}%",
            f"Market Cap: {fc.get('market_cap', 'N/A')}",
        ]
        peers = fc.get("peers", []) or []
        if peers:
            peer_lines = [
                f"  {p.get('ticker','?')}: PE={p.get('pe','-')}, PB={p.get('pb','-')}, "
                f"ROE={p.get('roe','-')}%"
                for p in peers[:6]
            ]
            fin_ref_parts.append("Peer Data (actual):\n" + "\n".join(peer_lines))
        fin_reference = "\n".join(fin_ref_parts)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a senior financial data auditor. Your ONLY task is to verify "
                    "the accuracy of financial numbers cited in the draft against the reference "
                    "data. You do NOT check compliance, logic, or format — other reviewers handle those.\n"
                    "Report ONLY concrete numeric discrepancies where the draft cites a specific "
                    "number that conflicts with the reference. Do NOT flag missing data or "
                    "stylistic issues — only factual mismatches. If no discrepancies, return 'PASS'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reference data:\n{fin_reference}\n\n"
                    f"Draft:\n{draft}\n\n"
                    f"List any specific number in the draft that conflicts with the reference. "
                    f"Be precise: cite the draft's number and the reference's number. "
                    f"Only flag clear mismatches (not missing data, not style). "
                    f"If none, reply 'PASS'."
                ),
            },
        ]
        try:
            response = await call_model(prompt, model=task.get("model"))
        except Exception as e:
            # LLM 失败时仍有程序化结果兜底
            return prog_findings if prog_findings else []

        llm_findings: list[str] = []
        if response and "PASS" not in response.upper()[:20]:
            # 把 LLM 回复按行/段拆成单条发现
            for line in response.split("\n"):
                line = line.strip().lstrip("-*•0123456789. )")
                if line and len(line) > 10:
                    llm_findings.append(line)

        # 去重（程序化与 LLM 可能都报同一偏差）
        merged = list(prog_findings)
        for f in llm_findings:
            if not any(_similar(f, existing) for existing in merged):
                merged.append(f)
        return merged

    # -------------------- 维度 2: 逻辑完整性 --------------------
    async def _review_logic(self, draft: str, fc: dict, task: dict) -> list[str]:
        """逻辑完整性维度：独立 LLM 调用，强制对比草稿主张与参考数据。"""
        ticker = fc.get("ticker", "")
        company_name = fc.get("company_name", ticker)
        # 只传与逻辑判断相关的参考数据（避免 prompt 过长）
        logic_ref = (
            f"Reference: {company_name} ({ticker}) — "
            f"PE={fc.get('pe_ratio','N/A')}, "
            f"PB={fc.get('pb_ratio','N/A')}, "
            f"ROE={fc.get('roe','N/A')}%, "
            f"Profit Margin={fc.get('profit_margin','N/A')}%, "
            f"Revenue Growth={fc.get('revenue_growth','N/A')}%"
        )
        peers = fc.get("peers", []) or []
        peer_summary = ""
        if peers:
            peer_summary = (
                " | Peers: "
                + " | ".join(
                    f"{p.get('ticker','?')}(ROE={p.get('roe','-')}%)"
                    for p in peers[:4]
                )
            )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a senior investment analyst reviewing the LOGICAL INTEGRITY of a "
                    "research report draft. Your ONLY task is to check whether the draft's "
                    "conclusions and claims are supported by — and consistent with — the reference "
                    "financial data. You do NOT check data accuracy, compliance, or format.\n"
                    "Flag ONLY claims/conclusions that contradict the reference data or lack "
                    "evidentiary support. Examples: claiming 'industry-leading ROE' when ROE is "
                    "below peers; 'strong buy' with negligible downside when PE is elevated and "
                    "ROE is low; conclusions that cherry-pick data. If no logical issues, return 'PASS'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{logic_ref}{peer_summary}\n\n"
                    f"Draft:\n{draft}\n\n"
                    f"List any claim or conclusion in the draft that contradicts the reference data "
                    f"or lacks evidentiary support. Be specific: cite the claim and the contradiction. "
                    f"If none, reply 'PASS'."
                ),
            },
        ]
        try:
            response = await call_model(prompt, model=task.get("model"))
        except Exception:
            return []
        return _parse_llm_findings(response)

    # -------------------- 维度 3: 合规性 --------------------
    async def _review_compliance(self, draft: str, task: dict) -> list[str]:
        """合规性维度：独立短 prompt，无数据比对负担（解决注意力稀释）。

        合规问题（保证上涨/零风险/过度承诺）不依赖任何金融参考数据，
        让 reviewer 在短 prompt 下专注识别这些「显眼」问题。
        """
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a compliance reviewer for financial research reports. Your ONLY task "
                    "is to identify non-compliant statements. You do NOT check data accuracy, logic, "
                    "or format.\n"
                    "Flag ONLY statements that fall into these categories:\n"
                    "1. Guaranteed/promised returns ('保证上涨', 'guaranteed return', '预计年化收益不低于X%')\n"
                    "2. Risk-free claims ('零风险', 'risk-free', '下行风险可忽略')\n"
                    "3. Misleading absolute statements ('绝对优势', '稳居第一' without support)\n"
                    "4. Missing risk disclosures when investment recommendations are made\n"
                    "Be conservative: only flag clear-cut violations, not borderline phrasing. "
                    "If no violations, return 'PASS'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Draft:\n{draft}\n\n"
                    f"List any non-compliant statements (guaranteed returns, risk-free claims, "
                    f"misleading absolutes, missing risk disclosures). Cite the exact phrase. "
                    f"If none, reply 'PASS'."
                ),
            },
        ]
        try:
            response = await call_model(prompt, model=task.get("model"))
        except Exception:
            return []
        return _parse_llm_findings(response)

    # -------------------- 维度 4: 格式规范 --------------------
    def _programmatic_format_check(self, draft: str) -> list[str]:
        """程序化格式预检：标题数、链接数、段落结构。

        解决 LLM 对「本来应该有什么」失明的问题。
        LLM 看到无标题正文不会知道「原本应有标题」，但程序可以基于
        金融研报规范硬性检查最低标题数、链接数。
        """
        findings: list[str] = []
        headers = re.findall(r"^#{2,3}\s+\S", draft, flags=re.MULTILINE)
        if len(headers) < 1:
            findings.append(
                "格式问题：草稿缺少 markdown 章节标题（### / ##），"
                "专业研报应有清晰的小节结构。"
            )
        # 数据来源链接检查
        links = re.findall(r"\[.+?\]\(https?://.+?\)", draft)
        if len(links) < 1:
            findings.append(
                "格式问题：草稿未包含任何数据来源超链接 [text](url)，"
                "专业研报应标注关键数据来源。"
            )
        return findings

    async def _review_format(self, draft: str, task: dict) -> list[str]:
        """格式规范维度：程序化预检 + LLM 兜底。"""
        # Phase 1: 程序化预检 — 精确检查最低结构要求
        prog_findings = self._programmatic_format_check(draft)

        # Phase 2: LLM 兜底 — 检查程序无法判定的（如标题层级错乱、引用格式不规范）
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a formatting reviewer for financial research reports. Your ONLY task "
                    "is to check professional formatting standards. You do NOT check data accuracy, "
                    "logic, or compliance.\n"
                    "Flag ONLY clear formatting violations:\n"
                    "1. Missing or broken markdown section structure (no ##/### headers when body warrants them)\n"
                    "2. Missing source citations for specific numbers (no [text](url) links)\n"
                    "3. Broken tables or lists\n"
                    "Do NOT flag stylistic preferences or content quality — only structural formatting. "
                    "If no structural issues, return 'PASS'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Draft:\n{draft}\n\n"
                    f"List any clear structural formatting violations (missing headers, missing "
                    f"source links, broken tables/lists). If none, reply 'PASS'."
                ),
            },
        ]
        try:
            response = await call_model(prompt, model=task.get("model"))
        except Exception:
            return prog_findings
        llm_findings = _parse_llm_findings(response)
        merged = list(prog_findings)
        for f in llm_findings:
            if not any(_similar(f, existing) for existing in merged):
                merged.append(f)
        return merged

    async def run(self, draft_state: dict):
        task = draft_state.get("task")
        guidelines = task.get("guidelines")
        to_follow_guidelines = task.get("follow_guidelines")

        # --- Phase 2: Force financial review ---
        financial_context = draft_state.get("financial_context") or {}
        is_financial = bool(financial_context)
        # -------------------------------------

        review = None
        #判断是否需要审查
        if to_follow_guidelines or is_financial:
            if is_financial:
                print_agent_output(
                    f"Reviewing financial draft with 4-dim parallel checks...",
                    agent="REVIEWER",
                )
            else:
                print_agent_output(f"Reviewing draft...", agent="REVIEWER")

            if task.get("verbose"):
                print_agent_output(
                    f"Following guidelines {guidelines}...", agent="REVIEWER"
                )

            review = await self.review_draft(draft_state)
        else:
            print_agent_output(f"Ignoring guidelines...", agent="REVIEWER")
        return {"review": review}


# ======================================================================
# 辅助函数
# ======================================================================
def _parse_llm_findings(response: str | None) -> list[str]:
    """把 LLM 回复按行/段拆成单条发现。'PASS' 视为无发现。"""
    if not response:
        return []
    if "PASS" in response.upper()[:20]:
        return []
    findings: list[str] = []
    for line in response.split("\n"):
        line = line.strip().lstrip("-*•0123456789. )")
        if line and len(line) > 5:
            findings.append(line)
    return findings


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """粗略判定两段文本是否相似（用于去重）。

    基于 token 级 Jaccard 相似度，足够过滤「同一发现被程序和 LLM 都报」的情况。
    """
    if not a or not b:
        return False
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) >= threshold


def _pair_finding_with_dim(results: list, dim_names: list[str]) -> list[tuple[str, str]]:
    """把每个维度的 findings 与维度名配对，用于最终输出。

    results 与 dim_names 一一对应；每个 result 可能是 list[str]（findings）
    或 Exception（该维度出错）。
    """
    paired: list[tuple[str, str]] = []
    for name, res in zip(dim_names, results):
        if isinstance(res, Exception) or not res:
            continue
        for f in res:
            paired.append((f, name))
    return paired
