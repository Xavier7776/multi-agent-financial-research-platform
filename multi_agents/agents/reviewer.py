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

    async def _review_financial_draft(self, draft_state, task, financial_context, revision_notes):
        """Review a financial research report draft with data accuracy checks.

        Upgrades the standard review to include:
        1. Cross-verification of financial metrics against the provided context
        2. Data consistency checks (do cited numbers match known values?)
        3. Logical completeness of the investment thesis
        4. Compliance (no over-promising or misleading statements)
        5. 8-section financial report format adherence
        """
        ticker = financial_context.get("ticker", "")
        company_name = financial_context.get("company_name", ticker)
        peers = financial_context.get("peers", [])

        # Build financial reference data for cross-verification
        fin_ref_parts = [
            f"Ticker: {ticker}",
            f"Company: {company_name}",
            f"PE Ratio (actual): {financial_context.get('pe_ratio', 'N/A')}",
            f"PB Ratio (actual): {financial_context.get('pb_ratio', 'N/A')}",
            f"ROE (actual): {financial_context.get('roe', 'N/A')}%",
            f"Revenue Growth (actual): {financial_context.get('revenue_growth', 'N/A')}%",
            f"Profit Margin (actual): {financial_context.get('profit_margin', 'N/A')}%",
            f"Market Cap: {financial_context.get('market_cap', 'N/A')}",
        ]
        fin_reference = "\n".join(fin_ref_parts)

        peer_ref = ""
        if peers:
            peer_lines = [
                f"  {p.get('ticker','?')}: PE={p.get('pe','-')}, PB={p.get('pb','-')}, "
                f"ROE={p.get('roe','-')}%"
                for p in peers[:6]
            ]
            peer_ref = "Peer Data (actual):\n" + "\n".join(peer_lines)

        revise_instruction = ""
        if revision_notes:
            revise_instruction = (
                f"The reviser has already revised the draft based on previous notes:\n"
                f"{revision_notes}\n"
                f"Provide additional feedback only if critical.\n"
            )

        review_prompt = f"""You are reviewing a financial research report draft for {company_name} ({ticker}).

【Financial Data Reference — Use this to verify accuracy】
{fin_reference}
{peer_ref}

【Draft to Review】
{draft_state.get("draft")}

【Review Criteria — Check ALL of the following】
1. DATA ACCURACY: Does the draft cite financial numbers (PE, PB, ROE, revenue growth, etc.) that match the reference data above? 
   - If ANY number in the draft differs from the reference, flag it with the specific discrepancy.
   - Example: "Draft says PE=30 but actual PE=35.96"

2. LOGICAL INTEGRITY: Is the investment logic chain complete? 
   - Are conclusions supported by data?
   - Do peer comparisons make sense?

3. COMPLIANCE: Does the draft contain any of the following?
   - Over-promising returns ("guaranteed to go up", "risk-free")
   - Misleading statements that cherry-pick data
   - Missing risk disclosures

4. FORMAT: Does the draft follow professional financial report standards?
   - Clear section structure
   - Data sources cited
   - No fabricated numbers

{revise_instruction}
If the draft passes ALL criteria above, return None.
If any criterion fails, return specific revision notes explaining exactly what needs to change.
"""
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a senior financial research reviewer. Your role is to rigorously "
                    "verify the accuracy of financial data in research drafts. You cross-check "
                    "every cited number against the provided reference data. You flag any "
                    "discrepancy, no matter how small. You also evaluate the logical integrity, "
                    "compliance, and professional formatting of financial reports.\n"
                ),
            },
            {"role": "user", "content": review_prompt},
        ]

        response = await call_model(prompt, model=task.get("model"))

        if task.get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs", "financial_review_feedback",
                    f"Financial review feedback: {response}...",
                    self.websocket,
                )
            else:
                print_agent_output(f"Financial review: {response}", agent="REVIEWER")

        if "None" in response:
            return None
        return response

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
                    f"Reviewing financial draft with data accuracy checks...",
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
