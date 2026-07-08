from .utils.views import print_agent_output
from .utils.llms import call_model
import json

sample_revision_notes = """
{
  "draft": { 
    draft title: The revised draft that you are submitting for review 
  },
  "revision_notes": Your message to the reviewer about the changes you made to the draft based on their feedback
}
"""


class ReviserAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def revise_draft(self, draft_state: dict):
        """
        Review a draft article
        :param draft_state:
        :return:
        """
        review = draft_state.get("review")
        task = draft_state.get("task")
        draft_report = draft_state.get("draft")

        # --- Phase 2: Financial revision context ---
        financial_context = draft_state.get("financial_context") or {}
        is_financial = bool(financial_context)

        system_content = (
            "You are an expert writer. Your goal is to revise drafts based on reviewer notes. "
            "You must keep all other aspects of the draft the same."
        )
        if is_financial:
            # Build financial reference for the reviser
            fin_ref = ""
            if financial_context:
                fin_ref = (
                    "Financial Data Reference (TRUSTED NUMBERS):\n"
                    f"- PE Ratio: {financial_context.get('pe_ratio', 'N/A')}\n"
                    f"- PB Ratio: {financial_context.get('pb_ratio', 'N/A')}\n"
                    f"- ROE: {financial_context.get('roe', 'N/A')}%\n"
                    f"- Revenue Growth: {financial_context.get('revenue_growth', 'N/A')}%\n"
                    f"- Profit Margin: {financial_context.get('profit_margin', 'N/A')}%\n"
                    "\nIf the reviewer flagged data inaccuracies, you MUST use ONLY "
                    "the numbers above — do NOT fabricate or guess."
                )
            system_content = (
                "You are a senior financial report reviser. Your goal is to revise "
                "financial research drafts based on reviewer notes. "
                "If the reviewer points out data inconsistencies, you MUST use the "
                "financial data reference provided below — never guess or fabricate numbers. "
                "You must keep all other aspects of the draft the same. "
                "You maintain the professional 8-section financial report format.\n" + fin_ref
            )

        prompt = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"""Draft:\n{draft_report}\n\nReviewer's notes:\n{review}\n\n
You have been tasked by your reviewer with revising the following draft.
You MUST follow the reviewer's notes and address ALL of the points they raised.
If the reviewer flagged data inaccuracies and financial reference data is provided,
you MUST use those exact numbers — do NOT fabricate or guess.
Write a complete revised draft that incorporates all reviewer feedback.
Keep all other aspects of the draft the same.
You MUST return nothing but a JSON in the following format:
{sample_revision_notes}
""",
            },
        ]

        response = await call_model(
            prompt,
            model=task.get("model"),
            response_format="json",
        )
        return response

    async def run(self, draft_state: dict):
        print_agent_output(f"Rewriting draft based on feedback...", agent="REVISOR")
        revision = await self.revise_draft(draft_state)

        if draft_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "revision_notes",
                    f"Revision notes: {revision.get('revision_notes')}",
                    self.websocket,
                )
            else:
                print_agent_output(
                    f"Revision notes: {revision.get('revision_notes')}", agent="REVISOR"
                )

        return {
            "draft": revision.get("draft"),
            "revision_notes": revision.get("revision_notes"),
        }
