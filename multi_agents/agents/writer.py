from datetime import datetime
import json5 as json
from .utils.views import print_agent_output
from .utils.llms import call_model

sample_json = """
{
  "table_of_contents": A table of contents in markdown syntax (using '-') based on the research headers and subheaders,
  "introduction": An indepth introduction to the topic in markdown syntax and hyperlink references to relevant sources,
  "conclusion": A conclusion to the entire research based on all research data in markdown syntax and hyperlink references to relevant sources,
  "sources": A list with strings of all used source links in the entire research data in markdown syntax and apa citation format. For example: ['-  Title, year, Author [source url](source)', ...]
}
"""


class WriterAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers

    def get_headers(self, research_state: dict):
        # 根据语言设置返回对应的章节标题，默认中文
        task = research_state.get("task", {}) or {}
        language = task.get("language", "chinese")
        headers_map = {
            "chinese": {
                "date": "日期",
                "introduction": "引言",
                "table_of_contents": "目录",
                "conclusion": "结论",
                "references": "参考文献",
            },
            "english": {
                "date": "Date",
                "introduction": "Introduction",
                "table_of_contents": "Table of Contents",
                "conclusion": "Conclusion",
                "references": "References",
            },
            "japanese": {
                "date": "日付",
                "introduction": "はじめに",
                "table_of_contents": "目次",
                "conclusion": "結論",
                "references": "参考文献",
            },
            "korean": {
                "date": "날짜",
                "introduction": "서론",
                "table_of_contents": "목차",
                "conclusion": "결론",
                "references": "참고문헌",
            },
        }
        h = headers_map.get(language, headers_map["chinese"])
        return {
            "title": research_state.get("title"),
            "date": h["date"],
            "introduction": h["introduction"],
            "table_of_contents": h["table_of_contents"],
            "conclusion": h["conclusion"],
            "references": h["references"],
        }

    #写引言和结论
    async def write_sections(self, research_state: dict):
        query = research_state.get("title")
        data = research_state.get("research_data")
        task = research_state.get("task")
        follow_guidelines = task.get("follow_guidelines")
        guidelines = task.get("guidelines")

        # 读取语言设置，默认中文
        language = task.get("language", "chinese")
        language_instruction = {
            "chinese": "You MUST write the introduction and conclusion in Chinese (中文).",
            "english": "You MUST write the introduction and conclusion in English.",
            "japanese": "You MUST write the introduction and conclusion in Japanese (日本語).",
            "korean": "You MUST write the introduction and conclusion in Korean (한국어).",
        }.get(language, "You MUST write the introduction and conclusion in Chinese (中文).")

        # --- Phase 2: Financial report mode ---
        financial_data = research_state.get("financial_data") or {}
        industry_peers = research_state.get("industry_peers") or []
        is_financial = bool(financial_data)

        if is_financial:
            return await self._write_financial_sections(
                query, data, task, language_instruction,
                financial_data, industry_peers, follow_guidelines, guidelines,
            )

        # --- Original generic mode ---
        prompt = [
            {
                "role": "system",
                "content": "You are a research writer. Your sole purpose is to write a well-written "
                "research reports about a "
                "topic based on research findings and information.\n ",
            },
            {
                "role": "user",
                "content": f"Today's date is {datetime.now().strftime('%d/%m/%Y')}\n."
                f"Query or Topic: {query}\n"
                f"Research data: {str(data)}\n"
                f"Your task is to write an in depth, well written and detailed "
                f"introduction and conclusion to the research report based on the provided research data. "
                f"Do not include headers in the results.\n"
                f"You MUST include any relevant sources to the introduction and conclusion as markdown hyperlinks -"
                f"For example: 'This is a sample text. ([url website](url))'\n\n"
                f"{language_instruction}\n\n"
                f"{f'You must follow the guidelines provided: {guidelines}' if follow_guidelines else ''}\n"
                f"You MUST return nothing but a JSON in the following format (without json markdown):\n"
                f"{sample_json}\n\n",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )
        # DEBUG: check if Writer output is truncated
        intro_len = len(str(response.get("introduction", "")) or "")
        concl_len = len(str(response.get("conclusion", "")) or "")
        toc_len = len(str(response.get("table_of_contents", "")) or "")
        sources_len = len(response.get("sources", []) or [])
        print_agent_output(
            f"[Writer DEBUG] intro={intro_len}chars, conclusion={concl_len}chars, "
            f"toc={toc_len}chars, sources={sources_len}items, "
            f"keys={list(response.keys()) if response else 'None'}",
            agent="WRITER",
        )
        return response

    async def _write_financial_sections(
        self, query, data, task, language_instruction,
        financial_data, industry_peers, follow_guidelines, guidelines,
    ):
        """Write the introduction and conclusion for a financial research report.

        Uses a financial-specific prompt that requires:
        - Data citations with specific numbers (revenue growth %, PE percentile, etc.)
        - Investment thesis and risk-return assessment
        - Source attribution for all financial metrics
        - No fabrication of numbers
        """
        ticker = financial_data.get("ticker", "")
        overview = financial_data.get("overview", {})
        company_name = overview.get("name", ticker)
        statements = financial_data.get("statements", {})

        # Build financial context summary for the prompt (skip N/A/None)
        fin_context_parts = [f"Ticker: {ticker}", f"Company: {company_name}"]
        if overview:
            _fields = {
                "Sector": overview.get("sector"),
                "Industry": overview.get("industry"),
                "Market Cap": overview.get("market_cap"),
                "PE Ratio": overview.get("pe_ratio"),
                "PB Ratio": overview.get("pb_ratio"),
                "ROE": overview.get("roe"),
                "Revenue Growth": overview.get("revenue_growth"),
                "Profit Margin": overview.get("profit_margin"),
                "Dividend Yield": overview.get("dividend_yield"),
            }
            for label, v in _fields.items():
                if v is not None and v != "":
                    suffix = "%" if label in ("ROE", "Revenue Growth", "Profit Margin", "Dividend Yield") else ""
                    fin_context_parts.append(f"{label}: {v}{suffix}")

        peer_context = ""
        if industry_peers:
            peer_lines = [
                f"  [{p.get('ticker','?')}] {p.get('name','')} — "
                f"PE: {p.get('pe','-')}, PB: {p.get('pb','-')}, "
                f"ROE: {p.get('roe','-')}%, Rev Growth: {p.get('revenue_growth','-')}%"
                for p in industry_peers[:6]
            ]
            peer_context = "Industry Peers:\n" + "\n".join(peer_lines)

        financial_overview = "\n".join(fin_context_parts)

        financial_sample_json = """{
  "table_of_contents": A table of contents in markdown syntax (using '-') based on the 8-section financial report headers and subheaders,
  "introduction": A professional financial research report introduction in markdown syntax. Must include: (1) research purpose and scope, (2) core investment thesis summary, (3) target audience. Use specific data points (revenue, PE, market cap) from the financial overview. Hyperlink all data sources.
  "conclusion": A professional conclusion in markdown syntax that summarizes: (1) investment logic and key drivers, (2) valuation analysis highlights with peer comparison, (3) risk-return assessment, (4) final investment recommendation (Buy/Hold/Sell). Cite specific valuation metrics. Hyperlink all data sources.
  "sources": A list with strings of all used source links in markdown syntax and apa citation format. For example: ['-  Title, year, Author [source url](source)', ...]
}"""

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a senior financial research report writer with expertise in "
                    "equity analysis, valuation, and investment research. Your reports are "
                    "data-driven, professionally structured, and suitable for institutional investors. "
                    "You NEVER fabricate financial data. "
                    "When citing valuation metrics (PE, PB, ROE), always include peer comparison context. "
                    "Your writing style is precise, analytical, and avoids vague language.\n"
                    "CRITICAL: The research data may contain different metric values from different "
                    "analysts. Pick ONE consistent value per key metric and use it throughout."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Today's date is {datetime.now().strftime('%d/%m/%Y')}\n\n"
                    f"【Financial Report Context】\n"
                    f"Report Title: {query}\n"
                    f"Financial Overview ({ticker}):\n{financial_overview}\n\n"
                    f"{peer_context}\n\n"
                    f"【Research Data】\n{str(data)}\n\n"
                    f"【Writing Task】\n"
                    f"Write an in-depth introduction and conclusion for this financial "
                    f"research report based on the provided research data and financial overview.\n\n"
                    f"Introduction requirements:\n"
                    f"- State the research purpose and the investment question being addressed\n"
                    f"- Present the core investment thesis with supporting data points from the financial overview\n"
                    f"- Briefly preview the 8-section report structure\n"
                    f"- Reference specific metrics: revenue, PE ratio, market cap, ROE\n\n"
                    f"Conclusion requirements:\n"
                    f"- Summarize the investment logic: why {company_name} ({ticker}) is attractive/unattractive\n"
                    f"- Present valuation analysis: compare {ticker}'s PE/PB/ROE against peers\n"
                    f"- Assess risk-return: key upside drivers vs. downside risks\n"
                    f"- Provide a clear investment recommendation (Buy/Hold/Sell) with rationale\n"
                    f"- All numbers MUST come from the financial overview or research data above\n\n"
                    f"{language_instruction}\n\n"
                    f"{f'You must follow the guidelines provided: {guidelines}' if follow_guidelines else ''}\n"
                    f"You MUST return nothing but a JSON in the following format (without json markdown):\n"
                    f"{financial_sample_json}\n\n"
                ),
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )
        # DEBUG: check if financial Writer output is complete
        intro_len = len(str(response.get("introduction", "")) or "")
        concl_len = len(str(response.get("conclusion", "")) or "")
        toc_len = len(str(response.get("table_of_contents", "")) or "")
        sources_len = len(response.get("sources", []) or [])
        print_agent_output(
            f"[FinancialWriter DEBUG] intro={intro_len}chars, conclusion={concl_len}chars, "
            f"toc={toc_len}chars, sources={sources_len}items, "
            f"keys={list(response.keys()) if response else 'None'}",
            agent="WRITER",
        )
        return response

    async def revise_headers(self, task: dict, headers: dict):
        # 读取语言设置，默认中文
        language = task.get("language", "chinese")
        language_instruction = {
            "chinese": "You MUST write the headers in Chinese (中文).",
            "english": "You MUST write the headers in English.",
            "japanese": "You MUST write the headers in Japanese (日本語).",
            "korean": "You MUST write the headers in Korean (한국어).",
        }.get(language, "You MUST write the headers in Chinese (中文).")
        prompt = [
            {
                "role": "system",
                "content": """You are a research writer. 
Your sole purpose is to revise the headers data based on the given guidelines.""",
            },
            {
                "role": "user",
                "content": f"""Your task is to revise the given headers JSON based on the guidelines given.
You are to follow the guidelines but the values should be in simple strings, ignoring all markdown syntax.
{language_instruction}
You must return nothing but a JSON in the same format as given in headers data.
Guidelines: {task.get("guidelines")}\n
Headers Data: {headers}\n
""",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )
        return {"headers": response}

    async def run(self, research_state: dict):
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "writer_start",
                f"Writing final research report based on research data...",
                self.websocket,
            )
        else:
            print_agent_output(
                f"Writing final research report based on research data...",
                agent="WRITER",
            )
        #写结论和引言
        # {
        #     "table_of_contents": "## 一、市场规模\n## 二、技术突破\n## 三、中国AI企业...",
        #     "introduction": "2024年人工智能产业...（正文内容）",
        #     "conclusion": "综上所述...（正文内容）",
        #     "sources": [
        #         "- Smith, 2024, AI Industry Report [https://...](https://...)",
        #         "- 工信部, 2024, 人工智能发展白皮书 [...]",
        #         ...
        #     ]
        # }
        research_layout_content = await self.write_sections(research_state)

        if research_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                research_layout_content_str = json.dumps(
                    research_layout_content, indent=2
                )
                await self.stream_output(
                    "logs",
                    "research_layout_content",
                    research_layout_content_str,
                    self.websocket,
                )
            else:
                print_agent_output(research_layout_content, agent="WRITER")

        headers = self.get_headers(research_state)
        if research_state.get("task").get("follow_guidelines"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "rewriting_layout",
                    "Rewriting layout based on guidelines...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    "Rewriting layout based on guidelines...", agent="WRITER"
                )
            headers = await self.revise_headers(
                task=research_state.get("task"), headers=headers
            )
            headers = headers.get("headers")

        return {**research_layout_content, "headers": headers}
