from .utils.file_formats import \
    write_md_to_pdf, \
    write_md_to_word, \
    write_text_to_md

from .utils.views import print_agent_output

import re
import logging
import json
import os

logger = logging.getLogger(__name__)


class PublisherAgent:
    def __init__(self, output_dir: str, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.output_dir = output_dir.strip()
        self.headers = headers or {}
        
    async def publish_research_report(self, research_state: dict, publish_formats: dict):
        layout = self.generate_layout(research_state)
        await self.write_report_by_formats(layout, publish_formats)

        # --- Phase 2: Save financial data ---
        await self._save_financial_data(research_state)
        # -----------------------------------

        return layout

    def generate_layout(self, research_state: dict):
        sections = []
        for subheader in research_state.get("research_data", []):
            text = ""
            if isinstance(subheader, dict):
                for key, value in subheader.items():
                    text = f"{value}"
            else:
                text = f"{subheader}"

            # Strip per-section intro/conclusion/references —
            # these are already provided centrally by the Writer
            text = re.sub(r'\n##\s*结论\s*\n.*', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\n##\s*参考文献\s*\n.*', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\n##\s*引言\s*\n.*', '', text, flags=re.DOTALL | re.IGNORECASE)
            if text.strip():
                sections.append(text.strip())
        
        sections_text = '\n\n'.join(sections)

        # Demote all ## to ### — sub-agent internal headings are nested content,
        # not top-level report sections
        sections_text = re.sub(r'^## ', '### ', sections_text, flags=re.MULTILINE)
        references = '\n'.join(f"{reference}" for reference in research_state.get("sources", []))
        headers = research_state.get("headers", {})

        # --- Phase 2: Financial disclaimer ---
        financial_data = research_state.get("financial_data") or {}
        disclaimer = ""
        if financial_data:
            ticker = financial_data.get("ticker", "")
            disclaimer = (
                f"\n\n---\n\n"
                f"> ⚠️ **免责声明**：本报告仅供参考，不构成投资建议。\n"
                f"> 数据来源包括 Yahoo Finance{f' ({ticker})' if ticker else ''} 及公开渠道。\n"
                f"> 投资有风险，入市需谨慎。过往表现不代表未来收益。\n"
            )
        # -----------------------------------

        # Strip leading h1/h2 from intro/conclusion (LLM often repeats title)
        intro = (research_state.get('introduction') or "").strip()
        intro = re.sub(r'^#\s+.*?\n+', '', intro).strip()
        conclusion = (research_state.get('conclusion') or "").strip()
        conclusion = re.sub(r'^#\s+.*?\n+', '', conclusion).strip()

        layout = f"""# {headers.get('title')}
#### {headers.get("date")}: {research_state.get('date')}

## {headers.get("introduction")}
{intro}

## {headers.get("table_of_contents")}
{research_state.get('table_of_contents')}

{sections_text}

## {headers.get("conclusion")}
{conclusion}
{disclaimer}
## {headers.get("references")}
{references}
"""
        return layout

    async def write_report_by_formats(self, layout:str, publish_formats: dict):
        if publish_formats.get("pdf"):
            await write_md_to_pdf(layout, self.output_dir)
        if publish_formats.get("docx"):
            await write_md_to_word(layout, self.output_dir)
        if publish_formats.get("markdown"):
            await write_text_to_md(layout, self.output_dir)

    async def run(self, research_state: dict):
        task = research_state.get("task")
        publish_formats = task.get("publish_formats")
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "publisher_start", f"Publishing final research report based on retrieved data...", self.websocket)
        else:
            print_agent_output(output="Publishing final research report based on retrieved data...", agent="PUBLISHER")
        final_research_report = await self.publish_research_report(research_state, publish_formats)
        return {"report": final_research_report}

    async def _save_financial_data(self, research_state: dict):
        """保存金融原始数据 JSON 到输出目录，便于调试和验证。"""
        financial_data = research_state.get("financial_data") or {}
        if not financial_data:
            return
        try:
            # Make serializable: convert NaN to null
            def _safe_serialize(obj):
                if isinstance(obj, dict):
                    return {k: _safe_serialize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_safe_serialize(v) for v in obj]
                if isinstance(obj, float):
                    return round(obj, 2) if obj == obj else None  # NaN check
                return obj

            output = {
                "ticker": financial_data.get("ticker"),
                "overview": _safe_serialize(financial_data.get("overview", {})),
                "statements": _safe_serialize(financial_data.get("statements", {})),
                "industry_peers": _safe_serialize(
                    research_state.get("industry_peers", [])
                ),
            }
            path = os.path.join(self.output_dir, "financial_data.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"Financial data saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save financial data: {e}")
