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
                    text = self._stringify(value)
            else:
                text = self._stringify(subheader)

            # Strip per-section intro/conclusion/references —
            # these are already provided centrally by the Writer.
            # Only strip the heading + content up to the next ## heading,
            # not to the end of the entire section (avoids losing content
            # that appears after a sub-agent's internal conclusion).
            text = self._strip_internal_section(text, '结论')
            text = self._strip_internal_section(text, '参考文献')
            text = self._strip_internal_section(text, '引言')
            if text.strip():
                sections.append(text.strip())

        # Process each section: keep first H2 as section title, demote rest to H3.
        # This replaces the old global demotion (re.sub(r'^## ', '### '))
        # which flattened all section titles to H3, making them the same level
        # as subsections and destroying the document hierarchy.
        processed_sections = [self._normalize_section_headings(s) for s in sections]
        sections_text = '\n\n'.join(processed_sections)
        references = self._format_references(research_state.get("sources", []))
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

        # Strip leading h1/h2 from intro/conclusion (LLM often repeats the section
        # title, which would duplicate the H2 emitted from the template below)
        intro = (research_state.get('introduction') or "").strip()
        intro = re.sub(r'^#{1,2}\s+.*?\n+', '', intro).strip()
        conclusion = (research_state.get('conclusion') or "").strip()
        conclusion = re.sub(r'^#{1,2}\s+.*?\n+', '', conclusion).strip()

        layout = f"""# {headers.get('title')}
**{headers.get("date")}**: {research_state.get('date')}

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
        # Defensive: remove consecutive duplicate headings (e.g. LLM repeats "## 引言")
        layout = self._dedup_consecutive_headings(layout)
        return layout

    @staticmethod
    def _stringify(value) -> str:
        """Safely convert a research_data value to string.

        Handles list values (sub-agent may return a list of paragraphs instead
        of a single string) by joining with double newlines. Without this,
        f"{value}" on a list produces Python repr like "['para1', 'para2']".
        """
        if isinstance(value, list):
            return "\n\n".join(str(item) for item in value)
        return str(value)

    @staticmethod
    def _normalize_section_headings(section: str) -> str:
        """Normalize heading levels within a section.

        Sub-agents use ## (H2) for the section title and ### (H3) for subsections.
        The old code globally demoted all ## to ###, which flattened the hierarchy
        and made section titles the same level as subsections.

        This method:
        - Keeps the first ## (or #) as ## (H2 section title)
        - Demotes subsequent ## (or #) to ### (H3 subsections)
        - Removes ** bold markers from all headings
        - Strips conflicting numbering from H3 headings
          (e.g., "### 一、 text" -> "### text", "### 2.1 text" -> "### text")
        """
        lines = section.split('\n')
        first_h2_found = False
        out = []
        for line in lines:
            m = re.match(r'^(#{1,4})\s+(.+?)\s*$', line)
            if m:
                level = len(m.group(1))
                text = m.group(2).replace('**', '').strip()
                if level <= 2:
                    if not first_h2_found:
                        first_h2_found = True
                        out.append(f'## {text}')
                    else:
                        text = PublisherAgent._clean_h3_text(text)
                        out.append(f'### {text}')
                elif level == 3:
                    text = PublisherAgent._clean_h3_text(text)
                    out.append(f'### {text}')
                else:
                    out.append(f'{"#" * level} {text}')
            else:
                out.append(line)
        return '\n'.join(out)

    @staticmethod
    def _clean_h3_text(text: str) -> str:
        """Remove conflicting numbering from H3 heading text.

        Strips:
        - Decimal numbering like "2.1 ", "3.2 "
        - Chinese numbering like "一、 ", "二、 "
        - Simple numeric numbering like "1. ", "2. "
        These conflict with H2 section numbering and make H3 look like H2.
        """
        text = re.sub(r'^\d+\.\d+\s*', '', text)  # "2.1 " first
        text = re.sub(r'^[\d一二三四五六七八九十]+[.、]\s*', '', text)  # "1. " or "一、 "
        return text.strip()

    @staticmethod
    def _strip_internal_section(text: str, heading_keyword: str) -> str:
        """Strip a sub-agent's internal section (引言/结论/参考文献).

        Only strips the heading and its content up to the next ## heading,
        not to the end of the entire text. This prevents losing content
        that appears after the internal section.
        """
        pattern = rf'\n##\s*{heading_keyword}\s*\n.*?(?=\n##\s|$)'
        return re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _format_references(sources) -> str:
        """Format source citations as a clean numbered list.

        Handles three corruption patterns seen in sub-agent output:
        1. Normal case: "- Title, year, Author [text](url)" — strip bullet,
           emit as numbered entry.
        2. Char-per-line corruption: the LLM occasionally emits every character
           on its own line (including inside URLs). Detect this (avg line length
           < 2 chars) and collapse all whitespace, then re-insert spaces only at
           meaningful positions (after commas, around brackets).
        3. Bare-link format: "[url]" instead of "[text](url)" — detect URLs
           inside brackets and convert to proper markdown links.
        """
        if not sources:
            return ""

        formatted = []
        for idx, ref in enumerate(sources, 1):
            ref_str = str(ref).strip()
            # Strip bullet prefixes ("- ", "* ")
            ref_str = re.sub(r'^[-*]\s+', '', ref_str)

            # Detect char-per-line corruption: if the average non-empty line
            # length is < 2 characters, the source was emitted one char per line.
            non_empty_lines = [l for l in ref_str.split('\n') if l.strip()]
            if len(non_empty_lines) > 3:
                avg_len = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
                if avg_len < 2:
                    # Collapse ALL whitespace (per-char newlines are noise),
                    # then re-insert spaces at meaningful positions.
                    ref_str = re.sub(r'\s+', '', ref_str)
                    ref_str = ref_str.replace(',', ', ')
                    ref_str = ref_str.replace('[', ' [')
                    ref_str = ref_str.replace(']', '] ')
                    ref_str = re.sub(r'\s+', ' ', ref_str).strip()

            # Fix bare-link format: [url] -> [url](url)
            def _fix_link(m):
                url = m.group(1)
                if re.match(r'https?://|www\.', url):
                    return f'[{url}]({url})'
                return m.group(0)
            ref_str = re.sub(r'\[([^\]]+)\]', _fix_link, ref_str)

            if ref_str:
                formatted.append(f"{idx}. {ref_str}")

        return '\n'.join(formatted)

    @staticmethod
    def _dedup_consecutive_headings(text: str) -> str:
        """Remove consecutive duplicate markdown headings (same text, any level).

        Handles two patterns:
        1. Exact duplicate: "## 引言\\n## 引言" → "## 引言"
        2. Level variant: "## 引言\\n### 引言" or "## 引言\\n### 1. 引言"
           (LLM repeats the section title as a sub-heading)
        Only triggers when the heading text matches and the two headings are
        adjacent (optionally separated by a single blank line).
        """
        lines = text.split('\n')
        out: list[str] = []
        prev_heading_text: str | None = None  # text part only (e.g. "引言")
        prev_was_heading = False

        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
            if m:
                heading_text = m.group(2).strip()
                # Strip leading numbering like "1. " or "一、" for comparison
                clean_text = re.sub(r'^[\d一二三四五六七八九十]+[.、]\s*', '', heading_text)
                # Also strip markdown bold markers
                clean_text = clean_text.replace('**', '').strip()

                if prev_was_heading and prev_heading_text is not None:
                    # Check if this heading duplicates the previous one (text match)
                    if clean_text == prev_heading_text:
                        continue  # skip this duplicate heading

                out.append(line)
                prev_heading_text = clean_text
                prev_was_heading = True
            else:
                # Non-heading line: reset tracking unless it's a blank line
                # (blank line between two headings still counts as "consecutive")
                if line.strip() != '':
                    prev_heading_text = None
                    prev_was_heading = False
                # else: keep prev_heading_text for blank-line-separated headings
                out.append(line)
        return '\n'.join(out)

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
