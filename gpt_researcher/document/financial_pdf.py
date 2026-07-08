"""Financial PDF Document Loader for GPT Researcher.

This module provides FinancialPDFLoader, a specialized document loader
for financial reports, earnings statements, and annual filings in PDF format.
It goes beyond raw text extraction by identifying financial statement sections
and extracting structured financial data points.

Architecture:
    FinancialPDFLoader (specialized) → PyMuPDF (fitz) → structured financial text
    DocumentLoader (generic)          → PyMuPDFLoader → raw text

Usage:
    loader = FinancialPDFLoader("path/to/AAPL-10K.pdf")
    docs = await loader.load()
    # Returns [{raw_content: "...", url: "...", financial_type: "income_statement"}, ...]
"""

import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class FinancialPDFLoader:
    """Specialized loader for financial report PDFs.

    Extracts text from PDFs using PyMuPDF and enriches the output with
    financial context markers to help downstream LLMs locate and interpret
    key financial data.

    Features:
    - Page-by-page text extraction with PyMuPDF (fitz)
    - Financial statement type detection (income statement, balance sheet, cash flow)
    - Key metric extraction via regex (revenue, net income, EPS, total assets, etc.)
    - Standard GPT Researcher document format output
    """

    # Financial statement type keywords for section detection
    _STATEMENT_PATTERNS = {
        "income_statement": [
            # English
            "income statement", "statement of operations", "statement of earnings",
            "consolidated statements of operations",
            "revenue", "net income", "gross profit", "operating income",
            "earnings per share", "diluted eps", "basic eps",
            "ebit", "ebitda", "cost of revenue", "cost of sales",
            "r&d", "research and development",
            "selling, general and administrative", "sga",
            "interest expense", "income tax", "tax expense",
            # Chinese
            "利润表", "损益表", "收入表",
            "营业收入", "营业成本", "营业利润", "营业费用",
            "税前利润", "所得税", "所得税费用",
            "净利润", "归属于母公司", "少数股东损益",
            "每股收益", "基本每股收益", "稀释每股收益",
            "研发费用", "销售费用", "管理费用", "财务费用",
            "投资收益", "公允价值变动",
        ],
        "balance_sheet": [
            # English
            "balance sheet", "statement of financial position",
            "consolidated balance sheets",
            "total assets", "total liabilities", "stockholders equity",
            "shareholders' equity",
            "current assets", "current liabilities",
            "non-current assets", "non-current liabilities",
            "intangible assets", "goodwill",
            "retained earnings", "accumulated other comprehensive",
            "property, plant and equipment", "ppe",
            "accounts receivable", "accounts payable",
            "inventory", "inventories",
            "long-term debt", "short-term debt",
            "treasury stock", "common stock", "additional paid-in capital",
            # Chinese
            "资产负债表", "财务状况表",
            "总资产", "总负债", "股东权益", "所有者权益",
            "流动资产", "流动负债", "非流动资产", "非流动负债",
            "无形资产", "商誉",
            "未分配利润", "盈余公积", "资本公积",
            "固定资产", "在建工程",
            "应收账款", "应付账款", "存货",
            "长期借款", "短期借款", "应付债券",
            "归属于母公司所有者权益", "少数股东权益",
        ],
        "cash_flow": [
            # English
            "cash flow", "statement of cash flows",
            "consolidated statements of cash flows",
            "operating activities", "investing activities", "financing activities",
            "free cash flow", "capital expenditure", "capex",
            "depreciation", "amortization", "depreciation and amortization",
            "stock-based compensation", "share-based compensation",
            "working capital", "changes in working capital",
            "net cash provided by", "net cash used in",
            # Chinese
            "现金流量表", "现金流",
            "经营活动", "投资活动", "筹资活动",
            "经营活动产生的现金流量",
            "投资活动产生的现金流量",
            "筹资活动产生的现金流量",
            "资本支出", "资本开支",
            "折旧", "摊销", "折旧和摊销",
            "营运资金", "营运资本变动",
            "自由现金流", "现金及现金等价物",
        ],
        "management_discussion": [
            # English
            "management discussion", "management's discussion",
            "management's discussion and analysis", "md&a",
            "business overview", "results of operations",
            "forward-looking", "outlook", "future",
            "risk factors", "key risk",
            "segment information", "segment results",
            "business strategy", "strategic initiatives",
            "market conditions", "competitive landscape",
            "liquidity and capital resources",
            # Chinese
            "管理层讨论", "管理层分析", "经营情况讨论",
            "经营情况讨论与分析", "董事会报告",
            "主营业务", "经营回顾", "业绩回顾",
            "主营业务分析", "收入构成",
            "展望", "未来展望", "发展战略", "经营计划",
            "风险因素", "风险提示", "风险分析",
            "行业格局", "竞争格局", "市场环境",
            "核心竞争力", "竞争优势",
            "融资情况", "资金需求",
        ],
        "notes": [
            # English
            "notes to financial statements", "notes to consolidated",
            "notes to the consolidated financial statements",
            "accounting policies", "significant accounting",
            "related party", "related parties",
            "contingent liabilities", "contingencies",
            "subsequent events", "commitments",
            "fair value", "fair value measurements",
            "segment reporting", "segment and geographic",
            "share-based", "stock-based",
            "income taxes", "tax reconciliation",
            "earnings per share", "eps computation",
            # Chinese
            "财务报表附注", "报表附注", "附注",
            "公司基本情况", "主要会计政策",
            "会计政策", "会计估计", "会计估计变更",
            "关联方", "关联交易", "关联方关系",
            "或有负债", "或有事项",
            "期后事项", "资产负债表日后事项",
            "承诺事项", "担保事项",
            "公允价值", "金融工具",
            "股份支付", "股权激励",
            "所得税", "递延所得税",
            "每股收益", "分部报告",
        ],
    }

    # Regex patterns for extracting common financial metrics
    _METRIC_PATTERNS = [
        (re.compile(r"(?:total\s+)?revenue[s]?\s*[:：]?\s*\$?([\d,]+\.?\d*)\s*(?:million|billion|M|B|亿|万)?", re.IGNORECASE), "Revenue"),
        (re.compile(r"net\s+income\s*[:：]?\s*\$?([\d,]+\.?\d*)\s*(?:million|billion|M|B|亿|万)?", re.IGNORECASE), "Net Income"),
        (re.compile(r"earnings\s+per\s+share\s*[:：]?\s*\$?([\d.]+)", re.IGNORECASE), "EPS"),
        (re.compile(r"total\s+assets\s*[:：]?\s*\$?([\d,]+\.?\d*)\s*(?:million|billion|M|B|亿|万)?", re.IGNORECASE), "Total Assets"),
        (re.compile(r"total\s+liabilities\s*[:：]?\s*\$?([\d,]+\.?\d*)\s*(?:million|billion|M|B|亿|万)?", re.IGNORECASE), "Total Liabilities"),
        (re.compile(r"(?:gross|operating)\s+margin\s*[:：]?\s*([\d.]+)\s*%?", re.IGNORECASE), "Margin"),
    ]

    def __init__(self, path: str):
        """Initialize the financial PDF loader.

        Args:
            path: Path to a financial PDF file.
        """
        self.path = path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Financial PDF not found: {path}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def load(self) -> List[dict]:
        """Load and parse the financial PDF.

        Returns:
            List of document dicts with keys:
                - raw_content: Extracted page text with financial markers
                - url: Source filename
                - financial_type: Detected statement type (or "general")
                - metrics: Extracted financial metrics dict
            Returns empty list on failure.
        """
        try:
            import fitz  # PyMuPDF

            #获取文件名
            filename = os.path.basename(self.path)
            all_docs = []

            #打开文件
            doc = fitz.open(self.path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                if not text or not text.strip():
                    continue

                # Clean up common PDF extraction artifacts
                text = self._clean_text(text)

                # Detect financial statement type
                fin_type = self._detect_statement_type(text)

                # Extract key financial metrics
                metrics = self._extract_metrics(text)

                # Build enriched content with financial context header
                enriched = self._enrich_content(text, fin_type, metrics, page_num)

                doc_entry = {
                    "raw_content": enriched,
                    "url": f"{filename}#page={page_num + 1}",
                    "financial_type": fin_type,
                }

                # Only include metrics field if we found something
                if metrics:
                    doc_entry["metrics"] = metrics

                all_docs.append(doc_entry)

            doc.close()

            if not all_docs:
                logger.warning(
                    f"[FinancialPDFLoader] No extractable text found in {filename}"
                )
                return []

            logger.info(
                f"[FinancialPDFLoader] Extracted {len(all_docs)} pages "
                f"from {filename}"
                f" (detected types: {self._summarize_types(all_docs)})"
            )
            return all_docs

        except ImportError:
            logger.warning(
                "[FinancialPDFLoader] PyMuPDF (fitz) not installed. "
                "Install with: pip install PyMuPDF"
            )
            return []
        except Exception as e:
            logger.error(f"[FinancialPDFLoader] Failed to load {self.path}: {e}")
            return []

    # ------------------------------------------------------------------
    # Internal: text processing
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text from common PDF artifacts.

        - Remove excessive whitespace
        - Normalize line endings
        - Remove page number artifacts
        """
        # Remove excessive blank lines (keep single blank lines)
        #合并过多的换行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove isolated page numbers at bottom of pages
        #识别并删除独立成行的页码
        text = re.sub(r'\n\d{1,3}\s*\n', '\n', text)

        # Normalize spaces
        #中括号内包含一个普通空格和一个制表符 \t
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    # ------------------------------------------------------------------
    # Internal: financial analysis
    # ------------------------------------------------------------------

    @classmethod
    def _detect_statement_type(cls, text: str) -> str:
        """Detect the financial statement type from page text.

        Uses keyword matching to classify pages as:
        - "income_statement" (利润表)
        - "balance_sheet" (资产负债表)
        - "cash_flow" (现金流量表)
        - "general" (其他财务内容)
        """
        if not text:
            return "general"

        text_lower = text.lower()

        for stmt_type, keywords in cls._STATEMENT_PATTERNS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    return stmt_type

        return "general"

    @classmethod
    def _extract_metrics(cls, text: str) -> dict:
        """Extract key financial metrics using regex patterns.

        Returns:
            Dict mapping metric names to extracted values.
            Empty dict if no metrics found.
        """
        metrics = {}
        # 在不传入任何参数时，text.split()
        # 默认以所有的连续空白字符（包括空格、制表符 \t、换行符 \n、回车符 \r
        # 等）作为切割依据。
        text_combined = " ".join(text.split())

        for pattern, name in cls._METRIC_PATTERNS:
            match = pattern.search(text_combined)
            if match:
                value = match.group(1).replace(",", "")
                try:
                    metrics[name] = float(value)
                except ValueError:
                    metrics[name] = value

        return metrics

    @staticmethod
    def _enrich_content(
        text: str,
        fin_type: str,
        metrics: dict,
        page_num: int,
    ) -> str:
        """Enrich raw text with financial context markers.

        Adds a header block at the top of each page that tells the LLM:
        - What type of financial statement this page contains
        - Key metrics extracted from this page
        - The page number in the original document
        """
        header_lines = [f"[第 {page_num + 1} 页 / 财务文档]"]

        # Statement type label
        type_labels = {
            "income_statement": "利润表",
            "balance_sheet": "资产负债表",
            "cash_flow": "现金流量表",
            "management_discussion": "管理层讨论与分析",
            "notes": "财务报表附注",
            "general": "财务文档",
        }
        header_lines.append(
            f"[报表类型: {type_labels.get(fin_type, fin_type)}]"
        )

        # Extracted metrics summary
        if metrics:
            metric_strs = [
                f"{k}={v}" for k, v in metrics.items()
            ]
            header_lines.append(f"[关键指标: {', '.join(metric_strs)}]")

        header = "\n".join(header_lines) + "\n" + "─" * 50 + "\n\n"
        return header + text

    @staticmethod
    def _summarize_types(docs: List[dict]) -> str:
        """Generate a summary of detected financial statement types."""
        from collections import Counter
        # Counter({"general": 20, "income_statement": 8, "balance_sheet": 7})
        type_counts = Counter(d.get("financial_type", "unknown") for d in docs)
        parts = [f"{t}={c}" for t, c in type_counts.most_common()]
        return ", ".join(parts) if parts else "none"
