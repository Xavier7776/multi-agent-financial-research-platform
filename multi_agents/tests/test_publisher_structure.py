"""Demo test: verify publisher.generate_layout handles all known problem patterns.

This test does NOT call any LLM or external API — it constructs synthetic
research_state inputs that simulate the problematic patterns we've seen in
real outputs, and verifies that generate_layout produces clean structure.

Run: python -m pytest multi_agents/tests/test_publisher_structure.py -v
Or:  python multi_agents/tests/test_publisher_structure.py
"""
import sys
import os
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agents.agents.publisher import PublisherAgent


def make_publisher():
    return PublisherAgent(output_dir="/tmp/test")


def count_headings(text, level=None):
    """Count headings at a given level (or all if level=None)."""
    if level:
        pattern = rf'^#{{{level}}}\s+'
    else:
        pattern = r'^#{1,6}\s+'
    return sum(1 for line in text.split('\n') if re.match(pattern, line.strip()))


def find_consecutive_duplicates(text):
    """Find any consecutive duplicate headings (same text, any level)."""
    lines = text.split('\n')
    duplicates = []
    prev_text = None
    prev_was_heading = False
    for i, line in enumerate(lines):
        m = re.match(r'^#{1,6}\s+(.+?)\s*$', line.strip())
        if m:
            clean = re.sub(r'^[\d一二三四五六七八九十]+[.、]\s*', '', m.group(1).replace('**', '').strip())
            if prev_was_heading and clean == prev_text:
                duplicates.append((i + 1, line.strip()))
            prev_text = clean
            prev_was_heading = True
        elif line.strip() == '':
            pass  # keep prev_text
        else:
            prev_text = None
            prev_was_heading = False
    return duplicates


# ============================================================
# Test 1: Normal case — intro/conclusion without H1/H2 prefix
# ============================================================
def test_normal_case():
    p = make_publisher()
    state = {
        "research_data": [
            {"section1": "## 章节一\n内容1"},
            {"section2": "## 章节二\n内容2"},
        ],
        "introduction": "这是引言内容。",
        "conclusion": "这是结论内容。",
        "table_of_contents": "- 章节1\n- 章节2",
        "sources": ["[ref1](http://example.com)"],
        "headers": {
            "title": "测试报告",
            "date": "日期",
            "introduction": "引言",
            "table_of_contents": "目录",
            "conclusion": "结论",
            "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # Should have exactly 1 H1
    assert count_headings(layout, 1) == 1, f"Expected 1 H1, got {count_headings(layout, 1)}"

    # Should have exactly 1 引言 H2
    intro_h2 = [l for l in layout.split('\n') if l.strip() == '## 引言']
    assert len(intro_h2) == 1, f"Expected 1 '## 引言', got {len(intro_h2)}"

    # Should have exactly 1 结论 H2
    conc_h2 = [l for l in layout.split('\n') if l.strip() == '## 结论']
    assert len(conc_h2) == 1, f"Expected 1 '## 结论', got {len(conc_h2)}"

    # No H4
    assert count_headings(layout, 4) == 0, f"Expected 0 H4, got {count_headings(layout, 4)}"

    # No consecutive duplicates
    dups = find_consecutive_duplicates(layout)
    assert len(dups) == 0, f"Found consecutive duplicate headings: {dups}"

    print("  ✓ test_normal_case passed")


# ============================================================
# Test 2: Intro starts with "## 引言" — should be stripped
# ============================================================
def test_intro_with_h2_prefix():
    p = make_publisher()
    state = {
        "research_data": [],
        "introduction": "## 引言\n这是引言内容。",
        "conclusion": "## 结论\n这是结论内容。",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # Should NOT have duplicate ## 引言
    intro_h2 = [l for l in layout.split('\n') if l.strip() == '## 引言']
    assert len(intro_h2) == 1, f"Expected 1 '## 引言', got {len(intro_h2)}"

    conc_h2 = [l for l in layout.split('\n') if l.strip() == '## 结论']
    assert len(conc_h2) == 1, f"Expected 1 '## 结论', got {len(conc_h2)}"

    dups = find_consecutive_duplicates(layout)
    assert len(dups) == 0, f"Found consecutive duplicates: {dups}"

    print("  ✓ test_intro_with_h2_prefix passed")


# ============================================================
# Test 3: Intro starts with "# 引言" (H1) — should be stripped
# ============================================================
def test_intro_with_h1_prefix():
    p = make_publisher()
    state = {
        "research_data": [],
        "introduction": "# 引言\n这是引言内容。",
        "conclusion": "# 结论\n这是结论内容。",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    intro_h2 = [l for l in layout.split('\n') if l.strip() == '## 引言']
    assert len(intro_h2) == 1, f"Expected 1 '## 引言', got {len(intro_h2)}"

    print("  ✓ test_intro_with_h1_prefix passed")


# ============================================================
# Test 4: Section title H2 is kept as H2 (not demoted to H3)
# This is the KEY fix — old code demoted ALL ## to ###, flattening hierarchy
# ============================================================
def test_section_title_kept_as_h2():
    p = make_publisher()
    state = {
        "research_data": [
            {"section1": "## 子节标题\n内容"},
        ],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # The section title "## 子节标题" should be KEPT as H2 (section title)
    h2_lines = [l for l in layout.split('\n') if l.strip() == '## 子节标题']
    h3_lines = [l for l in layout.split('\n') if l.strip() == '### 子节标题']
    assert len(h2_lines) == 1, f"Section title should be H2 (found: {h2_lines})"
    assert len(h3_lines) == 0, f"Section title should NOT be demoted to H3 (found: {h3_lines})"

    print("  ✓ test_section_title_kept_as_h2 passed")


# ============================================================
# Test 4b: Subsequent H2 within a section IS demoted to H3
# ============================================================
def test_subsequent_h2_demoted():
    p = make_publisher()
    state = {
        "research_data": [
            {"section1": "## 章节标题\n内容\n## 子节\n更多内容"},
        ],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # First H2 kept as H2
    h2_section = [l for l in layout.split('\n') if l.strip() == '## 章节标题']
    assert len(h2_section) == 1, f"First H2 should be kept as H2"

    # Second H2 demoted to H3
    h3_sub = [l for l in layout.split('\n') if l.strip() == '### 子节']
    assert len(h3_sub) == 1, f"Second H2 should be demoted to H3"

    print("  ✓ test_subsequent_h2_demoted passed")


# ============================================================
# Test 5: Per-section intro/conclusion/references stripped
# ============================================================
def test_per_section_strip():
    p = make_publisher()
    state = {
        "research_data": [
            {"section1": "## 子节\n内容\n## 引言\n不应出现\n## 结论\n不应出现\n## 参考文献\n不应出现"},
        ],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # The per-section 引言/结论/参考文献 should be stripped
    # (only the template-level ones should remain)
    intro_count = sum(1 for l in layout.split('\n') if '引言' in l and l.strip().startswith('#'))
    conc_count = sum(1 for l in layout.split('\n') if '结论' in l and l.strip().startswith('#'))

    assert intro_count == 1, f"Expected 1 heading with '引言', got {intro_count}"
    assert conc_count == 1, f"Expected 1 heading with '结论', got {conc_count}"

    print("  ✓ test_per_section_strip passed")


# ============================================================
# Test 6: H3 duplicate of H2 (### 引言 right after ## 引言)
# This is the key new test — verifies the enhanced _dedup_consecutive_headings
# ============================================================
def test_h3_duplicate_of_h2():
    p = make_publisher()
    # Simulate: LLM generates intro content that starts with "### 1. 引言"
    # after the template already emitted "## 引言"
    state = {
        "research_data": [],
        "introduction": "### 1. 引言\n这是引言内容。",
        "conclusion": "### 8. 结论\n这是结论内容。",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # The "### 1. 引言" should be removed by dedup (text "引言" matches "## 引言")
    h3_intro = [l for l in layout.split('\n') if '引言' in l and l.strip().startswith('###')]
    assert len(h3_intro) == 0, f"Expected 0 H3 with '引言', got {len(h3_intro)}: {h3_intro}"

    h3_conc = [l for l in layout.split('\n') if '结论' in l and l.strip().startswith('###')]
    assert len(h3_conc) == 0, f"Expected 0 H3 with '结论', got {len(h3_conc)}: {h3_conc}"

    # Should still have exactly 1 H2 引言 and 1 H2 结论
    intro_h2 = [l for l in layout.split('\n') if l.strip() == '## 引言']
    assert len(intro_h2) == 1, f"Expected 1 '## 引言', got {len(intro_h2)}"

    conc_h2 = [l for l in layout.split('\n') if l.strip() == '## 结论']
    assert len(conc_h2) == 1, f"Expected 1 '## 结论', got {len(conc_h2)}"

    dups = find_consecutive_duplicates(layout)
    assert len(dups) == 0, f"Found consecutive duplicates: {dups}"

    print("  ✓ test_h3_duplicate_of_h2 passed")


# ============================================================
# Test 7: No H4 headings (date should be bold text, not H4)
# ============================================================
def test_no_h4():
    p = make_publisher()
    state = {
        "research_data": [],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    assert count_headings(layout, 4) == 0, f"Expected 0 H4, got {count_headings(layout, 4)}"
    assert '**日期**' in layout, "Date should be bold text '**日期**'"

    print("  ✓ test_no_h4 passed")


# ============================================================
# Test 8: Financial data adds disclaimer
# ============================================================
def test_financial_disclaimer():
    p = make_publisher()
    state = {
        "research_data": [],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": {"ticker": "AAPL"},
    }
    layout = p.generate_layout(state)

    assert '免责声明' in layout, "Financial disclaimer should be present"
    assert 'AAPL' in layout, "Ticker should appear in disclaimer"

    print("  ✓ test_financial_disclaimer passed")


# ============================================================
# Test 8b: Sources are formatted as a numbered list (1. 2. 3.)
# Sub-agent may emit "- ..." prefixes; they should be stripped
# and replaced with sequential numbering.
# ============================================================
def test_sources_numbered_list():
    p = make_publisher()
    state = {
        "research_data": [],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [
            "- Source A, 2026 [link](http://a.com)",
            "- Source B, 2026 [link](http://b.com)",
            "* Source C, 2026 [link](http://c.com)",
        ],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # Should be numbered 1. 2. 3.
    assert "1. Source A, 2026" in layout, f"Expected '1. Source A', got:\n{layout}"
    assert "2. Source B, 2026" in layout, f"Expected '2. Source B', got:\n{layout}"
    assert "3. Source C, 2026" in layout, f"Expected '3. Source C', got:\n{layout}"

    # Should NOT contain bullet prefixes
    assert "- Source A" not in layout, "Bullet prefix '- ' should be stripped"
    assert "- Source B" not in layout, "Bullet prefix '- ' should be stripped"
    assert "* Source C" not in layout, "Bullet prefix '* ' should be stripped"

    print("  ✓ test_sources_numbered_list passed")


# ============================================================
# Test 8c: Char-per-line corruption in sources
# The LLM occasionally emits every character on its own line.
# The publisher should detect and collapse this.
# ============================================================
def test_sources_char_per_line_corruption():
    p = make_publisher()
    # Simulate char-per-line corruption (as seen in real output)
    corrupted_ref = "- \n宁\n德\n时\n代\n,\n \n2\n0\n2\n6\n,\n \n作\n者\n \n[\nh\nt\nt\np\ns\n:\n/\n/\nx\n.\nc\no\nm\n]"
    state = {
        "research_data": [],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [corrupted_ref],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # Should be numbered
    assert "1. 宁德时代" in layout, f"Expected '1. 宁德时代', got:\n{layout[-300:]}"

    # Should NOT have per-char newlines in the reference
    ref_section = layout.split("## 参考文献")[-1]
    # The reference line should be a single line (no per-char newlines)
    ref_line = [l for l in ref_section.split('\n') if '宁德时代' in l]
    assert len(ref_line) == 1, f"Expected 1 ref line, got {len(ref_line)}"
    assert '宁德时代, 2026' in ref_line[0], f"Expected collapsed text, got: {ref_line[0]}"

    # Bare-link [https://x.com] should be converted to [https://x.com](https://x.com)
    assert '[https://x.com](https://x.com)' in layout, f"Expected proper markdown link, got:\n{layout[-300:]}"

    print("  ✓ test_sources_char_per_line_corruption passed")


# ============================================================
# Test 9: Full layout simulation — verify heading hierarchy
# ============================================================
def test_full_layout_hierarchy():
    p = make_publisher()
    state = {
        "research_data": [
            {"section1": "## 投资摘要\n内容"},
            {"section2": "## 宏观经济\n内容"},
        ],
        "introduction": "## 引言\n引言内容",
        "conclusion": "## 结论\n结论内容",
        "table_of_contents": "- 投资\n- 宏观",
        "sources": ["[ref](http://x.com)"],
        "headers": {
            "title": "完整测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": {"ticker": "000001"},
    }
    layout = p.generate_layout(state)

    # H1: exactly 1
    assert count_headings(layout, 1) == 1

    # H2: 引言 + 目录 + 投资摘要 + 宏观经济 + 结论 + 参考文献 = 6
    # (section titles are kept as H2, NOT demoted to H3 — this is the key fix)
    h2_count = count_headings(layout, 2)
    assert h2_count == 6, f"Expected 6 H2, got {h2_count}\n{layout}"

    # H3: 0 (no subsections in test data; section titles stay as H2)
    h3_count = count_headings(layout, 3)
    assert h3_count == 0, f"Expected 0 H3, got {h3_count}"

    # No H4
    assert count_headings(layout, 4) == 0

    # No duplicates
    dups = find_consecutive_duplicates(layout)
    assert len(dups) == 0, f"Found duplicates: {dups}"

    print("  ✓ test_full_layout_hierarchy passed")


# ============================================================
# Test 10: List value in research_data (sub-agent returns list)
# ============================================================
def test_list_value_in_research_data():
    """Sub-agent may return a list instead of a string.
    Without _stringify, f"{value}" on a list produces "['item1', 'item2']".
    """
    p = make_publisher()
    state = {
        "research_data": [
            {"section1": ["段落1", "段落2", "段落3"]},  # list value!
        ],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    # Should NOT contain Python list repr
    assert "['" not in layout, f"Python list repr found in layout: {layout[:200]}"
    assert "段落1" in layout, "Content from list should appear in layout"
    assert "段落2" in layout, "Content from list should appear in layout"
    assert "段落3" in layout, "Content from list should appear in layout"

    print("  ✓ test_list_value_in_research_data passed")


# ============================================================
# Test 11: List subheader (not dict)
# ============================================================
def test_list_subheader():
    """research_data may contain list items directly (not wrapped in dict)."""
    p = make_publisher()
    state = {
        "research_data": [
            ["段落A", "段落B"],  # list directly as subheader
        ],
        "introduction": "引言",
        "conclusion": "结论",
        "table_of_contents": "",
        "sources": [],
        "headers": {
            "title": "测试", "date": "日期", "introduction": "引言",
            "table_of_contents": "目录", "conclusion": "结论", "references": "参考文献",
        },
        "date": "2026-01-01",
        "financial_data": None,
    }
    layout = p.generate_layout(state)

    assert "['" not in layout, f"Python list repr found in layout"
    assert "段落A" in layout
    assert "段落B" in layout

    print("  ✓ test_list_subheader passed")


# ============================================================
# Run all tests
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Publisher Structure Tests")
    print("=" * 60)

    tests = [
        test_normal_case,
        test_intro_with_h2_prefix,
        test_intro_with_h1_prefix,
        test_section_title_kept_as_h2,
        test_subsequent_h2_demoted,
        test_per_section_strip,
        test_h3_duplicate_of_h2,
        test_no_h4,
        test_financial_disclaimer,
        test_sources_numbered_list,
        test_sources_char_per_line_corruption,
        test_full_layout_hierarchy,
        test_list_value_in_research_data,
        test_list_subheader,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
