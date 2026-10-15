#!/usr/bin/env python3
"""MaintainIQ Markdown-to-HTML Report Renderer.

Converts reports/final_report.md into a high-fidelity academic HTML report
(reports/final_report.html) styled for clean A4 printing and PDF export
via Google Chrome on macOS.

Features:
- Pure Python implementation (standard library only, no external dependencies).
- Exact content preservation: headings, tables, links, lists, code blocks, references.
- Proper LaTeX typesetting via MathJax 3 (SVG renderer).
- Academic print CSS with A4 page rules, typography, wide-table handling, and page-break logic.
- Table of Contents anchor compatibility.
"""

from __future__ import annotations

import html
from pathlib import Path
import re
import sys


def generate_slug(text: str) -> str:
    """Generate a clean URL/anchor slug from a heading text."""
    # Strip markdown formatting
    clean = re.sub(r'[*_`]', '', text)
    # Lowercase and keep alphanumeric and spaces/hyphens
    slug = re.sub(r'[^a-zA-Z0-9\s-]', '', clean.lower())
    slug = re.sub(r'\s+', '-', slug).strip('-')
    return slug


def format_inline_sub(text: str) -> str:
    """Format bold and italic markdown tokens."""
    # Bold: **text**
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text* (ensure not adjacent to another asterisk)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    return text


def format_inline(text: str) -> str:
    """Format inline markdown: math, code, links, bold, italic, line breaks."""
    # 1. Protect block math if present in inline flow
    math_tokens: list[str] = []

    def save_block_math(m: re.Match[str]) -> str:
        idx = len(math_tokens)
        math_tokens.append(f"$${m.group(1)}$$")
        return f"__MATH_BLOCK_{idx}__"

    text = re.sub(r'\$\$(.+?)\$\$', save_block_math, text, flags=re.DOTALL)

    # 2. Protect inline math: $...$
    def save_inline_math(m: re.Match[str]) -> str:
        idx = len(math_tokens)
        math_tokens.append(f"${m.group(1)}$")
        return f"__MATH_INL_{idx}__"

    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', save_inline_math, text)

    # 3. Protect inline code: `...`
    code_tokens: list[str] = []

    def save_code(m: re.Match[str]) -> str:
        idx = len(code_tokens)
        escaped_code = html.escape(m.group(1))
        code_tokens.append(f'<code class="inline-code">{escaped_code}</code>')
        return f"__CODE_{idx}__"

    text = re.sub(r'`([^`]+)`', save_code, text)

    # 4. Links: [text](url)
    def repl_link(m: re.Match[str]) -> str:
        link_text = format_inline_sub(m.group(1))
        url = html.escape(m.group(2))
        if url.startswith('http'):
            return f'<a href="{url}" class="external-link" target="_blank" rel="noopener noreferrer">{link_text}</a>'
        return f'<a href="{url}" class="internal-link">{link_text}</a>'

    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', repl_link, text)

    # 5. Bold & Italic
    text = format_inline_sub(text)

    # 6. Hard line breaks (two trailing spaces before newline)
    text = re.sub(r'  \n', '<br />\n', text)

    # 7. Restore inline code
    for idx, c in enumerate(code_tokens):
        text = text.replace(f"__CODE_{idx}__", c)

    # 8. Restore math
    for idx, m_str in enumerate(math_tokens):
        if m_str.startswith('$$'):
            clean_math = m_str[2:-2].strip()
            text = text.replace(
                f"__MATH_BLOCK_{idx}__",
                f'<div class="math-display">$${clean_math}$$</div>',
            )
        else:
            clean_math = m_str[1:-1].strip()
            text = text.replace(
                f"__MATH_INL_{idx}__",
                f'<span class="math-inline">${clean_math}$</span>',
            )

    return text


def parse_list_block(lines: list[str]) -> str:
    """Parse hierarchical list lines into nested HTML <ul>/<ol>."""
    items: list[dict] = []
    curr_item: dict | None = None

    for line in lines:
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if not stripped:
            continue

        m_ol = re.match(r'^(\d+)\.\s+(.*)', stripped)
        m_ul = re.match(r'^[-*]\s+(.*)', stripped)

        if m_ol or m_ul:
            list_type = 'ol' if m_ol else 'ul'
            content = m_ol.group(2) if m_ol else m_ul.group(1)
            num = m_ol.group(1) if m_ol else None
            curr_item = {
                'indent': indent,
                'type': list_type,
                'content': content,
                'num': num,
                'children': [],
            }
            items.append(curr_item)
        else:
            if curr_item is not None:
                curr_item['children'].append(stripped)
            else:
                items.append({
                    'indent': indent,
                    'type': 'p',
                    'content': stripped,
                    'children': [],
                })

    def get_level(indent: int) -> int:
        if indent <= 1:
            return 0
        elif indent <= 4:
            return 1
        return 2

    res: list[str] = []
    tag_stack: list[tuple[str, int]] = []

    for item in items:
        lvl = get_level(item['indent'])
        tag = item['type']

        while tag_stack and tag_stack[-1][1] > lvl:
            t, _ = tag_stack.pop()
            res.append(f'</li></{t}>')

        if not tag_stack or tag_stack[-1][1] < lvl:
            res.append(f'<{tag} class="report-list">')
            tag_stack.append((tag, lvl))
            res.append('<li>')
        elif tag_stack[-1][1] == lvl:
            if tag_stack[-1][0] != tag:
                t, _ = tag_stack.pop()
                res.append(f'</li></{t}>')
                res.append(f'<{tag} class="report-list">')
                tag_stack.append((tag, lvl))
                res.append('<li>')
            else:
                res.append('</li><li>')

        res.append(format_inline(item['content']))

        for child in item['children']:
            if child.startswith('$$') and child.endswith('$$'):
                math_c = child[2:-2].strip()
                res.append(f'<div class="math-display">$${math_c}$$</div>')
            else:
                res.append(f'<p class="list-continuation">{format_inline(child)}</p>')

    while tag_stack:
        t, _ = tag_stack.pop()
        res.append(f'</li></{t}>')

    return '\n'.join(res)


def render_markdown_to_html(md_text: str) -> str:
    """Parse Markdown text into structured semantic HTML."""
    lines = md_text.splitlines()
    html_blocks: list[str] = []

    i = 0
    n = len(lines)

    in_toc = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Blank line
        if not stripped:
            i += 1
            continue

        # Fenced code block
        if stripped.startswith('```'):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # Skip closing ```
            escaped_code = html.escape('\n'.join(code_lines))
            cls = f"language-{lang}" if lang else "ascii-art"
            container_cls = "code-block-container" if lang else "code-block-container ascii-container"
            html_blocks.append(
                f'<div class="{container_cls}"><pre><code class="{cls}">{escaped_code}</code></pre></div>'
            )
            continue

        # Block math $$...$$
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            math_content = stripped[2:-2].strip()
            html_blocks.append(f'<div class="math-display">$${math_content}$$</div>')
            i += 1
            continue

        # Heading (# to ######)
        m_head = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if m_head:
            level = len(m_head.group(1))
            raw_title = m_head.group(2).strip()

            slug = generate_slug(raw_title)
            formatted_title = format_inline(raw_title)

            # Special TOC anchor aliases to ensure perfect navigation
            extra_anchors: list[str] = []
            if "literature review" in raw_title.lower() and level == 2:
                # Add TOC anchor with double hyphen for ampersand
                extra_anchors.append('<span id="3-literature-review--theoretical-framework" class="toc-anchor-target"></span>')
            elif "streamlit" in raw_title.lower() and level == 2:
                extra_anchors.append('<span id="9-streamlit-dashboard-architecture-deployment-and-user-workflow" class="toc-anchor-target"></span>')

            anchors_html = ''.join(extra_anchors)

            if level == 1:
                html_blocks.append(f'<header class="report-header"><h1 id="{slug}" class="report-title">{anchors_html}{formatted_title}</h1></header>')
            elif raw_title.lower() == "table of contents":
                in_toc = True
                html_blocks.append(f'<nav class="toc-wrapper" aria-label="Table of Contents"><h2 id="{slug}" class="toc-heading">{anchors_html}{formatted_title}</h2>')
            else:
                html_blocks.append(f'<h{level} id="{slug}" class="academic-h{level}">{anchors_html}{formatted_title}</h{level}>')

            i += 1
            continue

        # Horizontal rule
        if stripped in ['---', '***', '___']:
            if in_toc:
                in_toc = False
                html_blocks.append('</nav><hr class="section-divider post-toc-divider" />')
            else:
                html_blocks.append('<hr class="section-divider" />')
            i += 1
            continue

        # Blockquote
        if stripped.startswith('>'):
            quote_lines = []
            while i < n and (lines[i].strip().startswith('>') or (lines[i].strip() and not lines[i].strip().startswith('#'))):
                if lines[i].strip().startswith('>'):
                    quote_lines.append(lines[i].strip()[1:].strip())
                elif lines[i].strip():
                    quote_lines.append(lines[i].strip())
                else:
                    break
                i += 1
            quote_text = ' '.join(quote_lines)
            html_blocks.append(f'<blockquote class="academic-blockquote"><p>{format_inline(quote_text)}</p></blockquote>')
            continue

        # Markdown tables
        if stripped.startswith('|') and '|' in stripped[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                header_cells = [c.strip() for c in table_lines[0].split('|')[1:-1]]
                alignments = []
                for c in table_lines[1].split('|')[1:-1]:
                    c = c.strip()
                    if c.startswith(':') and c.endswith(':'):
                        alignments.append('center')
                    elif c.endswith(':'):
                        alignments.append('right')
                    else:
                        alignments.append('left')
                while len(alignments) < len(header_cells):
                    alignments.append('left')

                is_wide = len(header_cells) >= 8
                table_cls = "academic-table table-wide" if is_wide else "academic-table"

                rows_html = []
                for row_line in table_lines[2:]:
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    tds = []
                    for idx, cell in enumerate(cells):
                        align = alignments[idx] if idx < len(alignments) else 'left'
                        tds.append(f'<td style="text-align: {align};">{format_inline(cell)}</td>')
                    rows_html.append('<tr>' + ''.join(tds) + '</tr>')

                ths = [
                    f'<th style="text-align: {alignments[idx]};">{format_inline(c)}</th>'
                    for idx, c in enumerate(header_cells)
                ]
                table_html = (
                    f'<div class="table-wrapper">'
                    f'<table class="{table_cls}">'
                    f'<thead><tr>{"".join(ths)}</tr></thead>'
                    f'<tbody>{"".join(rows_html)}</tbody>'
                    f'</table></div>'
                )
                html_blocks.append(table_html)
                continue

        # Formal references [1], [2], ... [9]
        m_ref = re.match(r'^\[(\d+)\]\s+(.*)', stripped)
        if m_ref:
            ref_num = m_ref.group(1)
            ref_body = m_ref.group(2)
            html_blocks.append(
                f'<div class="reference-item" id="ref-{ref_num}">'
                f'<span class="ref-num">[{ref_num}]</span> '
                f'<span class="ref-body">{format_inline(ref_body)}</span></div>'
            )
            i += 1
            continue

        # Lists (ordered or unordered)
        m_ol = re.match(r'^(\d+)\.\s+(.*)', stripped)
        m_ul = re.match(r'^[-*]\s+(.*)', stripped)
        if m_ol or m_ul:
            list_lines = []
            while i < n:
                curr_l = lines[i]
                curr_s = curr_l.strip()
                if not curr_s:
                    if i + 1 < n and (
                        lines[i + 1].startswith(' ')
                        or re.match(r'^(\d+\.|[-*])\s+', lines[i + 1].strip())
                    ):
                        i += 1
                        continue
                    break
                if curr_s.startswith(('#', '---', '***', '|', '>')):
                    break
                if not curr_l.startswith(' ') and not re.match(r'^(\d+\.|[-*])\s+', curr_s):
                    break
                list_lines.append(curr_l)
                i += 1

            html_blocks.append(parse_list_block(list_lines))
            continue

        # Paragraph
        para_lines = []
        while i < n and lines[i].strip() and not lines[i].strip().startswith(('#', '```', '|', '>', '---', '***')):
            curr_s = lines[i].strip()
            if curr_s.startswith('$$') and curr_s.endswith('$$'):
                break
            if re.match(r'^(\d+\.|[-*]|\[\d+\])\s+', curr_s):
                break
            para_lines.append(lines[i])
            i += 1

        para_text = '\n'.join(para_lines)
        if para_text.strip().startswith('**Table ') and len(para_lines) == 1:
            html_blocks.append(f'<p class="table-caption">{format_inline(para_text.strip())}</p>')
        elif para_text.strip().startswith('*(') and para_text.strip().endswith(')*.'):
            html_blocks.append(f'<p class="table-note">{format_inline(para_text.strip())}</p>')
        elif i <= 8 and ('Course:' in para_text or 'Academic Term:' in para_text):
            html_blocks.append(f'<div class="report-meta-block"><p>{format_inline(para_text)}</p></div>')
        else:
            html_blocks.append(f'<p class="body-para">{format_inline(para_text)}</p>')

    return '\n\n'.join(html_blocks)


def build_full_html(body_html: str) -> str:
    """Wrap body HTML into a complete, standalone academic document."""
    css = """
    /* Academic Print & Screen Stylesheet for MaintainIQ */
    :root {
      --primary-color: #0f172a;
      --text-color: #1e293b;
      --text-muted: #475569;
      --link-color: #0369a1;
      --border-color: #cbd5e1;
      --bg-light: #f8fafc;
      --table-header: #f1f5f9;
      --accent: #2563eb;
    }

    @page {
      size: A4 portrait;
      margin: 20mm 18mm 20mm 18mm;
      @bottom-right {
        content: counter(page);
        font-size: 8.5pt;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #64748b;
      }
    }

    * {
      box-sizing: border-box;
    }

    html {
      font-size: 14.5px;
      line-height: 1.58;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--text-color);
      background-color: #e2e8f0;
      margin: 0;
      padding: 0;
    }

    .document-container {
      max-width: 900px;
      margin: 24px auto;
      background: #ffffff;
      padding: 48px 56px;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
      border-radius: 6px;
    }

    /* Screen-only Print Bar */
    .screen-control-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #0f172a;
      color: #f8fafc;
      padding: 10px 20px;
      font-size: 0.85rem;
      border-radius: 6px;
      margin-bottom: 24px;
    }

    .screen-control-bar button {
      background: #2563eb;
      color: #ffffff;
      border: none;
      padding: 6px 16px;
      border-radius: 4px;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.85rem;
      transition: background 0.15s ease;
    }

    .screen-control-bar button:hover {
      background: #1d4ed8;
    }

    /* Headings */
    .report-title {
      font-size: 1.95rem;
      font-weight: 750;
      line-height: 1.25;
      color: var(--primary-color);
      margin-top: 0;
      margin-bottom: 12px;
      letter-spacing: -0.02em;
    }

    .academic-h2 {
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--primary-color);
      margin-top: 28px;
      margin-bottom: 10px;
      padding-bottom: 5px;
      border-bottom: 1.5px solid var(--border-color);
      break-after: avoid;
      page-break-after: avoid;
    }

    .academic-h3 {
      font-size: 1.12rem;
      font-weight: 650;
      color: var(--primary-color);
      margin-top: 20px;
      margin-bottom: 8px;
      break-after: avoid;
      page-break-after: avoid;
    }

    /* Metadata Block */
    .report-meta-block {
      background-color: var(--bg-light);
      border-left: 4px solid var(--accent);
      padding: 10px 16px;
      margin: 12px 0 20px 0;
      font-size: 0.92rem;
      color: var(--text-muted);
      border-radius: 0 4px 4px 0;
    }

    .report-meta-block p {
      margin: 0;
      line-height: 1.6;
    }

    /* Table of Contents */
    .toc-wrapper {
      background: #f8fafc;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 16px 24px;
      margin: 20px 0;
    }

    .toc-heading {
      font-size: 1.2rem;
      font-weight: 700;
      margin-top: 0;
      margin-bottom: 10px;
      color: var(--primary-color);
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 4px;
    }

    .toc-wrapper ol {
      margin: 0;
      padding-left: 24px;
    }

    .toc-wrapper li {
      margin: 4px 0;
      font-size: 0.92rem;
    }

    .toc-anchor-target {
      display: inline-block;
      position: relative;
      top: -10px;
      visibility: hidden;
    }

    /* Paragraphs and Lists */
    .body-para {
      margin: 10px 0;
      text-align: justify;
      hyphens: auto;
    }

    .report-list {
      margin: 8px 0;
      padding-left: 24px;
    }

    .report-list li {
      margin: 4px 0;
    }

    .list-continuation {
      margin: 4px 0 6px 0;
    }

    /* Links */
    a {
      color: var(--link-color);
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    /* Blockquotes */
    .academic-blockquote {
      margin: 16px 0;
      padding: 10px 18px;
      background-color: var(--bg-light);
      border-left: 3.5px solid var(--accent);
      font-style: italic;
      color: var(--text-muted);
      break-inside: avoid;
      page-break-inside: avoid;
    }

    .academic-blockquote p {
      margin: 0;
    }

    /* Tables */
    .table-wrapper {
      width: 100%;
      overflow-x: auto;
      margin: 16px 0 10px 0;
      break-inside: avoid;
      page-break-inside: avoid;
    }

    .academic-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      line-height: 1.4;
      font-variant-numeric: tabular-nums;
    }

    .academic-table th {
      background-color: var(--table-header);
      color: var(--primary-color);
      font-weight: 650;
      padding: 7px 8px;
      border-top: 1.5pt solid var(--primary-color);
      border-bottom: 1pt solid var(--primary-color);
    }

    .academic-table td {
      padding: 6px 8px;
      border-bottom: 0.5pt solid #e2e8f0;
      vertical-align: top;
    }

    .academic-table tr:last-child td {
      border-bottom: 1.5pt solid var(--primary-color);
    }

    .academic-table tr:nth-child(even) td {
      background-color: #fafbfd;
    }

    /* Wide table adjustment */
    .table-wide {
      font-size: 0.78rem;
      letter-spacing: -0.01em;
    }

    .table-wide th, .table-wide td {
      padding: 4.5pt 4pt;
    }

    .table-caption {
      font-weight: 650;
      margin: 14px 0 4px 0;
      color: var(--primary-color);
      font-size: 0.92rem;
      break-after: avoid;
      page-break-after: avoid;
    }

    .table-note {
      font-size: 0.82rem;
      color: var(--text-muted);
      margin: 4px 0 14px 0;
      font-style: italic;
    }

    /* Code Blocks & ASCII Art */
    .code-block-container {
      margin: 14px 0;
      break-inside: avoid;
      page-break-inside: avoid;
    }

    pre {
      background-color: var(--bg-light);
      border: 1px solid var(--border-color);
      border-radius: 4px;
      padding: 9pt 11pt;
      font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.80rem;
      line-height: 1.32;
      overflow-x: auto;
      white-space: pre;
      margin: 0;
    }

    code.inline-code {
      font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.86em;
      background-color: #f1f5f9;
      color: #0f172a;
      padding: 1.5px 4.5px;
      border-radius: 3px;
      border: 0.5px solid #e2e8f0;
    }

    .ascii-container pre {
      font-size: 0.74rem;
      line-height: 1.25;
      background-color: #fafbfc;
      border-color: #e2e8f0;
    }

    /* LaTeX Math */
    .math-display {
      margin: 12pt 0;
      text-align: center;
      break-inside: avoid;
      page-break-inside: avoid;
    }

    .math-inline {
      display: inline-block;
      vertical-align: middle;
    }

    /* References */
    .reference-item {
      padding-left: 2.2em;
      text-indent: -2.2em;
      margin-bottom: 8px;
      font-size: 0.88rem;
      line-height: 1.45;
      word-break: break-word;
    }

    .ref-num {
      font-weight: 600;
      color: var(--primary-color);
    }

    hr.section-divider {
      border: none;
      border-top: 1px solid var(--border-color);
      margin: 24px 0;
    }

    /* PRINT RULES FOR A4 */
    @media print {
      body {
        background-color: #ffffff;
      }

      .document-container {
        max-width: 100%;
        margin: 0;
        padding: 0;
        box-shadow: none;
        border-radius: 0;
      }

      .screen-control-bar {
        display: none !important;
      }

      .toc-wrapper {
        page-break-after: always;
        break-after: always;
      }

      .post-toc-divider {
        display: none;
      }

      a {
        color: var(--primary-color);
        text-decoration: none;
      }

      .table-wide {
        font-size: 7.4pt;
        letter-spacing: -0.015em;
      }

      .table-wide th, .table-wide td {
        padding: 3.5pt 3pt;
      }

      pre {
        font-size: 7.4pt;
        padding: 6pt 8pt;
      }

      .ascii-container pre {
        font-size: 6.8pt;
        line-height: 1.2;
      }

      .academic-h2, .academic-h3, .table-caption {
        break-after: avoid;
        page-break-after: avoid;
      }

      .table-wrapper, pre, .academic-blockquote, .math-display, .reference-item {
        break-inside: avoid;
        page-break-inside: avoid;
      }
    }
    """

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MaintainIQ: Academic Research & Technical Implementation Report</title>
  
  <!--
    NOTE ON EQUATION RENDERING:
    Inline and block LaTeX equations are rendered via MathJax 3 (SVG mode).
    An active internet connection is required during page load / PDF export
    to fetch MathJax from cdn.jsdelivr.net.
  -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true,
        processEnvironments: true
      }},
      svg: {{
        fontCache: 'global'
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  
  <style>
{css}
  </style>
</head>
<body>
  <div class="document-container">
    <div class="screen-control-bar no-print">
      <span>MaintainIQ Academic Report &bull; Ready for Chrome PDF Export (A4)</span>
      <button onclick="window.print()">Print to PDF via Chrome</button>
    </div>

{body_html}

  </div>
</body>
</html>
"""
    return doc


def main() -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    report_md_path = workspace_root / "reports" / "final_report.md"
    report_html_path = workspace_root / "reports" / "final_report.html"

    if not report_md_path.exists():
        print(f"Error: {report_md_path} not found.", file=sys.stderr)
        return 1

    print(f"Reading Markdown source from: {report_md_path}")
    md_content = report_md_path.read_text(encoding="utf-8")

    print(f"Parsing {len(md_content)} characters of Markdown...")
    body_html = render_markdown_to_html(md_content)

    print("Assembling standalone HTML document with academic print CSS & MathJax...")
    full_html = build_full_html(body_html)

    print(f"Saving generated HTML to: {report_html_path}")
    report_html_path.write_text(full_html, encoding="utf-8")

    print("Success: Report rendered successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
