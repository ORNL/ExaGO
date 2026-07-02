"""PDF report generator for completed AgentiGrid search sessions.

Uses ReportLab with DejaVu Sans font for diacritics support.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from agentigrid.engine.agent_loop import SearchSession
from agentigrid.parsers.opflow_results import OPFLOWResult

try:
    from charts import (
        convergence_chart, voltage_range_chart, voltage_profile_chart,
        generator_dispatch_chart, line_loading_chart, multi_objective_trend_chart,
        reserve_trajectory_chart,
    )
except ModuleNotFoundError:
    from launcher.charts import (
        convergence_chart, voltage_range_chart, voltage_profile_chart,
        generator_dispatch_chart, line_loading_chart, multi_objective_trend_chart,
        reserve_trajectory_chart,
    )

logger = logging.getLogger("launcher.report_generator")

# ── Font Registration ────────────────────────────────────────────────────────

_FONT_NAME = "DejaVuSans"
_FONT_REGISTERED = False


def _register_fonts():
    """Register DejaVu Sans font if available."""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    _bundled = Path(__file__).parent / "assets" / "fonts"
    font_paths = [
        str(_bundled / "DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    bold_paths = [
        str(_bundled / "DejaVuSans-Bold.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for p in font_paths:
        if Path(p).exists():
            pdfmetrics.registerFont(TTFont("DejaVuSans", p))
            for bp in bold_paths:
                if Path(bp).exists():
                    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bp))
                    break
            _FONT_REGISTERED = True
            logger.info("Registered DejaVu Sans font from %s", p)
            return
    logger.warning("DejaVu Sans not found; using Helvetica fallback")


# ── Chart Export Helper ──────────────────────────────────────────────────────

def _export_chart_image(fig, width_px: int = 800, height_px: int = 400) -> bytes | None:
    """Export a Plotly figure to PNG bytes.

    Returns None if export fails (e.g., kaleido not installed).
    """
    try:
        return fig.to_image(format="png", width=width_px, height=height_px, scale=2)
    except Exception as exc:
        logger.warning("Failed to export chart image: %s", exc)
        return None


# ── Markdown Table Helpers ────────────────────────────────────────────────────

def _is_separator_row(line: str) -> bool:
    """Check if a line is a markdown table separator (e.g., |------|------|)."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    content = stripped.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")
    return len(content) == 0 and "-" in stripped


def _parse_markdown_table(text: str) -> list[list[str]] | None:
    """Parse a markdown table from text into a list of rows.

    Returns None if the text is not a markdown table.
    The separator row (with dashes) is excluded.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return None

    # All lines must contain |
    if not all("|" in l for l in lines):
        return None

    rows = []
    for line in lines:
        if _is_separator_row(line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # Strip empty cells from leading/trailing |
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)

    return rows if len(rows) >= 2 else None


# ── Report Generator ─────────────────────────────────────────────────────────

class ReportGenerator:
    """Generates PDF reports from completed search sessions."""

    def __init__(self):
        _register_fonts()
        self._font = "DejaVuSans" if _FONT_REGISTERED else "Helvetica"
        self._font_bold = "DejaVuSans-Bold" if _FONT_REGISTERED else "Helvetica-Bold"
        self._styles = self._build_styles()

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape XML special characters for ReportLab Paragraphs."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _is_ascii_art(self, text: str) -> bool:
        """Check if text contains ASCII art (box-drawing characters, etc.)."""
        art_chars = set("┤├┬┴┼─│┐┘┌└╭╮╯╰✗✓✕✔▉▊▋▌▍▎▏")
        char_count = sum(1 for c in text if c in art_chars)
        return char_count > 5

    def _preprocess_text(self, text: str) -> str:
        """Preprocess markdown text to normalize paragraph boundaries."""
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                if result and result[-1].strip():
                    result.append("")
                result.append(line)
                result.append("")
            elif stripped == "---":
                if result and result[-1].strip():
                    result.append("")
                result.append("")
            else:
                result.append(line)
        return "\n".join(result)

    def _build_styles(self) -> dict[str, ParagraphStyle]:
        """Create custom paragraph styles."""
        return {
            "title": ParagraphStyle(
                "title", fontName=self._font_bold, fontSize=24,
                alignment=TA_CENTER, spaceAfter=12,
            ),
            "subtitle": ParagraphStyle(
                "subtitle", fontName=self._font, fontSize=14,
                alignment=TA_CENTER, spaceAfter=6, textColor=colors.grey,
            ),
            "heading1": ParagraphStyle(
                "heading1", fontName=self._font_bold, fontSize=18,
                spaceBefore=20, spaceAfter=10,
            ),
            "heading2": ParagraphStyle(
                "heading2", fontName=self._font_bold, fontSize=14,
                spaceBefore=14, spaceAfter=8,
            ),
            "body": ParagraphStyle(
                "body", fontName=self._font, fontSize=10,
                spaceAfter=6, leading=14,
            ),
            "body_small": ParagraphStyle(
                "body_small", fontName=self._font, fontSize=8,
                spaceAfter=4, leading=10,
            ),
            "caption": ParagraphStyle(
                "caption", fontName=self._font, fontSize=9,
                textColor=colors.grey, spaceAfter=4,
            ),
            "bullet": ParagraphStyle(
                "bullet", fontName=self._font, fontSize=10,
                spaceAfter=3, leading=14,
                leftIndent=15, bulletIndent=5,
                bulletFontName=self._font, bulletFontSize=10,
            ),
        }

    def generate(
        self,
        session: SearchSession,
        summary_text: str | None = None,
        base_result: OPFLOWResult | None = None,
        best_result: OPFLOWResult | None = None,
        goal_classification: dict | None = None,
        steering_history: list[dict] | None = None,
    ) -> bytes:
        """Generate a PDF report and return it as bytes.

        Args:
            session: Completed search session.
            summary_text: Optional LLM-generated summary analysis.
            base_result: Base case OPFLOW results (for charts).
            best_result: Best feasible OPFLOW results (for charts).
            goal_classification: Optional dict with goal_type, best_iteration,
                best_iteration_rationale from LLM analysis.

        Returns:
            PDF file contents as bytes.
        """
        gc = goal_classification
        best_iter_override = gc.get("best_iteration") if gc else None
        goal_type = gc.get("goal_type") if gc else None

        sweep_entry = session.journal.get_sweep_entry()
        is_sweep = sweep_entry is not None
        contingency_entry = session.journal.get_contingency_entry()

        v_min = session.enforced_vmin if session.enforced_vmin is not None else 0.95
        v_max = session.enforced_vmax if session.enforced_vmax is not None else 1.05

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
        )

        story: list = []
        story.extend(self._build_title_page(
            session, goal_type=goal_type, contingency_entry=contingency_entry,
        ))
        story.append(PageBreak())
        story.extend(self._build_executive_summary(
            session, summary_text, goal_classification=gc,
            contingency_entry=contingency_entry,
        ))
        story.append(PageBreak())
        if contingency_entry is not None:
            story.extend(self._build_contingency_results_section(session, contingency_entry))
        elif is_sweep:
            story.extend(self._build_sweep_results_section(session, sweep_entry, goal_type=goal_type))
        else:
            story.extend(self._build_convergence_section(
                session, best_iteration=best_iter_override, goal_type=goal_type,
                v_min=v_min, v_max=v_max,
            ))
            story.append(PageBreak())
            story.extend(self._build_comparison_section(
                session, base_result, best_result, goal_type=goal_type,
                best_iteration_override=best_iter_override,
                v_min=v_min, v_max=v_max,
            ))
        story.append(PageBreak())
        story.extend(self._build_iteration_log(session))

        # TCOPFLOW temporal analysis section
        tcopflow_period_data = getattr(session, "tcopflow_period_data", None)
        if session.application == "tcopflow" and tcopflow_period_data:
            story.append(PageBreak())
            story.extend(self._build_tcopflow_temporal_section(session, tcopflow_period_data))

        # SOPFLOW stochastic analysis section
        sopflow_num_scenarios = getattr(session, "sopflow_num_scenarios", 0)
        if session.application == "sopflow" and sopflow_num_scenarios > 0:
            story.append(PageBreak())
            story.extend(self._build_sopflow_stochastic_section(session, sopflow_num_scenarios))

        # Add steering history section if any directives were used
        if steering_history:
            story.append(PageBreak())
            story.extend(self._build_steering_section(steering_history))

        # PFLOW vs OPFLOW benchmark section
        benchmark_result = getattr(session, "benchmark_result", None)
        if benchmark_result:
            story.append(PageBreak())
            story.extend(self._build_benchmark_section(benchmark_result))

        # Multi-objective section (only when applicable)
        if (
            hasattr(session.journal, "objective_registry")
            and session.journal.objective_registry.is_multi_objective
        ):
            story.append(PageBreak())
            story.extend(self._build_multi_objective_section(session, gc))

        doc.build(story)
        return buffer.getvalue()

    # ── Markdown / Summary Text Rendering ────────────────────────────────

    def _build_markdown_table(self, rows: list[list[str]]) -> Table:
        """Convert parsed markdown table rows into a styled ReportLab Table."""
        s = self._styles
        n_cols = max(len(r) for r in rows)
        padded = [r + [""] * (n_cols - len(r)) for r in rows]

        header_style = ParagraphStyle(
            "table_header", parent=s["body_small"],
            fontName=self._font_bold, fontSize=8,
            textColor=colors.white, leading=10,
        )
        cell_style = ParagraphStyle(
            "table_cell", parent=s["body_small"],
            fontName=self._font, fontSize=8, leading=10,
        )

        table_data = []
        for row_idx, row in enumerate(padded):
            style = header_style if row_idx == 0 else cell_style
            table_data.append([
                Paragraph(self._escape_xml(cell), style) for cell in row
            ])

        available_width = 17 * cm
        col_width = available_width / n_cols
        table = Table(table_data, colWidths=[col_width] * n_cols)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def _render_summary_text(self, text: str) -> list:
        """Render summary text, converting markdown tables to ReportLab Tables."""
        text = self._preprocess_text(text)
        elements: list = []
        lines = text.split("\n")
        current_block: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for code block start
            if line.strip().startswith("```"):
                if current_block:
                    elements.extend(self._render_text_block("\n".join(current_block)))
                    current_block = []

                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    i += 1  # Skip closing ```

                code_text = "\n".join(code_lines).strip()
                if code_text and not self._is_ascii_art(code_text):
                    escaped = self._escape_xml(code_text)
                    code_style = ParagraphStyle(
                        "code_block", fontName="Courier", fontSize=7,
                        spaceAfter=6, leading=9,
                        backColor=colors.HexColor("#f5f5f5"),
                        leftIndent=10, rightIndent=10,
                        spaceBefore=4,
                    )
                    elements.append(Paragraph(escaped.replace("\n", "<br/>"), code_style))
                continue

            # Check if this line starts a markdown table
            if "|" in line and i + 1 < len(lines) and _is_separator_row(lines[i + 1]):
                if current_block:
                    elements.extend(self._render_text_block("\n".join(current_block)))
                    current_block = []

                table_lines = [line]
                i += 1
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1

                rows = _parse_markdown_table("\n".join(table_lines))
                if rows:
                    elements.append(Spacer(1, 0.3 * cm))
                    elements.append(self._build_markdown_table(rows))
                    elements.append(Spacer(1, 0.3 * cm))
                continue

            current_block.append(line)
            i += 1

        if current_block:
            elements.extend(self._render_text_block("\n".join(current_block)))

        return elements

    def _render_text_block(self, text: str) -> list:
        """Render a non-table text block as Paragraph flowables."""
        s = self._styles
        elements: list = []

        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue

            # Skip ASCII art
            if self._is_ascii_art(para):
                continue

            # Headings
            if para.startswith("### "):
                heading_text = self._escape_xml(para[4:].strip())
                elements.append(Paragraph(heading_text, s["heading2"]))
            elif para.startswith("## "):
                heading_text = self._escape_xml(para[3:].strip())
                elements.append(Paragraph(heading_text, s["heading1"]))
            elif para.startswith("# "):
                heading_text = self._escape_xml(para[2:].strip())
                elements.append(Paragraph(heading_text, s["heading1"]))
            elif para.startswith("```"):
                # Code blocks that weren't caught at line level
                code_lines = para.split("\n")
                code_content = "\n".join(
                    l for l in code_lines if not l.strip().startswith("```")
                )
                if code_content.strip() and not self._is_ascii_art(code_content):
                    escaped = self._escape_xml(code_content)
                    code_style = ParagraphStyle(
                        "code", fontName="Courier", fontSize=8,
                        spaceAfter=6, leading=10,
                        backColor=colors.HexColor("#f5f5f5"),
                        leftIndent=10,
                    )
                    elements.append(Paragraph(escaped.replace("\n", "<br/>"), code_style))
            else:
                # Check for bullet lists
                lines = para.split("\n")
                bullet_lines = [l for l in lines if l.strip().startswith("- ")]
                if len(bullet_lines) > len(lines) / 2:
                    for line in lines:
                        line = line.strip()
                        if line.startswith("- "):
                            bullet_text = self._escape_xml(line[2:].strip())
                            elements.append(Paragraph(f"\u2022 {bullet_text}", s["bullet"]))
                        elif line:
                            elements.append(Paragraph(self._escape_xml(line), s["body"]))
                    continue

                # Regular paragraph
                cleaned = para.replace("**", "")
                cleaned = self._escape_xml(cleaned)
                cleaned = " ".join(cleaned.split("\n"))
                elements.append(Paragraph(cleaned, s["body"]))

        return elements

    # ── Title Page ───────────────────────────────────────────────────────

    def _build_title_page(
        self, session: SearchSession, goal_type: str | None = None,
        contingency_entry=None,
    ) -> list:
        s = self._styles
        elements: list = []
        elements.append(Spacer(1, 6 * cm))
        elements.append(Paragraph("AgentiGrid Search Report", s["title"]))
        if contingency_entry is not None:
            if getattr(contingency_entry, "reserve_meta", None):
                type_label = "Hot Reserve / N-1 Generator Security"
            else:
                meta = contingency_entry.contingency_meta or {}
                order = meta.get("order", "?")
                type_label = f"Contingency Analysis (N-{order})"
            elements.append(Paragraph(f"Search Type: {type_label}", s["caption"]))
        elif goal_type:
            type_label = goal_type.replace("_", " ").title()
            elements.append(Paragraph(f"Search Type: {type_label}", s["caption"]))
        elements.append(Spacer(1, 1 * cm))
        elements.append(Paragraph(self._escape_xml(session.goal), s["subtitle"]))
        elements.append(Spacer(1, 2 * cm))

        start = datetime.fromisoformat(session.start_time)
        elements.append(Paragraph(
            f"Date: {start.strftime('%Y-%m-%d %H:%M:%S')}", s["body"],
        ))
        _APP_LABELS = {
            "opflow": "Optimal Power Flow (OPFLOW)",
            "dcopflow": "DC Optimal Power Flow (DCOPFLOW)",
            "scopflow": "Security-Constrained OPF (SCOPFLOW)",
            "tcopflow": "Multi-Period OPF (TCOPFLOW)",
            "sopflow": "Stochastic OPF (SOPFLOW)",
            "pflow": "Power Flow (PFLOW)",
        }
        app_label = _APP_LABELS.get(session.application, session.application)
        elements.append(Paragraph(
            f"Application: {app_label}", s["body"],
        ))
        elements.append(Paragraph(
            f"Backend: {session.config.llm.backend} / {session.config.llm.model}", s["body"],
        ))
        elements.append(Spacer(1, 3 * cm))
        elements.append(Paragraph("Generated by AgentiGrid v0.1.0", s["caption"]))
        return elements

    # ── Executive Summary ────────────────────────────────────────────────

    def _build_executive_summary(
        self,
        session: SearchSession,
        summary_text: str | None,
        goal_classification: dict | None = None,
        contingency_entry=None,
    ) -> list:
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Executive Summary", s["heading1"]))

        gc = goal_classification
        best_iter_override = gc.get("best_iteration") if gc else None
        goal_type = gc.get("goal_type") if gc else None

        stats = session.journal.summary_stats(
            best_iteration_override=best_iter_override,
            goal_type=goal_type,
        )
        start = datetime.fromisoformat(session.start_time)
        end = datetime.fromisoformat(session.end_time) if session.end_time else datetime.now()
        duration = end - start
        total_tokens = session.total_prompt_tokens + session.total_completion_tokens

        # Reserve shortcut — checked before the contingency/sweep shortcuts
        if contingency_entry is not None and getattr(contingency_entry, "reserve_meta", None):
            rm = contingency_entry.reserve_meta
            n_on = rm.get("n_on", 0)
            passed = rm.get("passed_count", 0)
            failed = rm.get("failed_count", 0)
            secure = "N-1 secure" if rm.get("n1_secure") else "NOT N-1 secure"
            res_lines = []
            if rm.get("minimize"):
                secure_min = ("N-1 secure" if rm.get("n1_secure_min", rm.get("n1_secure"))
                              else "NOT N-1 secure")
                _lb_pg = rm.get("lower_bound_pg", 0)
                _lb_bus = rm.get("lower_bound_bus", "?")
                res_lines.append(
                    f"Minimum feasible hot reserve for N-1 = "
                    f"{rm.get('min_reserve', 0):.1f} MW ({secure_min}) — upper bound on the "
                    f"true minimum, global optimality not claimed. Found by greedy largest-Pmax-"
                    f"first de-commitment ({rm.get('n_decommitted', 0)} of {n_on} units "
                    f"de-committed; full-commitment reserve {rm.get('reserve_full', 0):.1f} MW). "
                    f"At this commitment the largest committed unit is gen@{_lb_bus} "
                    f"({_lb_pg:.1f} MW), so the N-1 reserve requirement at this operating "
                    f"point is {_lb_pg:.1f} MW."
                )
            res_lines.extend([
                (
                    f"Hot-reserve / N-1 generator security screen over {n_on} committed "
                    f"units: {passed}/{n_on} single-unit losses feasible ({secure})."
                ),
                (
                    f"Hot reserve available: {rm.get('hot_reserve_available', 0):.1f} MW; "
                    f"minimum required for N-1 (largest committed unit, "
                    f"gen@{rm.get('largest_pg_bus', '?')} at {rm.get('largest_pg', 0):.1f} MW): "
                    f"{rm.get('required_reserve_n1', 0):.1f} MW; "
                    f"margin {rm.get('margin', 0):.1f} MW."
                ),
            ])
            if failed:
                res_lines.append(
                    f"{failed} unit outage(s) infeasible — the arithmetic reserve margin "
                    "is not deliverable for these losses (required reserve is mis-located)."
                )
            res_lines.extend([
                f"Total iterations: {stats['total_iterations']}",
                f"Duration: {duration.total_seconds():.0f}s",
                f"Termination: {session.termination_reason}",
                (f"Token usage: {total_tokens:,}" if total_tokens > 0 else "Token usage: N/A"),
            ])
            for line in res_lines:
                elements.append(Paragraph(self._escape_xml(line), s["body"]))
            if summary_text:
                elements.append(Spacer(1, 0.5 * cm))
                elements.append(Paragraph("Analysis", s["heading2"]))
                elements.extend(self._render_summary_text(summary_text))
            return elements

        # Contingency shortcut — checked before sweep shortcut
        if contingency_entry is not None:
            meta = contingency_entry.contingency_meta or {}
            order = meta.get("order", "?")
            target = meta.get("target_bus", "?")
            neighbors = meta.get("neighbors") or []
            passed = meta.get("passed_count", 0)
            failed = meta.get("failed_count", 0)
            total = passed + failed
            nb_parts = [
                f"bus {nb} ({hop} hop{'s' if hop != 1 else ''})"
                for nb, hop in neighbors
            ]
            nb_str = ", ".join(nb_parts) if nb_parts else "—"
            headline = (
                f"N-{order} contingency screen on the {len(neighbors)} nearest "
                f"neighbors of bus {target} ({nb_str}): "
                f"{passed}/{total} contingencies feasible, {failed} failed."
            )
            cont_lines = [headline]
            # Relief summary when applicable
            variants = contingency_entry.explored_variants or []
            failed_vs = [v for v in variants if not v.get("passed")]
            if failed_vs and any("relief" in v for v in failed_vs):
                relief_vs = [v for v in failed_vs if "relief" in v]
                resolved_count = sum(
                    1 for v in relief_vs if (v.get("relief") or {}).get("resolved")
                )
                unresolved_count = len(relief_vs) - resolved_count
                measures_used = sorted({
                    v["relief"]["measure"]
                    for v in relief_vs
                    if (v.get("relief") or {}).get("resolved")
                    and (v.get("relief") or {}).get("measure")
                })
                relief_line = f"Relief: {resolved_count} of {failed} resolved"
                if measures_used:
                    relief_line += f" ({', '.join(measures_used)})"
                relief_line += f"; {unresolved_count} unresolved."
                cont_lines.append(relief_line)
            cont_lines.extend([
                f"Total iterations: {stats['total_iterations']}",
                f"Duration: {duration.total_seconds():.0f}s",
                f"Termination: {session.termination_reason}",
                (f"Token usage: {total_tokens:,}" if total_tokens > 0 else "Token usage: N/A"),
            ])
            for line in cont_lines:
                elements.append(Paragraph(self._escape_xml(line), s["body"]))
            if summary_text:
                elements.append(Spacer(1, 0.5 * cm))
                elements.append(Paragraph("Analysis", s["heading2"]))
                elements.extend(self._render_summary_text(summary_text))
            return elements

        # Sweep shortcut: replace the scalar-objective block with sweep summary
        sweep_entry = session.journal.get_sweep_entry()
        if sweep_entry is not None:
            n_cand = sweep_entry.candidate_count or 0
            n_feas = len(sweep_entry.feasible_buses or [])
            _variants = sweep_entry.explored_variants or []
            _is_boundary = any("max_feasible_mw" in v for v in _variants)
            if _is_boundary:
                _det = [v for v in _variants if v.get("max_feasible_mw") is not None]
                if _det:
                    _best = max(_det, key=lambda v: v.get("max_feasible_mw") or 0.0)
                    _headline = (
                        f"Boundary sweep complete: hosting capacity determined for "
                        f"{len(_det)} of {n_cand} buses. Highest: "
                        f"{_best['max_feasible_mw']:,.1f} MW at bus {_best['bus']}."
                    )
                else:
                    _headline = (
                        f"Boundary sweep complete: no hosting capacity could be determined "
                        f"for the {n_cand} candidate buses."
                    )
            else:
                _headline = f"Sweep complete: {n_feas} of {n_cand} candidate buses are feasible."
            sweep_lines = [
                _headline,
                f"Total iterations: {stats['total_iterations']}",
                f"Duration: {duration.total_seconds():.0f}s",
                f"Termination: {session.termination_reason}",
                f"Token usage: {total_tokens:,}" if total_tokens > 0 else "Token usage: N/A",
            ]
            for line in sweep_lines:
                elements.append(Paragraph(self._escape_xml(line), s["body"]))
            if summary_text:
                elements.append(Spacer(1, 0.5 * cm))
                elements.append(Paragraph("Analysis", s["heading2"]))
                elements.extend(self._render_summary_text(summary_text))
            return elements

        # Key results
        marginal_count = sum(
            1 for e in session.journal.entries if e.feasibility_detail == "marginal"
        )
        lines = [
            f"Total iterations: {stats['total_iterations']}",
            f"Feasible solutions: {stats['feasible_count']}",
            f"Infeasible: {stats['infeasible_count']}",
        ]
        if marginal_count > 0:
            lines.append(f"Marginal convergence: {marginal_count}")
        if session.application == "tcopflow":
            max_np = max((e.num_steps for e in session.journal.entries if e.num_steps > 0), default=0)
            if max_np > 0:
                lines.append(f"Time periods per run: {max_np}")
        if session.application == "sopflow":
            max_ns = max((e.num_scenarios for e in session.journal.entries if e.num_scenarios > 0), default=0)
            if max_ns > 0:
                lines.append(f"Wind scenarios per run: {max_ns}")
        lines.extend([
            f"Duration: {duration.total_seconds():.0f}s",
            f"Termination: {session.termination_reason}",
            f"Token usage: {total_tokens:,}" if total_tokens > 0 else "Token usage: N/A",
        ])

        if stats["best_objective"] is not None:
            base_entry = session.journal.entries[0] if session.journal.entries else None
            best_cost_str = f"${stats['best_objective']:,.2f} (iteration {stats['best_iteration']})"
            if base_entry and base_entry.objective_value and base_entry.objective_value != 0:
                pct = (stats["best_objective"] - base_entry.objective_value) / base_entry.objective_value * 100
                if goal_type in (None, "cost_minimization"):
                    reduction = -pct  # positive = saving
                    best_str = f"Best objective: {best_cost_str} — {reduction:.1f}% cost reduction vs base case"
                elif goal_type == "feasibility_boundary":
                    best_str = f"Cost at best solution: {best_cost_str} ({pct:+.1f}% vs base case — increase expected)"
                else:
                    best_str = f"Cost at best solution: {best_cost_str} ({pct:+.1f}% vs base case)"
            else:
                best_str = f"Best objective: {best_cost_str}"
            # For non-cost-minimization, lead with the rationale if available
            if goal_type not in (None, "cost_minimization") and gc and gc.get("best_iteration_rationale"):
                rationale = self._escape_xml(gc["best_iteration_rationale"])
                lines.insert(0, f"Goal achievement: {rationale}")
            lines.insert(0 if goal_type in (None, "cost_minimization") else 1, best_str)
        else:
            lines.insert(0, "No feasible solution found.")

        for line in lines:
            elements.append(Paragraph(self._escape_xml(line), s["body"]))

        # Goal achievement rationale for cost_minimization (non-cost types already inserted above)
        if goal_type in (None, "cost_minimization") and gc and gc.get("best_iteration_rationale"):
            rationale = self._escape_xml(gc["best_iteration_rationale"])
            elements.append(Paragraph(f"Goal achievement: {rationale}", s["body"]))

        # LLM Analysis
        if summary_text:
            elements.append(Spacer(1, 0.5 * cm))
            elements.append(Paragraph("Analysis", s["heading2"]))
            elements.extend(self._render_summary_text(summary_text))

        return elements

    # ── Sweep Results Section ────────────────────────────────────────────

    @staticmethod
    def _certified_reason(v: dict) -> str:
        """Derive displayed infeasibility reason from solver status (defensive).

        Uses the journaled `status` field as source of truth, so even historical
        journals that stored old heuristic labels ("line overload", etc.) render
        honestly.  Only a CONVERGED solve certifies a constraint violation; every
        other outcome is non-convergence and nothing about the iterate is certified.
        """
        status = (v.get("status") or "").upper()
        if status.startswith("CONVERGED"):
            return "Constraint violation"
        return "Did not converge"

    def _build_boundary_table(self, sweep_entry, variants: list, n_cand: int) -> list:
        """Build the hosting-capacity table for a boundary (max-injection) sweep."""
        s = self._styles
        elements: list = []

        determined = [v for v in variants if v.get("max_feasible_mw") is not None]
        undetermined = [v for v in variants if v.get("max_feasible_mw") is None]
        determined.sort(key=lambda v: v.get("max_feasible_mw") or 0.0, reverse=True)
        undetermined.sort(key=lambda v: v["bus"])

        entity = next((v.get("entity") for v in variants if v.get("entity")), "injection")

        elements.append(Paragraph(
            f"The maximum feasible {self._escape_xml(str(entity))} injection was searched "
            f"at {n_cand} candidate buses by bisection. A boundary was determined for "
            f"{len(determined)} of {n_cand} buses.",
            s["body"],
        ))
        elements.append(Spacer(1, 0.4 * cm))

        elements.append(Paragraph(
            f"Hosting capacity by bus ({len(determined)})", s["heading2"],
        ))
        if determined:
            header = [
                "Bus", "Max Feasible (MW)", "Binding Constraint",
                "Boundary V_min", "Boundary V_max", "Max Load (%)", "Probes",
            ]
            rows = [header]
            for v in determined:
                mfm = v.get("max_feasible_mw")
                rows.append([
                    str(v["bus"]),
                    f"{mfm:,.1f}" if isinstance(mfm, (int, float)) else "—",
                    self._escape_xml(str(v.get("binding_constraint", ""))),
                    f"{v.get('voltage_min', 0):.3f}",
                    f"{v.get('voltage_max', 0):.3f}",
                    f"{v.get('max_line_loading_pct', 0):.1f}",
                    str(v.get("probes_used", 0)),
                ])
            col_widths = [1.5 * cm, 2.8 * cm, 5.2 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 1.5 * cm]
            table = Table(rows, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a085")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eafaf6")]),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 0), (2, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("No boundary could be determined for any bus.", s["body"]))

        if undetermined:
            elements.append(Spacer(1, 0.3 * cm))
            bus_list = ", ".join(str(v["bus"]) for v in undetermined)
            elements.append(Paragraph(
                f"Undetermined buses ({len(undetermined)}): {self._escape_xml(bus_list)}.",
                s["caption"],
            ))

        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph(
            "The boundary reported is the OPFLOW convergence boundary: the largest injection "
            "for which IPOPT converges with the voltage band and thermal limits (Rate A) "
            "enforced as in-solve hard constraints. Non-convergence is treated as the "
            "infeasible signal that caps the bisection — a probe that fails for numerical "
            "rather than physical reasons would understate the true hosting capacity. "
            "The binding constraint is identified at the maximum-feasible operating point.",
            s["caption"],
        ))
        return elements

    def _build_sweep_results_section(
        self, session, sweep_entry, goal_type: str | None = None,
    ) -> list:
        """Build the sweep-results section: feasible-bus table + infeasible table."""
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Sweep Results", s["heading1"]))

        variants = sweep_entry.explored_variants or []
        n_cand = sweep_entry.candidate_count or len(variants)

        # Boundary (hosting-capacity) sweeps carry max_feasible_mw per candidate
        # and use a dedicated table instead of the feasible/infeasible split.
        if any("max_feasible_mw" in v for v in variants):
            elements.extend(
                self._build_boundary_table(sweep_entry, variants, n_cand)
            )
            return elements

        feasible = [v for v in variants if v.get("feasible")]
        infeasible = [v for v in variants if not v.get("feasible")]
        infeasible.sort(key=lambda v: v["bus"])

        # C2/C3 detection: dispatchable-generator siting and custom metric/predicate sweeps.
        is_dispatchable = any(v.get("dispatched_pg") is not None for v in variants)
        has_dispatched_q = any(v.get("dispatched_q") is not None for v in variants)
        metric_name = next((v.get("metric_name") for v in variants if v.get("metric_name")), None)
        predicate_name = next((v.get("predicate_name") for v in variants if v.get("predicate_name")), None)
        if metric_name == "max_delta_v":
            feasible.sort(key=lambda v: v.get("metric_value") or 0.0, reverse=True)
        else:
            feasible.sort(key=lambda v: v["bus"])

        elements.append(Paragraph(
            f"The mutation described as “{self._escape_xml(sweep_entry.description)}” "
            f"was tested at {n_cand} candidate buses. "
            f"{len(feasible)} are feasible; {len(infeasible)} are infeasible "
            f"under the stated criteria.",
            s["body"],
        ))

        # Fix 3 — voltage-criterion honesty note
        elements.append(Paragraph(
            "Note: Under OPFLOW, the voltage band and thermal limits (Rate A) are enforced "
            "as in-solve hard constraints, so the stated voltage criterion is satisfied by "
            "construction for any converged candidate and is not an independent discriminating "
            "filter for feasibility.",
            s["caption"],
        ))
        elements.append(Spacer(1, 0.5 * cm))

        # ── Feasible-bus table ───────────────────────────────────────────
        if predicate_name == "reactive_adequacy":
            feas_heading = f"Reactive-adequate buses ({len(feasible)})"
            elements.append(Paragraph(feas_heading, s["heading2"]))
            elements.append(Paragraph(
                "A bus is reactive-adequate if a feasible OPF exists with the added unit "
                "forced to (P = Pmax, Q = Qmax) — Qmin and Qmax are pinned to the target so "
                "the reactive output is forced, not merely bounded. This is a reactive-headroom "
                "test, not the standard voltage/loading criterion. “Dispatched Q” audits the "
                "forcing: it should equal the Qmax target at every adequate bus.",
                s["caption"],
            ))
        elif metric_name == "max_delta_v":
            elements.append(Paragraph(f"Buses by voltage step ({len(feasible)})", s["heading2"]))
            elements.append(Paragraph(
                "Max ΔV is the worst system-wide voltage change |V − V_base| between the "
                "base-case OPF solution and the OPF solution with the load block added at the "
                "candidate bus; both are cost-optimal solutions in which the solver re-dispatches "
                "reactive support. It is a steady-state sensitivity, not a physical "
                "switching/energization transient. Because this sweep relaxes the bus voltage "
                "band, the cost-optimal voltage profile has slack, so small differences between "
                "the top-ranked buses reflect the optimizer's freedom to redistribute voltage "
                "setpoints rather than a physically meaningful ranking — read the result as "
                "identifying the high-sensitivity region, not a single uniquely-worst bus.",
                s["caption"],
            ))
        else:
            elements.append(Paragraph(f"Feasible buses ({len(feasible)})", s["heading2"]))
            if is_dispatchable:
                elements.append(Paragraph(
                    "Generator mode: dispatchable (Pmin = 0, Pmax = cap) under economic "
                    "dispatch with a cost curve. The OPF chooses each unit's output; "
                    "“Dispatched Pg” is the optimized output at that location (a unit "
                    "dispatching ~0 MW is not helping there).",
                    s["caption"],
                ))
        if feasible:
            # Column set adapts to the sweep type (C2 dispatched Pg, C3 metric column).
            col_specs: list = [
                ("Bus", 1.6 * cm, lambda v: str(v["bus"])),
                ("V_min (pu)", 2.3 * cm, lambda v: f"{v.get('voltage_min', 0):.3f}"),
                ("V_max (pu)", 2.3 * cm, lambda v: f"{v.get('voltage_max', 0):.3f}"),
                ("Max Loading (%)", 2.8 * cm, lambda v: f"{v.get('max_line_loading_pct', 0):.1f}"),
                ("Violations", 2.0 * cm, lambda v: str(v.get("violations", 0))),
            ]
            if metric_name == "max_delta_v":
                col_specs.append((
                    "Max ΔV (pu)", 2.6 * cm,
                    lambda v: f"{v.get('metric_value'):.4f}"
                    if isinstance(v.get("metric_value"), (int, float)) else "—",
                ))
            else:
                col_specs.append((
                    "System Cost ($)", 3 * cm,
                    lambda v: f"{v.get('cost'):,.2f}"
                    if isinstance(v.get("cost"), (int, float)) and v.get("cost") else "—",
                ))
            if is_dispatchable:
                col_specs.append((
                    "Dispatched Pg (MW)", 3 * cm,
                    lambda v: f"{v.get('dispatched_pg'):,.1f}"
                    if isinstance(v.get("dispatched_pg"), (int, float)) else "—",
                ))
            if has_dispatched_q:
                col_specs.append((
                    "Dispatched Q (MVAr)", 3 * cm,
                    lambda v: f"{v.get('dispatched_q'):,.1f}"
                    if isinstance(v.get("dispatched_q"), (int, float)) else "—",
                ))

            header = [c[0] for c in col_specs]
            col_widths = [c[1] for c in col_specs]
            rows = [header] + [[c[2](v) for c in col_specs] for v in feasible]
            table = Table(rows, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(table)

            # Fix 2 — cost-minimization near-optimal ranking (cost sweeps only,
            # not when a custom non-cost metric drives the ranking)
            if goal_type in (None, "cost_minimization") and metric_name != "max_delta_v":
                feasible_with_cost = sorted(
                    [(v, v.get("cost")) for v in feasible if isinstance(v.get("cost"), (int, float))],
                    key=lambda x: x[1],
                )
                if feasible_with_cost:
                    top_k = getattr(session.config.report, "cost_min_top_k", 10)
                    abs_tol = getattr(session.config.report, "near_optimal_abs_tol", 5.0)
                    top_buses = feasible_with_cost[:top_k]
                    best_cost = top_buses[0][1]

                    elements.append(Spacer(1, 0.4 * cm))
                    elements.append(Paragraph(
                        f"Cost ranking — top {len(top_buses)} cheapest buses", s["heading2"],
                    ))
                    rank_header = ["Rank", "Bus", "Cost ($/h)", "Δ from best ($/h)"]
                    rank_rows = [rank_header]
                    for rank, (v, cost) in enumerate(top_buses, 1):
                        delta = cost - best_cost
                        rank_rows.append([
                            str(rank),
                            str(v["bus"]),
                            f"{cost:,.2f}",
                            f"+{delta:.2f}" if delta > 0 else "0.00",
                        ])
                    rank_col_widths = [1.5 * cm, 2 * cm, 4.5 * cm, 4.5 * cm]
                    rank_table = Table(rank_rows, colWidths=rank_col_widths, repeatRows=1)
                    rank_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("FONTNAME", (0, 1), (-1, -1), self._font),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                        ("ALIGN", (0, 0), (1, -1), "CENTER"),
                        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))
                    elements.append(rank_table)

                    # Near-optimal caveat when gap is below solver tolerance
                    if len(top_buses) >= 2:
                        gap = top_buses[1][1] - best_cost
                        if gap < abs_tol:
                            elements.append(Spacer(1, 0.2 * cm))
                            elements.append(Paragraph(
                                f"Note: The cost difference between the leading candidates "
                                f"(${gap:.2f}/h) is below the solver tolerance threshold "
                                f"(${abs_tol:.1f}/h). These buses should be treated as "
                                f"equivalently optimal rather than strictly ranked.",
                                s["caption"],
                            ))
        else:
            elements.append(Paragraph("No buses were found to be feasible.", s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # ── Infeasible-bus table (Fix 1b) ────────────────────────────────
        elements.append(Paragraph(f"Infeasible buses ({len(infeasible)})", s["heading2"]))
        if infeasible:
            # Two-row header: Bus + Status span both rows; metric columns grouped
            # under "Last iterate — uncertified" in the first row.
            header_row0 = ["Bus", "Status", "Last iterate — uncertified", "", "", ""]
            header_row1 = ["", "", "V_min (pu)", "V_max (pu)", "Max Load (%)", "Violations"]
            data_rows = []
            for v in infeasible:
                data_rows.append([
                    str(v["bus"]),
                    self._certified_reason(v),
                    f"{v.get('voltage_min', 0):.3f}",
                    f"{v.get('voltage_max', 0):.3f}",
                    f"{v.get('max_line_loading_pct', 0):.1f}",
                    str(v.get("violations", 0)),
                ])
            rows = [header_row0, header_row1] + data_rows
            col_widths = [1.8 * cm, 3.8 * cm, 2.4 * cm, 2.4 * cm, 3.2 * cm, 2.4 * cm]
            table = Table(rows, colWidths=col_widths, repeatRows=2)
            table.setStyle(TableStyle([
                # Bus spans rows 0-1; Status spans rows 0-1
                ("SPAN", (0, 0), (0, 1)),
                ("SPAN", (1, 0), (1, 1)),
                # "Last iterate" spans metric columns in row 0 only
                ("SPAN", (2, 0), (5, 0)),
                # Header background for both header rows
                ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#c0392b")),
                ("TEXTCOLOR", (0, 0), (-1, 1), colors.white),
                ("FONTNAME", (0, 0), (-1, 1), self._font_bold),
                # Data rows
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 2), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#fdf2f1")]),
                # Alignment
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (2, 0), (5, 0), "CENTER"),  # "Last iterate" header centred
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3 * cm))
            elements.append(Paragraph(
                "The “Status” column reflects solver certification only: a solve either "
                "converged to a feasible operating point or it did not. "
                "The four rightmost columns are from the solver’s last uncertified iterate "
                "and must not be read as the certified cause of infeasibility — they are "
                "diagnostic hints only. No certified operating point exists for “Did not "
                "converge” buses, so their iterate metrics may appear within limits even "
                "though no feasible solution was found.",
                s["caption"],
            ))
        else:
            elements.append(Paragraph("None — all candidate buses are feasible.", s["body"]))

        return elements

    # ── Hot Reserve Assessment (C.8) ─────────────────────────────────────

    def _build_hot_reserve_assessment(self, rm: dict, v_min: float, v_max: float) -> list:
        """Build the Hot Reserve Assessment block for a reserve / N-1 generator screen."""
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Hot Reserve Assessment", s["heading2"]))

        n_on = rm.get("n_on", 0)

        # --- Minimization result (C.8 Path A) leads the section, if present ---
        if rm.get("minimize"):
            secure_min = "Yes" if rm.get("n1_secure_min", rm.get("n1_secure")) else "No"
            min_reserve = rm.get("min_reserve", 0.0)
            reserve_full = rm.get("reserve_full", 0.0)
            n_decommitted = rm.get("n_decommitted", 0)
            final_on = rm.get("final_on_count", n_on)
            lb_pg = rm.get("lower_bound_pg", 0.0)
            lb_bus = rm.get("lower_bound_bus", "?")
            elements.append(Paragraph(
                "Goal: minimum feasible hot reserve that remains N-1 secure. Method: greedy "
                "largest-Pmax-first security-constrained de-commitment (re-solving the OPF and "
                "the full N-1 generator screen after each de-commitment). The reported value is "
                "an upper bound on the true minimum; global optimality is not claimed.",
                s["caption"],
            ))
            elements.append(Spacer(1, 0.2 * cm))
            min_rows = [
                ["Metric", "Value"],
                [
                    "Minimum feasible hot reserve (N-1 secure, greedy upper bound)",
                    f"{min_reserve:,.1f} MW",
                ],
                ["Hot reserve at full commitment", f"{reserve_full:,.1f} MW"],
                [
                    "Units de-committed",
                    f"{n_decommitted} of {n_on} ({final_on} remain committed)",
                ],
                [
                    "Largest committed unit at minimized commitment (its N-1 requirement)",
                    f"{lb_pg:,.1f} MW at gen@{lb_bus}",
                ],
                ["N-1 secure at minimum", secure_min],
            ]
            if rm.get("hit_budget"):
                min_rows.append(
                    ["Solve budget", "reached — reported commitment is best-so-far"]
                )
            min_table = Table(min_rows, colWidths=[10 * cm, 7 * cm])
            min_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a085")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eafaf6")]),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(min_table)
            elements.append(Spacer(1, 0.3 * cm))
            # Reserve-reduction trajectory chart
            trajectory = rm.get("trajectory") or []
            if trajectory:
                traj_fig = reserve_trajectory_chart(trajectory, height=280)
                traj_bytes = _export_chart_image(traj_fig, width_px=700, height_px=280)
                if traj_bytes:
                    elements.append(Image(io.BytesIO(traj_bytes), width=16 * cm, height=7 * cm))
                    elements.append(Paragraph(
                        "Hot reserve as each greedy de-commitment is accepted. "
                        "Rejected attempts are not plotted; the chart shows only accepted steps.",
                        s["caption"],
                    ))
                    elements.append(Spacer(1, 0.3 * cm))
            elements.append(Paragraph(
                "The full-commitment starting point below is the base assessment. "
                "The N-1 generator screen table further down reflects the minimized "
                "commitment — global optimality not claimed.",
                s["caption"],
            ))
            elements.append(Spacer(1, 0.3 * cm))
            elements.append(Paragraph("Full-commitment starting point", s["heading2"]))

        secure = "Yes" if rm.get("n1_secure") else "No"
        rows = [
            ["Metric", "Value"],
            ["Committed (on-line) units", str(n_on)],
            ["Hot reserve available (Σ Pmax−Pg)", f"{rm.get('hot_reserve_available', 0):,.1f} MW"],
            [
                "Largest committed unit (worst N-1 loss)",
                f"gen@{rm.get('largest_pg_bus', '?')}: {rm.get('largest_pg', 0):,.1f} MW "
                f"(capacity {rm.get('largest_pmax', 0):,.1f} MW)",
            ],
            ["Minimum reserve required for N-1", f"{rm.get('required_reserve_n1', 0):,.1f} MW"],
            ["Reserve margin (available − required)", f"{rm.get('margin', 0):,.1f} MW"],
            [
                "N-1 generator secure",
                f"{secure} ({rm.get('passed_count', 0)}/{n_on} unit outages feasible)",
            ],
        ]
        table = Table(rows, colWidths=[8 * cm, 9 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16a085")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eafaf6")]),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph(
            "The minimum hot reserve required for N-1 is the output of the largest single "
            "committed generator — the reserve that must be deployable elsewhere to replace "
            "its loss. Feasibility of each unit loss is verified by a full post-outage OPF "
            "redispatch under the voltage band "
            f"(V ∈ [{v_min:.2f}, {v_max:.2f}] pu) and Rate A thermal limits (deliverability), "
            "not a copperplate balance. If any unit loss is infeasible the system is not N-1 "
            "secure regardless of the arithmetic margin, and the required reserve is "
            "mis-located (a congested-pocket unit, not the largest, is binding). Reserve is "
            "assessed at fixed commitment; OPFLOW does not de-commit units.",
            s["caption"],
        ))
        return elements

    # ── Contingency Results Section ──────────────────────────────────────

    def _build_contingency_results_section(self, session, entry) -> list:
        """Build the contingency-screening section: failed table, relief audit, passed rankings."""
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Contingency Screening", s["heading1"]))

        variants = entry.explored_variants or []
        v_min = session.enforced_vmin if session.enforced_vmin is not None else 0.95
        v_max = session.enforced_vmax if session.enforced_vmax is not None else 1.05

        reserve_meta = getattr(entry, "reserve_meta", None)
        is_reserve = bool(reserve_meta)

        if is_reserve:
            elements.extend(self._build_hot_reserve_assessment(reserve_meta, v_min, v_max))
            elements.append(Spacer(1, 0.4 * cm))
            if reserve_meta.get("minimize"):
                # variants now hold the minimized-commitment screen (not the 54-unit one).
                failed_count = sum(1 for v in variants if not v.get("passed"))
                passed_count = len(variants) - failed_count
                final_on = reserve_meta.get("final_on_count", len(variants))
                elements.append(Paragraph(
                    f"Generator N-1 screen at the minimized commitment ({final_on} units)",
                    s["heading2"],
                ))
                elements.append(Spacer(1, 0.2 * cm))
            else:
                passed_count = reserve_meta.get("passed_count", 0)
                failed_count = reserve_meta.get("failed_count", 0)
            failed_label = "Failed generator N-1 outages"
        else:
            meta = entry.contingency_meta or {}
            order = meta.get("order", "?")
            target_bus = meta.get("target_bus", "?")
            neighbors = meta.get("neighbors") or []
            passed_count = meta.get("passed_count", 0)
            failed_count = meta.get("failed_count", 0)
            total = passed_count + failed_count

            # Infer component kinds from variant data
            kinds = sorted({k for v in variants for k in (v.get("kinds") or [])})
            nb_parts = [
                f"bus {nb} ({hop} hop{'s' if hop != 1 else ''})"
                for nb, hop in neighbors
            ]
            nb_str = ", ".join(nb_parts) if nb_parts else "—"

            intro = (
                f"N-{order} contingency screen: {len(neighbors)} nearest neighbors of "
                f"bus {target_bus} ({nb_str}); "
                f"components tested: {', '.join(kinds) if kinds else 'all'}; "
                f"feasibility band: V ∈ [{v_min:.2f}, {v_max:.2f}] pu, Rate A. "
                f"Result: {passed_count}/{total} feasible, {failed_count} failed."
            )
            elements.append(Paragraph(self._escape_xml(intro), s["body"]))
            elements.append(Spacer(1, 0.3 * cm))

            elements.append(Paragraph(
                "Under OPFLOW, the voltage band and Rate A thermal limits are enforced as "
                "in-solve hard constraints. A contingency PASSES if and only if the post-outage "
                "OPF converges with all constraints satisfied. A FAIL is non-convergence; the "
                "solver's last-iterate metrics for failed cases are uncertified and may not "
                "reflect the actual binding constraint.",
                s["caption"],
            ))
            elements.append(Spacer(1, 0.4 * cm))
            failed_label = "Failed contingencies"

        # ── Failed-contingency table ─────────────────────────────────────
        failed_variants = [v for v in variants if not v.get("passed")]
        passed_variants = [v for v in variants if v.get("passed")]
        has_relief = any("relief" in v for v in failed_variants)

        elements.append(Paragraph(f"{failed_label} ({failed_count})", s["heading2"]))
        if failed_variants:
            if has_relief:
                header = ["Contingency", "Component(s)", "Status", "Relief measure", "Relief detail"]
                col_widths_f = [3.2 * cm, 2.6 * cm, 2.6 * cm, 3.1 * cm, 5.5 * cm]
            else:
                header = ["Contingency", "Component(s)", "Status"]
                col_widths_f = [5.5 * cm, 5.5 * cm, 6 * cm]

            data_rows = []
            for v in failed_variants:
                label = self._escape_xml(v.get("label") or "")
                kinds_str = self._escape_xml(", ".join(v.get("kinds") or []))
                status = self._certified_reason(v)
                row = [label, kinds_str, status]
                if has_relief:
                    rel = v.get("relief") or {}
                    if rel.get("resolved"):
                        measure = self._escape_xml(str(rel.get("measure") or "—"))
                        detail = self._escape_xml(str(rel.get("detail") or "—"))
                    else:
                        measure = "unresolved"
                        detail = self._escape_xml(rel.get("detail") or "no local measure")
                    row += [measure, detail]
                data_rows.append(row)

            rows = [header] + data_rows
            table = Table(rows, colWidths=col_widths_f, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c0392b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fdf2f1")]),
                ("ALIGN", (0, 0), (1, -1), "LEFT"),
                ("ALIGN", (2, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.3 * cm))
            elements.append(Paragraph(
                "Status reflects solver certification only: 'Did not converge' means no "
                "certified post-contingency operating point exists. Last-iterate metrics "
                "are omitted for failed cases as they are uncertified.",
                s["caption"],
            ))
        else:
            elements.append(Paragraph(
                "No contingencies failed — the network is N-secure under all tested outages.",
                s["body"],
            ))

        # ── Relief attempts audit ────────────────────────────────────────
        if failed_variants and has_relief:
            elements.append(Spacer(1, 0.4 * cm))
            elements.append(Paragraph("Relief attempts audit", s["heading2"]))
            elements.append(Paragraph(
                "Priority-ordered relief search for each failed contingency. "
                "generator_redispatch is an inherent OPF no-op: generator Pg is an "
                "optimization variable, so the post-contingency OPF already includes "
                "economic redispatch. It is listed for completeness but performs no "
                "additional solve and never resolves a failure on its own.",
                s["caption"],
            ))
            elements.append(Spacer(1, 0.2 * cm))

            for v in failed_variants:
                rel = v.get("relief")
                if not rel:
                    continue
                label = v.get("label") or ""
                attempts = rel.get("attempts") or []
                parts = []
                for m, resolved in attempts:
                    if m == "generator_redispatch":
                        parts.append(f"{m} ✗ [inherent to OPF]")
                    elif resolved:
                        parts.append(f"{m} ✓")
                    else:
                        parts.append(f"{m} ✗")
                trail = ", ".join(parts) if parts else "—"
                line = f"{label}: {trail}"
                elements.append(Paragraph(self._escape_xml(line), s["body_small"]))

        # ── Passed contingencies — stress rankings ───────────────────────
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(Paragraph(f"Passed contingencies ({passed_count})", s["heading2"]))
        if passed_variants:
            elements.append(Paragraph(
                f"{passed_count} contingencies passed (post-outage OPF converged feasibly). "
                "Full per-contingency data is available in the JSON journal. "
                "The tables below surface the most-stressed passing cases.",
                s["body"],
            ))
            elements.append(Spacer(1, 0.3 * cm))

            stress_col_widths = [4.5 * cm, 4 * cm, 2.5 * cm, 2.5 * cm, 3.5 * cm]
            stress_header = ["Contingency", "Component(s)", "V_min (pu)", "V_max (pu)", "Max Load (%)"]
            stress_style = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("ALIGN", (0, 0), (1, -1), "LEFT"),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])

            def _stress_row(v):
                return [
                    self._escape_xml(v.get("label") or ""),
                    self._escape_xml(", ".join(v.get("kinds") or [])),
                    f"{v.get('voltage_min', 0):.3f}",
                    f"{v.get('voltage_max', 0):.3f}",
                    f"{v.get('max_line_loading_pct', 0):.1f}",
                ]

            # Top 10 by max_line_loading_pct descending
            by_loading = sorted(
                [v for v in passed_variants if isinstance(v.get("max_line_loading_pct"), (int, float))],
                key=lambda v: v.get("max_line_loading_pct", 0),
                reverse=True,
            )[:10]
            if by_loading:
                elements.append(Paragraph("Most stressed: highest line loading", s["heading2"]))
                elements.append(Paragraph(
                    "Under OPF thermal enforcement all passing contingencies cluster "
                    "near 100% loading — this table is less discriminating for N-1 screens. "
                    "The lowest-voltage table below is the primary stress indicator.",
                    s["caption"],
                ))
                load_rows = [stress_header] + [_stress_row(v) for v in by_loading]
                load_table = Table(load_rows, colWidths=stress_col_widths, repeatRows=1)
                load_table.setStyle(stress_style)
                elements.append(load_table)
                elements.append(Spacer(1, 0.3 * cm))

            # Bottom 10 by voltage_min ascending
            by_vmin = sorted(
                [v for v in passed_variants if isinstance(v.get("voltage_min"), (int, float))],
                key=lambda v: v.get("voltage_min", 1.0),
            )[:10]
            if by_vmin:
                elements.append(Paragraph("Most stressed: lowest voltage minimum", s["heading2"]))
                vmin_rows = [stress_header] + [_stress_row(v) for v in by_vmin]
                vmin_table = Table(vmin_rows, colWidths=stress_col_widths, repeatRows=1)
                vmin_table.setStyle(stress_style)
                elements.append(vmin_table)
        else:
            elements.append(Paragraph("No contingencies passed.", s["body"]))

        # ── Relief scope caveat ──────────────────────────────────────────
        if failed_variants and has_relief:
            elements.append(Spacer(1, 0.4 * cm))
            elements.append(Paragraph(
                "Relief scope note: relief candidates are local to the contingency "
                "neighborhood (incident branches, transformers, and loads at the focus "
                "buses). 'Unresolved' means the failure could not be relieved by local "
                "measures; system-wide redispatch or remote switching may still restore "
                "feasibility but is outside this screen's scope. Load curtailment is "
                "applied uniformly across focus buses; in islanded sub-networks this may "
                "over-curtail relative to a targeted load shed.",
                s["caption"],
            ))

        return elements

    # ── Convergence Section ──────────────────────────────────────────────

    def _build_convergence_section(
        self,
        session: SearchSession,
        best_iteration: int | None = None,
        goal_type: str | None = None,
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> list:
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Convergence Analysis", s["heading1"]))

        stats = session.journal.summary_stats(
            best_iteration_override=best_iteration, goal_type=goal_type,
        )

        # Auto-generated text
        n = stats["total_iterations"]
        text = f"The search ran for {n} iterations."
        if stats["best_objective"] is not None:
            base_entry = session.journal.entries[0] if session.journal.entries else None
            if base_entry and base_entry.objective_value and base_entry.objective_value != 0:
                pct = (stats["best_objective"] - base_entry.objective_value) / base_entry.objective_value * 100
                if goal_type in (None, "cost_minimization"):
                    text += f" A {-pct:.1f}% cost reduction was achieved vs the base case."
                else:
                    text += f" Cost changed by {pct:+.1f}% vs the base case."
            text += f" The best solution (objective ${stats['best_objective']:,.2f}) was found at iteration {stats['best_iteration']}."
        elements.append(Paragraph(text, s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Convergence chart
        fig = convergence_chart(
            session.journal, highlight_best=True, height=350,
            best_iteration=best_iteration,
        )
        img_bytes = _export_chart_image(fig, width_px=700, height_px=350)
        if img_bytes:
            elements.append(Image(io.BytesIO(img_bytes), width=16 * cm, height=8 * cm))
            elements.append(Paragraph("Objective value convergence across iterations.", s["caption"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Voltage range chart
        fig_v = voltage_range_chart(session.journal, height=300, v_min_limit=v_min, v_max_limit=v_max)
        img_bytes_v = _export_chart_image(fig_v, width_px=700, height_px=300)
        if img_bytes_v:
            elements.append(Image(io.BytesIO(img_bytes_v), width=16 * cm, height=7 * cm))
            elements.append(Paragraph("Voltage range envelope across iterations.", s["caption"]))

        return elements

    # ── Comparison Section ───────────────────────────────────────────────

    def _build_comparison_section(
        self,
        session: SearchSession,
        base_result: OPFLOWResult | None,
        best_result: OPFLOWResult | None,
        goal_type: str | None = None,
        best_iteration_override: int | None = None,
        v_min: float = 0.95,
        v_max: float = 1.05,
    ) -> list:
        s = self._styles
        elements: list = []

        heading = "Results Comparison"
        if goal_type == "feasibility_boundary":
            heading = "Base Case vs Maximum Feasible Configuration"
        elif goal_type == "constraint_satisfaction":
            heading = "Base Case vs Best Constraint-Satisfying Configuration"
        elif goal_type == "parameter_exploration":
            heading = "Base Case vs Selected Exploration Result"
        elements.append(Paragraph(heading, s["heading1"]))

        # Build comparison table from journal entries
        stats = session.journal.summary_stats(
            best_iteration_override=best_iteration_override,
            goal_type=goal_type,
        )
        base_entry = session.journal.entries[0] if session.journal.entries else None
        best_entry = None
        if stats.get("best_iteration") is not None:
            for e in session.journal.entries:
                if e.iteration == stats["best_iteration"]:
                    best_entry = e
                    break

        def _fv(v, fmt=".2f"):
            return f"{v:{fmt}}" if v is not None and v != 0 else "—"

        header = ["Metric", "Base Case", "Best Solution", "Change"]
        rows = [header]

        if base_entry:
            bv = base_entry.objective_value
            sv = best_entry.objective_value if best_entry else None
            change = ""
            if bv is not None and sv is not None and bv != 0:
                pct = (sv - bv) / bv * 100
                change = f"{pct:+.1f}%"
            rows.append([
                "Objective ($)",
                f"${bv:,.2f}" if bv is not None else "—",
                f"${sv:,.2f}" if sv is not None else "N/A",
                change or "—",
            ])
            rows.append([
                "Generation (MW)",
                _fv(base_entry.total_gen_mw, ".1f"),
                _fv(best_entry.total_gen_mw, ".1f") if best_entry else "N/A",
                f"{best_entry.total_gen_mw - base_entry.total_gen_mw:+.1f}" if best_entry and base_entry.total_gen_mw else "—",
            ])
            rows.append([
                "Voltage Min (p.u.)",
                _fv(base_entry.voltage_min, ".4f"),
                _fv(best_entry.voltage_min, ".4f") if best_entry else "N/A",
                "—",
            ])
            rows.append([
                "Voltage Max (p.u.)",
                _fv(base_entry.voltage_max, ".4f"),
                _fv(best_entry.voltage_max, ".4f") if best_entry else "N/A",
                "—",
            ])
            rows.append([
                "Max Line Loading (%)",
                _fv(base_entry.max_line_loading_pct, ".1f"),
                _fv(best_entry.max_line_loading_pct, ".1f") if best_entry else "N/A",
                "—",
            ])
            rows.append([
                "Violations",
                str(base_entry.violations_count),
                str(best_entry.violations_count) if best_entry else "N/A",
                str(best_entry.violations_count - base_entry.violations_count) if best_entry else "—",
            ])

        table = Table(rows, colWidths=[5 * cm, 4 * cm, 4 * cm, 3 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 1 * cm))

        # Voltage profile chart
        fig_vp = voltage_profile_chart(base_result, best_result, v_min_limit=v_min, v_max_limit=v_max)
        if fig_vp is not None:
            img_bytes = _export_chart_image(fig_vp, width_px=700, height_px=400)
            if img_bytes:
                elements.append(Image(io.BytesIO(img_bytes), width=16 * cm, height=9 * cm))
                elements.append(Paragraph("Bus voltage profile comparison.", s["caption"]))
                elements.append(Spacer(1, 0.5 * cm))

        # Generator dispatch chart
        fig_gen = generator_dispatch_chart(base_result, best_result)
        if fig_gen is not None:
            img_bytes = _export_chart_image(fig_gen, width_px=700, height_px=400)
            if img_bytes:
                elements.append(Image(io.BytesIO(img_bytes), width=16 * cm, height=9 * cm))
                elements.append(Paragraph("Generator dispatch comparison.", s["caption"]))

        return elements

    # ── Iteration Log ────────────────────────────────────────────────────

    def _build_iteration_log(self, session: SearchSession) -> list:
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Iteration Log", s["heading1"]))

        is_tcopflow = session.application == "tcopflow"
        is_sopflow = session.application == "sopflow"
        if is_tcopflow:
            header = ["Iter", "Description", "Cost ($)", "Feas.", "Np", "V_min", "V_max", "Load%", "Time(s)"]
        elif is_sopflow:
            header = ["Iter", "Description", "Cost ($)", "Feas.", "Ns", "V_min", "V_max", "Load%", "Time(s)"]
        else:
            header = ["Iter", "Description", "Cost ($)", "Feas.", "V_min", "V_max", "Load%", "Time(s)"]
        rows = [header]

        _NON_SIM = {"SWEEP", "EXPLORE", "ANALYSIS", "COMPLETE", "CONTINGENCY"}

        for e in session.journal.entries:
            if e.convergence_status in _NON_SIM:
                feas_text = "—"
                if e.convergence_status == "CONTINGENCY":
                    cost_cell = "SCREEN"
                    meta = e.contingency_meta or getattr(e, "reserve_meta", None) or {}
                    pc = meta.get("passed_count", 0)
                    fc = meta.get("failed_count", 0)
                    desc = f"{e.description[:20]} — {pc}/{pc + fc} passed"[:40]
                elif e.convergence_status == "SWEEP" and e.candidate_count:
                    cost_cell = "SWEEP"
                    n_feas = len(e.feasible_buses or [])
                    desc = f"{e.description[:22]} — {n_feas}/{e.candidate_count} feasible"[:40]
                else:
                    cost_cell = e.convergence_status
                    desc = e.description[:40]
            else:
                if e.feasibility_detail == "marginal":
                    feas_text = "Marg"
                elif e.feasible:
                    feas_text = "Y"
                else:
                    feas_text = "N"
                cost_cell = f"${e.objective_value:,.2f}" if e.objective_value is not None else "FAILED"
                desc = e.description[:40]
            row = [
                str(e.iteration),
                desc,
                cost_cell,
                feas_text,
            ]
            if is_tcopflow:
                row.append(str(e.num_steps) if e.num_steps > 0 else "—")
            if is_sopflow:
                row.append(str(e.num_scenarios) if e.num_scenarios > 0 else "—")
            row.extend([
                f"{e.voltage_min:.3f}" if e.voltage_min > 0 else "—",
                f"{e.voltage_max:.3f}" if e.voltage_max > 0 else "—",
                f"{e.max_line_loading_pct:.1f}" if e.max_line_loading_pct > 0 else "—",
                f"{e.elapsed_seconds:.1f}",
            ])
            rows.append(row)

        if is_tcopflow:
            col_widths = [1.2 * cm, 4.5 * cm, 2.8 * cm, 1.2 * cm, 1.0 * cm, 1.8 * cm, 1.8 * cm, 1.5 * cm, 1.5 * cm]
        elif is_sopflow:
            col_widths = [1.2 * cm, 4.5 * cm, 2.8 * cm, 1.2 * cm, 1.0 * cm, 1.8 * cm, 1.8 * cm, 1.5 * cm, 1.5 * cm]
        else:
            col_widths = [1.2 * cm, 5.5 * cm, 2.8 * cm, 1.2 * cm, 1.8 * cm, 1.8 * cm, 1.5 * cm, 1.5 * cm]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(table)

        # Add EMPAR warning if any iteration used EMPAR solver
        has_empar = any(
            e.solver.strip().upper() == "EMPAR"
            for e in session.journal.entries
        )
        if has_empar:
            empar_warning = (
                "⚠ EMPAR solver was used. EMPAR always reports CONVERGED and does "
                "not verify N-1 security. Results reflect base-case feasibility only, "
                "not N-1-secure loadability. For accurate N-1 security analysis, use "
                "the IPOPT solver."
            )
            elements.append(Paragraph(empar_warning, s["warning"] if "warning" in s else s["body"]))

        # Add marginal convergence note if any iteration was marginal
        has_marginal = any(
            e.feasibility_detail == "marginal"
            for e in session.journal.entries
        )
        if has_marginal:
            marginal_note = (
                "Note: Iterations marked 'Marg' had marginal convergence "
                "(solver did not fully converge but no constraint violations were "
                "detected). These results should be treated with caution."
            )
            elements.append(Paragraph(marginal_note, s["body"]))

        return elements

    # ── TCOPFLOW Temporal Analysis ──────────────────────────────────────

    def _build_tcopflow_temporal_section(
        self,
        session: SearchSession,
        period_data: list[dict],
    ) -> list:
        """Build a TCOPFLOW temporal analysis section for the PDF report.

        Shows how generation, load, voltage, and line loading vary across
        the time horizon, demonstrating the influence of ramp coupling and
        temporal load profiles on the solution.
        """
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Temporal Analysis (TCOPFLOW)", s["heading1"]))

        num_steps = len(period_data)
        num_steps_journal = max((e.num_steps for e in session.journal.entries if e.num_steps > 0), default=0)
        dT = getattr(session, "_tcopflow_dT_min", 0.0)
        duration = getattr(session, "_tcopflow_duration_min", 0.0)
        coupling = getattr(session, "_tcopflow_is_coupling", True)

        coupling_str = "enabled" if coupling else "disabled"
        elements.append(Paragraph(
            f"TCOPFLOW solved a {num_steps}-period optimization over "
            f"{duration:.0f} minutes (dT = {dT:.0f} min) with "
            f"generator ramp coupling {coupling_str}. The table below shows "
            f"how network conditions evolve across the time horizon.",
            s["body"],
        ))
        elements.append(Spacer(1, 0.5 * cm))

        # Per-period table
        header = ["Period", "Load (MW)", "Gen (MW)", "V_min (pu)", "V_max (pu)", "Max Load (%)", "Losses (MW)"]
        rows = [header]
        for p in period_data:
            rows.append([
                str(p["period"]),
                f"{p['total_load_mw']:.1f}",
                f"{p['total_gen_mw']:.1f}",
                f"{p['voltage_min']:.3f}",
                f"{p['voltage_max']:.3f}",
                f"{p['max_line_loading_pct']:.1f}",
                f"{p['losses_mw']:.1f}",
            ])

        col_widths = [1.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.5 * cm))

        # Temporal trend analysis
        if len(period_data) >= 2:
            first = period_data[0]
            last = period_data[-1]
            load_delta = last["total_load_mw"] - first["total_load_mw"]
            gen_delta = last["total_gen_mw"] - first["total_gen_mw"]
            vmin_delta = last["voltage_min"] - first["voltage_min"]
            vmax_delta = last["voltage_max"] - first["voltage_max"]
            loading_delta = last["max_line_loading_pct"] - first["max_line_loading_pct"]

            peak_load = max(period_data, key=lambda p: p["total_load_mw"])
            worst_vmin = min(period_data, key=lambda p: p["voltage_min"])
            worst_loading = max(period_data, key=lambda p: p["max_line_loading_pct"])

            lines = [
                f"Load change: {first['total_load_mw']:.1f} → {last['total_load_mw']:.1f} MW ({load_delta:+.1f} MW across horizon)",
                f"Generation change: {first['total_gen_mw']:.1f} → {last['total_gen_mw']:.1f} MW ({gen_delta:+.1f} MW)",
                f"Voltage minimum: {first['voltage_min']:.3f} → {last['voltage_min']:.3f} pu ({vmin_delta:+.4f} pu)",
                f"Voltage maximum: {first['voltage_max']:.3f} → {last['voltage_max']:.3f} pu ({vmax_delta:+.4f} pu)",
                f"Max line loading: {first['max_line_loading_pct']:.1f}% → {last['max_line_loading_pct']:.1f}% ({loading_delta:+.1f}%)",
                "",
                f"Peak demand: period {peak_load['period']} ({peak_load['total_load_mw']:.1f} MW)",
                f"Worst voltage: period {worst_vmin['period']} (Vmin = {worst_vmin['voltage_min']:.3f} pu)",
                f"Worst line loading: period {worst_loading['period']} ({worst_loading['max_line_loading_pct']:.1f}%)",
            ]
            if coupling:
                lines.append(
                    "Generator ramp coupling was enabled — the solver must respect "
                    "generator output change limits between consecutive periods."
                )
            for line in lines:
                elements.append(Paragraph(self._escape_xml(line), s["body"]))

        # Worst-case period identification
        elements.append(Spacer(1, 0.3 * cm))
        worst_period = min(
            period_data,
            key=lambda p: p["voltage_min"] * 1000 - p["max_line_loading_pct"],
        )
        elements.append(Paragraph(
            f"The worst-case period is <b>period {worst_period['period']}</b> "
            f"(Vmin = {worst_period['voltage_min']:.3f} pu, "
            f"max loading = {worst_period['max_line_loading_pct']:.1f}%). "
            f"Overall feasibility is determined by the worst period.",
            s["body"],
        ))

        return elements

    def _build_steering_section(self, steering_history: list[dict]) -> list:
        """Build a 'Steering History' section for the PDF report."""
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Steering History", s["heading1"]))
        elements.append(Paragraph(
            f"The user injected {len(steering_history)} steering directive(s) "
            "during the search to guide the LLM's decision-making.",
            s["body"],
        ))
        elements.append(Spacer(1, 0.5 * cm))

        header = ["Iter", "Mode", "Directive"]
        rows: list = [header]
        for item in steering_history:
            rows.append([
                str(item.get("iteration", "—")),
                item.get("mode", "augment").upper(),
                item.get("directive", "")[:100],
            ])

        table = Table(rows, colWidths=[1.5 * cm, 2.5 * cm, 13 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (0, 0), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)

        return elements

    # ── Multi-Objective Section ──────────────────────────────────────────

    def _build_multi_objective_section(
        self,
        session: SearchSession,
        goal_classification: Optional[dict] = None,
    ) -> list:
        s = self._styles
        elements: list = []

        elements.append(Paragraph("Multi-Objective Tracking", s["heading1"]))
        elements.append(Spacer(1, 5 * mm))

        registry = session.journal.objective_registry
        obj_data = registry.to_dict_list()

        if obj_data:
            header = ["Objective", "Direction", "Priority", "Since Iter", "Source"]
            rows: list = [header]
            for obj in obj_data:
                dir_str = obj["direction"]
                if obj["direction"] == "constraint" and obj.get("threshold") is not None:
                    dir_str = f"constraint (\u2264 {obj['threshold']})"
                rows.append([
                    Paragraph(obj["name"], s["body"]),
                    dir_str,
                    obj["priority"],
                    str(obj.get("introduced_at", 0)),
                    obj.get("source", "initial"),
                ])

            col_widths = [5 * cm, 3.5 * cm, 2.5 * cm, 2 * cm, 2 * cm]
            obj_table = Table(rows, colWidths=col_widths)
            obj_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(obj_table)
            elements.append(Spacer(1, 0.5 * cm))

        # Multi-objective trend chart
        mo_chart = multi_objective_trend_chart(session.journal)
        if mo_chart is not None:
            img_bytes = _export_chart_image(mo_chart, width_px=700, height_px=450)
            if img_bytes:
                elements.append(Image(io.BytesIO(img_bytes), width=16 * cm, height=10 * cm))
                elements.append(Spacer(1, 0.5 * cm))

        # Tradeoff summary from goal classification
        if goal_classification and goal_classification.get("tradeoff_summary"):
            elements.append(Paragraph(
                f"<b>Tradeoff Analysis:</b> {goal_classification['tradeoff_summary']}",
                s["body"],
            ))
            elements.append(Spacer(1, 3 * mm))

        if goal_classification and goal_classification.get("recommended_solutions"):
            recs = goal_classification["recommended_solutions"]
            if len(recs) > 1:
                elements.append(Paragraph(
                    f"<b>Recommended tradeoff solutions:</b> iterations {recs}",
                    s["body"],
                ))

        return elements

    # ── PFLOW vs OPFLOW Benchmark ────────────────────────────────────────

    def _build_benchmark_section(self, benchmark_result: dict) -> list:
        """Build a PFLOW vs OPFLOW benchmark section for the PDF report."""
        s = self._styles
        elements: list = []
        elements.append(Paragraph("PFLOW vs OPFLOW Benchmark", s["heading1"]))
        elements.append(Paragraph(
            "Comparison of LLM-driven PFLOW search results against the "
            "OPFLOW optimal solution. OPFLOW finds the mathematically optimal "
            "dispatch; PFLOW uses Newton-Raphson power flow with LLM-guided "
            "modifications, so cost is computed from the resulting dispatch "
            "using generator cost curves.",
            s["body"],
        ))
        elements.append(Spacer(1, 0.5 * cm))

        if benchmark_result.get("error"):
            elements.append(Paragraph(
                f"<b>Benchmark error:</b> {self._escape_xml(benchmark_result['error'])}",
                s["body"],
            ))
            return elements

        # Key metrics table
        header = ["Metric", "Value"]
        rows = [header]

        if benchmark_result.get("opflow_converged"):
            rows.append(["OPFLOW converged", "Yes"])
        else:
            rows.append(["OPFLOW converged", "No"])

        opflow_obj = benchmark_result.get("opflow_objective")
        if opflow_obj is not None:
            rows.append(["OPFLOW optimal cost", f"${opflow_obj:,.2f}"])

        pflow_cost = benchmark_result.get("pflow_best_computed_cost")
        if pflow_cost is not None:
            rows.append(["Best PFLOW computed cost", f"${pflow_cost:,.2f}"])

        cost_gap_pct = benchmark_result.get("cost_gap_pct")
        if cost_gap_pct is not None:
            sign = "+" if cost_gap_pct >= 0 else ""
            rows.append(["Cost gap", f"{sign}{cost_gap_pct:.2f}%"])

        cost_gap_abs = benchmark_result.get("cost_gap_abs")
        if cost_gap_abs is not None:
            sign = "+" if cost_gap_abs >= 0 else ""
            rows.append(["Cost difference", f"{sign}${cost_gap_abs:,.2f}"])

        col_widths = [10 * cm, 7 * cm]
        table = Table(rows, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.5 * cm))

        # Dispatch comparison table
        dispatch_comparison = benchmark_result.get("dispatch_comparison", [])
        if dispatch_comparison:
            elements.append(Paragraph("Dispatch Comparison (top 10 by |delta|)", s["heading2"]))
            dc_header = ["Gen Bus", "Fuel", "OPFLOW MW", "PFLOW MW", "Delta MW", "% of Pmax"]
            dc_rows = [dc_header]
            for dc in dispatch_comparison[:10]:
                pct_pmax = (dc["delta"] / dc["opflow_pmax"] * 100) if dc["opflow_pmax"] > 0 else 0
                sign = "+" if dc["delta"] >= 0 else ""
                dc_rows.append([
                    str(dc["bus"]),
                    dc["fuel"],
                    f"{dc['opflow_pg']:.2f}",
                    f"{dc['pflow_pg']:.2f}",
                    f"{sign}{dc['delta']:.2f}",
                    f"{sign}{pct_pmax:.1f}%",
                ])
            dc_widths = [2 * cm, 2.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3.5 * cm]
            dc_table = Table(dc_rows, colWidths=dc_widths)
            dc_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(dc_table)
            elements.append(Spacer(1, 0.5 * cm))

        # Loadability comparison
        loadability = benchmark_result.get("loadability")
        if loadability:
            elements.append(Paragraph("Loadability Comparison", s["heading2"]))
            load_rows = [
                ["Metric", "Value"],
            ]
            if loadability.get("opflow_max_factor") is not None:
                load_rows.append(["OPFLOW max load factor", f"{loadability['opflow_max_factor']:.4f}"])
            if loadability.get("pflow_max_factor") is not None:
                load_rows.append(["PFLOW max load factor", f"{loadability['pflow_max_factor']:.4f}"])
            if loadability.get("gap_pct") is not None:
                load_rows.append(["Boundary gap", f"{loadability['gap_pct']:+.2f}%"])
            if loadability.get("detail"):
                load_rows.append(["Detail", loadability["detail"]])

            load_table = Table(load_rows, colWidths=[8 * cm, 9 * cm])
            load_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(load_table)

        return elements

    # ── SOPFLOW Stochastic Analysis ──────────────────────────────────────

    def _build_sopflow_stochastic_section(
        self,
        session: SearchSession,
        num_scenarios: int,
    ) -> list:
        """Build a SOPFLOW stochastic analysis section for the PDF report.

        Shows the number of wind scenarios and key stochastic metrics.
        """
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Stochastic Analysis (SOPFLOW)", s["heading1"]))

        solver = session.journal.entries[0].solver if session.journal.entries else "IPOPT"
        elements.append(Paragraph(
            f"SOPFLOW solved a two-stage stochastic optimization across "
            f"<b>{num_scenarios}</b> wind generation scenarios using the "
            f"<b>{solver}</b> solver. The first-stage dispatch must satisfy "
            f"network constraints across all scenarios simultaneously, ensuring "
            f"robustness against wind generation uncertainty.",
            s["body"],
        ))
        elements.append(Spacer(1, 0.5 * cm))

        base = session.journal.entries[0] if session.journal.entries else None
        if base and base.feasible:
            summary_data = [
                ["Metric", "Value"],
                ["Scenarios", str(num_scenarios)],
                ["Solver", solver],
                ["Objective (base cost)", f"${base.objective_value:,.2f}"],
                ["V_min", f"{base.voltage_min:.3f} pu"],
                ["V_max", f"{base.voltage_max:.3f} pu"],
                ["Max line loading", f"{base.max_line_loading_pct:.1f}%"],
                ["Violations", str(base.violations_count)],
                ["Total generation", f"{base.total_gen_mw:.2f} MW"],
                ["Total load", f"{base.total_load_mw:.2f} MW"],
            ]
            if base.feasibility_detail:
                summary_data.append(["Feasibility", base.feasibility_detail])

            col_widths = [8 * cm, 8 * cm]
            summary_table = Table(summary_data, colWidths=col_widths)
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 1), (-1, -1), self._font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(summary_table)

        return elements
