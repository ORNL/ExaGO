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
    )
except ModuleNotFoundError:
    from launcher.charts import (
        convergence_chart, voltage_range_chart, voltage_profile_chart,
        generator_dispatch_chart, line_loading_chart, multi_objective_trend_chart,
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
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    bold_paths = [
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

        v_min = session.enforced_vmin if session.enforced_vmin is not None else 0.95
        v_max = session.enforced_vmax if session.enforced_vmax is not None else 1.05

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
        )

        story: list = []
        story.extend(self._build_title_page(session, goal_type=goal_type))
        story.append(PageBreak())
        story.extend(self._build_executive_summary(
            session, summary_text, goal_classification=gc,
        ))
        story.append(PageBreak())
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
    ) -> list:
        s = self._styles
        elements: list = []
        elements.append(Spacer(1, 6 * cm))
        elements.append(Paragraph("AgentiGrid Search Report", s["title"]))
        if goal_type:
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

        for e in session.journal.entries:
            if e.feasibility_detail == "marginal":
                feas_text = "Marg"
            elif e.feasible:
                feas_text = "Y"
            else:
                feas_text = "N"
            row = [
                str(e.iteration),
                e.description[:40],
                f"${e.objective_value:,.2f}" if e.objective_value is not None else "FAILED",
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
