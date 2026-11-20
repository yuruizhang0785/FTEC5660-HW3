#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any


MODEL_NAME = "DeepSeek-V4.1-Flash-1017"
DEFAULT_API_BASE_URL = "https://www.moyu.cn/v1"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50
JUDGMENT_LENSES = (
    "Inductive: look for a consistent pattern across the supplied measures and evidence.",
    "Abductive and causal: assess whether recent product commercialization plausibly reflects effective R&D, without claiming causation from correlation alone.",
    "Analogical: compare the company with its industry benchmark and consider how strong or weak the difference is in context.",
    "Probabilistic: weigh evidence quality, consistency, and proximity to the benchmark; express uncertainty in the rationale.",
    "Counterfactual: consider whether a modest change in the supplied evidence would change the conclusion, without performing new arithmetic.",
)
MIN_VALID_JUDGMENTS = len(JUDGMENT_LENSES) // 2 + 1

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [1, 3, 7]


def build_agent() -> Any:
    """One DeepSeek model judges the interview evidence for Fisher point 7."""
    import os
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_API_BASE_URL),
        max_retries=3,
    )


def _single_sql_row(tool: Any, query: str) -> dict[str, float]:
    """Parse one aggregate row returned by the provided SQL tool."""
    lines = tool.invoke({"query": query}).strip().splitlines()
    if len(lines) != 2 or lines[0].startswith("SQL error"):
        raise ValueError(f"Unexpected SQL result: {lines[:3]}")
    names = [part.strip() for part in lines[0].split("|")]
    values = [part.strip() for part in lines[1].split("|")]
    if len(names) != len(values):
        raise ValueError("SQL result has inconsistent columns")
    return {key: float(value) for key, value in zip(names, values)}


def _fit_productivity_states(values: list[float]) -> tuple[float, float, float, float, float]:
    """Fit low and high productivity regimes on log industry-relative output."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    low = sum(ordered[:middle]) / middle
    high = sum(ordered[middle:]) / (len(ordered) - middle)
    spread = max(sum((x - sum(ordered) / len(ordered)) ** 2 for x in ordered) / len(ordered), 0.0025)
    prior = 0.5
    for _ in range(100):
        weights = []
        for x in values:
            a = prior * math.exp(-((x - high) ** 2) / (2 * spread))
            b = (1 - prior) * math.exp(-((x - low) ** 2) / (2 * spread))
            weights.append(a / (a + b) if a + b else 0.5)
        high_n = sum(weights)
        low_n = len(values) - high_n
        if min(high_n, low_n) < 1:
            break
        high = sum(w * x for w, x in zip(weights, values)) / high_n
        low = sum((1 - w) * x for w, x in zip(weights, values)) / low_n
        spread = max(sum(w * (x - high) ** 2 + (1 - w) * (x - low) ** 2 for w, x in zip(weights, values)) / len(values), 0.0025)
        prior = high_n / len(values)
    return low, high, spread, prior, (low + high) / 2


def _high_state_probability(x: float, state: tuple[float, float, float, float, float]) -> float:
    low, high, spread, prior, _ = state
    a = prior * math.exp(-((x - high) ** 2) / (2 * spread))
    b = (1 - prior) * math.exp(-((x - low) ** 2) / (2 * spread))
    return a / (a + b) if a + b else 0.5


async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Use the supplied tools, then propagate measurement uncertainty."""
    import random

    chosen = check_chosen_points(CHOSEN_POINTS)
    if chosen != [1, 3, 7]:
        raise ValueError("This analysis implements Fisher points 1, 3 and 7")
    semaphore = asyncio.Semaphore(4)

    async def gather_one(name: str, company_tools: list[Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                tools = {item.name: item for item in company_tools}
                sql = tools["sql_query"]
                growth = _single_sql_row(sql, """
                    WITH revenue AS (
                      SELECT SUM(CASE WHEN order_date >= '2023-01-01' AND order_date < '2024-01-01' THEN amount ELSE 0 END) AS prior,
                             SUM(CASE WHEN order_date >= '2025-01-01' AND order_date < '2026-01-01' THEN amount ELSE 0 END) AS current
                      FROM orders)
                    SELECT revenue.prior, revenue.current, i.revenue_growth_2y_pct AS benchmark
                    FROM revenue, company c JOIN industry_medians i ON i.industry = c.industry
                """)
                product = _single_sql_row(sql, """
                    WITH sales AS (
                      SELECT SUM(o.amount) AS new_sales, COUNT(*) AS order_count,
                             SUM(o.amount * o.amount) AS sum_squares
                      FROM orders o JOIN products p ON o.product = p.product
                      WHERE o.order_date >= '2025-01-01' AND o.order_date < '2026-01-01'
                        AND p.launch_date >= '2023-01-01' AND p.launch_date < '2026-01-01'),
                    spending AS (
                      SELECT SUM(research_development) AS rd FROM monthly_costs
                      WHERE month >= '2025-01' AND month < '2026-01')
                    SELECT sales.new_sales, sales.order_count, sales.sum_squares,
                           spending.rd, i.new_product_sales_per_rd AS benchmark
                    FROM sales, spending, company c JOIN industry_medians i ON i.industry = c.industry
                """)
                if min(growth["prior"], product["rd"], product["benchmark"], product["order_count"]) <= 0:
                    raise ValueError("Missing or nonpositive financial denominator")
                growth_pct = 100 * (growth["current"] / growth["prior"] - 1)
                relative = product["new_sales"] / product["rd"] / product["benchmark"]
                if relative <= 0:
                    raise ValueError("Nonpositive new-product productivity")
                answer = tools["interview"].invoke({"role": "current_employee", "topic": "working_conditions"})
                response = await agent.ainvoke([{"role": "system", "content":
                    "Judge Fisher point 7 from this employee statement alone. PASS for specific positive employee relations evidence; FAIL for clear problems or absent positive evidence. Return JSON with verdict (PASS/FAIL) and a brief rationale. Do not invent facts."},
                    {"role": "user", "content": answer}])
                raw = response.content
                if isinstance(raw, list):
                    raw = "".join(str(part.get("text", "")) for part in raw if isinstance(part, dict))
                raw = str(raw)
                decision = json.JSONDecoder().raw_decode(raw[raw.find("{"):])[0]
                employee = normalize_verdict(decision.get("verdict"))
                if employee is None:
                    raise ValueError("Invalid employee-relations verdict")
                return {"growth_pct": growth_pct, "growth_benchmark": growth["benchmark"],
                        "product": product, "relative": relative, "employee": employee,
                        "employee_reason": str(decision.get("rationale", "")).strip()}
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

    names = list(companies)
    observations = await asyncio.gather(*(gather_one(name, companies[name]) for name in names))
    evidence = dict(zip(names, observations))
    valid_logs = [math.log(item["relative"]) for item in observations if "relative" in item]
    state = _fit_productivity_states(valid_logs) if len(valid_logs) >= 4 else None
    output: dict[str, dict[str, Any]] = {}
    for name, item in evidence.items():
        if "error" in item or state is None:
            output[name] = {"points": {point: None for point in chosen},
                            "summary": item.get("error", "Too few companies for relative assessment")}
            continue
        product = item["product"]
        count = product["order_count"]
        average = product["new_sales"] / count
        variance = max(product["sum_squares"] / count - average * average, 0)
        relative_se = math.sqrt(variance / count) / average if average else 0
        # Orders are observed transactions. Resampling their aggregate reflects
        # measurement variation; it does not prove R&D caused the sales.
        rng = random.Random(zlib.crc32(name.encode("utf-8")))
        probabilities = []
        for _ in range(400):
            sampled_log = math.log(item["relative"]) + rng.gauss(0, relative_se)
            probabilities.append(_high_state_probability(sampled_log, state))
        support = sum(probabilities) / len(probabilities)
        points = {1: "PASS" if item["growth_pct"] > item["growth_benchmark"] else "FAIL",
                  3: "PASS" if support > 0.5 else "FAIL", 7: item["employee"]}
        ratio = item["relative"] * product["benchmark"]
        output[name] = {"points": points,
            "summary": (f"Point 1: {points[1]}; two-year revenue growth {item['growth_pct']:.2f}% "
                        f"versus industry median {item['growth_benchmark']:.2f}%. "
                        f"Point 3: {points[3]}; new-product sales per FY2025 R&D {ratio:.2f} "
                        f"versus industry median {product['benchmark']:.2f}; "
                        f"high-productivity state support {support:.1%} under order variation. "
                        "The sales/R&D timing mismatch limits causal attribution. "
                        f"Point 7: {points[7]}; {item['employee_reason']}")}
    return output


# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())
