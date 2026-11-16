"""Great Expectations validation suite for walmart_master.

Encodes the schema contract between the pipeline and the Module 4 model.
A failed expectation halts promotion to data/processed/.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import great_expectations as gx


def build_suite(df: pd.DataFrame) -> gx.ValidationDefinition:
    """Build the Great Expectations validation definition against df."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="walmart_master_src")
    data_asset = data_source.add_dataframe_asset(name="walmart_master")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full")

    suite = context.suites.add(gx.ExpectationSuite(name="walmart_master_suite"))

    # Schema
    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=["Store","Dept","Date","Weekly_Sales","IsHoliday","is_return",
                    "Temperature","Fuel_Price","MarkDown1","MarkDown2","MarkDown3",
                    "MarkDown4","MarkDown5","CPI","Unemployment","Type","Size",
                    "week_of_year","month","year","weeks_to_year_end",
                    "sales_lag_1w","sales_lag_4w","sales_lag_52w",
                    "Type_A","Type_B","Type_C"],
        exact_match=False,
    ))
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(
        min_value=400_000, max_value=500_000,
    ))

    # Value ranges
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="Store", min_value=1, max_value=45,
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="Dept", min_value=1, max_value=99,
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Type", value_set=["A", "B", "C"],
    ))

    # Null tolerance
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
        column="Date",
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
        column="CPI",
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
        column="Unemployment",
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
        column="Type",
    ))

    # Temporal
    suite.add_expectation(gx.expectations.ExpectColumnValuesToMatchRegex(
        column="Date",
        regex=r"^\d{4}-\d{2}-\d{2}.*",
    ))

    validation_def = context.validation_definitions.add(gx.ValidationDefinition(
        name="walmart_master_validation",
        data=batch_definition,
        suite=suite,
    ))
    return validation_def, context


def run_expectations(parquet_path: Path) -> tuple[bool, dict]:
    df = pd.read_parquet(parquet_path)
    validation_def, context = build_suite(df)
    results = validation_def.run(batch_parameters={"dataframe": df})
    success_count = sum(1 for r in results.results if r.success)
    total = len(results.results)
    passed = results.success
    summary = {
        "rows": len(df),
        "expectations_total": total,
        "expectations_met": success_count,
        "passed": passed,
        "ts": datetime.utcnow().isoformat(),
    }
    # Persist a simple HTML/JSON report
    out_dir = Path("reports/expectations")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latest.json"
    json_path.write_text(json.dumps(summary, indent=2))
    render_html_report(results, summary, out_dir / "latest.html")
    return passed, summary


def render_html_report(results, summary: dict, out_path: Path) -> None:
    rows = []
    for r in results.results:
        rows.append(f"""
        <tr>
          <td class='{"pass" if r.success else "fail"}'>{"✓" if r.success else "✗"}</td>
          <td>{r.expectation_config.type}</td>
          <td class='mono'>{json.dumps(r.expectation_config.kwargs, default=str)[:140]}</td>
          <td class='right'>{r.result.get('observed_value', '')}</td>
        </tr>
        """)
    rows_html = "\n".join(rows)
    status_color = "#2E8B57" if summary["passed"] else "#C0392B"
    html = f"""
<!doctype html><html><head><meta charset='utf-8'><title>Great Expectations Report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #1A1A1A; }}
h1 {{ color: #002E6D; font-size: 20px; }}
.banner {{ background: #E8F6EC; border-left: 6px solid {status_color};
          padding: 14px 18px; margin-bottom: 18px; border-radius: 4px; }}
.banner b {{ color: {status_color}; font-size: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #E1E4E8; vertical-align: top; }}
th {{ background: #002E6D; color: white; text-align: left; }}
td.pass {{ color: #2E8B57; font-weight: bold; }}
td.fail {{ color: #C0392B; font-weight: bold; }}
td.mono {{ font-family: Menlo, Consolas, monospace; font-size: 11px; color: #55606B; }}
td.right {{ text-align: right; font-weight: bold; color: {status_color}; }}
</style></head><body>
<h1>Great Expectations | walmart_master_suite</h1>
<div class='banner'>
<b>{"✓ Suite passed" if summary['passed'] else "✗ Suite failed"}</b><br>
{summary['expectations_met']} of {summary['expectations_total']} expectations met
&nbsp;|&nbsp; {summary['rows']:,} rows
&nbsp;|&nbsp; {summary['ts'].split('T')[0]}
</div>
<table>
<tr><th style='width:40px'></th><th>Expectation</th><th>Arguments</th><th>Observed</th></tr>
{rows_html}
</table>
</body></html>
"""
    out_path.write_text(html)


if __name__ == "__main__":
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    pq = repo_root / "data" / "interim" / "walmart_master_prevalidated.parquet"
    passed, summary = run_expectations(pq)
    print(json.dumps(summary, indent=2))
