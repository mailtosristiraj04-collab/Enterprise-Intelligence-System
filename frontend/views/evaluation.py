"""Evaluation dashboard backed by saved FastAPI reports."""

import streamlit as st

from frontend.api.client import APIClient, APIClientError


def _metric_table(report: dict) -> None:
    configurations = report.get("configurations", {})
    k_values = report.get("k_values", [])
    for k in k_values:
        st.subheader(f"Retrieval metrics @ {k}")
        rows = []
        for name, result in configurations.items():
            metrics = result.get("aggregate", {}).get(str(k), {})
            rows.append({
                "Configuration": name,
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "MRR": metrics.get("mrr"),
                "nDCG": metrics.get("ndcg"),
            })
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)


def render(client: APIClient) -> None:
    st.title("Evaluation")
    st.caption("Compare retrieval configurations using manually verified relevance labels.")
    try:
        runs = client.evaluation_runs()
    except APIClientError as exc:
        st.error(str(exc))
        return
    with st.expander("Run evaluation", expanded=not runs):
        st.info("The backend dataset is kept separate from production documents. Add verified JSONL labels before running.")
        if st.button("Run evaluation", type="primary"):
            with st.spinner("Running retrieval benchmark…"):
                try:
                    report = client.run_evaluation()
                    st.session_state["latest_evaluation"] = report
                    st.success(f"Completed {report.get('sample_count', 0)} evaluation samples.")
                    st.rerun()
                except APIClientError as exc:
                    st.error(str(exc))
    report = st.session_state.get("latest_evaluation")
    if report is None and runs:
        selected = st.selectbox("Historical run", [item["run_id"] for item in runs])
        try:
            report = client.evaluation_run(selected)
        except APIClientError as exc:
            st.error(str(exc))
    if report is None:
        st.info("No evaluation runs are available yet.")
        return
    st.caption(f"Run {report.get('run_id')} · {report.get('sample_count', 0)} samples")
    _metric_table(report)
    st.subheader("Latency")
    latency_rows = []
    for name, result in report.get("configurations", {}).items():
        summary = result.get("latency", {}).get("total_ms", {})
        latency_rows.append({
            "Configuration": name,
            "Average (ms)": summary.get("average_ms"),
            "Median (ms)": summary.get("median_ms"),
            "P95 (ms)": summary.get("p95_ms"),
        })
    if latency_rows:
        st.dataframe(latency_rows, use_container_width=True, hide_index=True)
