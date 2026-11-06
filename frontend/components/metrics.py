"""Metric card components."""

import streamlit as st


def metric_cards(values: list[tuple[str, str | int]]) -> None:
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values):
        with column:
            st.markdown(
                f'<div class="eki-card"><div class="eki-kicker">{label}</div>'
                f'<div style="font-size:1.8rem;font-weight:700;color:#18324b">{value}</div></div>',
                unsafe_allow_html=True,
            )
