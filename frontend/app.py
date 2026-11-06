"""Streamlit application entry point for the foundation phase."""

import streamlit as st

from backend.core.config import get_settings


def main() -> None:
    """Render the foundation status page without exposing unfinished features."""
    settings = get_settings()
    st.set_page_config(page_title="Enterprise Knowledge Intelligence", page_icon=":books:")
    st.title("Enterprise Knowledge Intelligence System")
    st.info("The Phase 1 foundation is ready. Document intelligence features are not enabled yet.")
    st.subheader("Configuration")
    st.write(
        {
            "backend_host": settings.backend_host,
            "backend_port": settings.backend_port,
            "database": str(settings.get_database_path()),
        }
    )


if __name__ == "__main__":
    main()
