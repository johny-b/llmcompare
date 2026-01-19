"""DataFrame viewer for browsing question results.

Spawns a local Streamlit server to interactively browse (api_kwargs, answer) pairs.

Usage:
    from llmcomp import Question
    
    question = Question.create(...)
    df = question.df(models)
    Question.view(df)
"""

import json
import os
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

# Streamlit imports are inside functions to avoid import errors when streamlit isn't installed


def render_dataframe(
    df: "pd.DataFrame",
    sort_by: str | None = None,
    sort_ascending: bool = True,
    open_browser: bool = True,
    port: int = 8501,
) -> None:
    """Launch a Streamlit viewer for the DataFrame.
    
    Args:
        df: DataFrame with at least 'api_kwargs' and 'answer' columns.
            Other columns (model, group, etc.) are displayed as metadata.
        sort_by: Column name to sort by initially. If None, keeps original order.
        sort_ascending: Sort order. Default: True (ascending).
        open_browser: If True, automatically open the viewer in default browser.
        port: Port to run the Streamlit server on.
    
    Raises:
        ValueError: If required columns are missing.
    """
    # Validate required columns
    if "api_kwargs" not in df.columns:
        raise ValueError("DataFrame must have an 'api_kwargs' column")
    if "answer" not in df.columns:
        raise ValueError("DataFrame must have an 'answer' column")
    if sort_by is not None and sort_by not in df.columns:
        raise ValueError(f"sort_by column '{sort_by}' not found in DataFrame")
    
    # Save DataFrame to a temp file
    temp_dir = tempfile.mkdtemp(prefix="llmcomp_viewer_")
    temp_path = os.path.join(temp_dir, "data.jsonl")
    
    # Convert DataFrame to JSONL
    with open(temp_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            f.write(json.dumps(row_dict, default=str) + "\n")
    
    url = f"http://localhost:{port}"
    print(f"Starting viewer at {url}")
    print(f"Data file: {temp_path}")
    print("Press Ctrl+C to stop the server.\n")
    
    if open_browser:
        # Open browser after a short delay to let server start
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    
    # Launch Streamlit
    viewer_path = Path(__file__).resolve()
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(viewer_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--",  # Separator for script args
        temp_path,
        sort_by or "",  # Empty string means no sorting
        "asc" if sort_ascending else "desc",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except OSError:
            pass


# =============================================================================
# Streamlit App (runs when this file is executed by streamlit)
# =============================================================================

def _get_data_path() -> str | None:
    """Get data file path from command line args."""
    # Args after -- are passed to the script
    if len(sys.argv) > 1:
        return sys.argv[1]
    return None


def _get_initial_sort() -> tuple[str | None, bool]:
    """Get initial sort settings from command line args."""
    sort_by = None
    sort_ascending = True
    
    if len(sys.argv) > 2:
        sort_by = sys.argv[2] if sys.argv[2] else None
    if len(sys.argv) > 3:
        sort_ascending = sys.argv[3] != "desc"
    
    return sort_by, sort_ascending


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _display_messages(messages: list[dict[str, str]]) -> None:
    """Display a list of chat messages in Streamlit chat format."""
    import streamlit as st
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Map roles to streamlit chat_message roles
        if role == "system":
            with st.chat_message("assistant", avatar="⚙️"):
                st.markdown("**System**")
                st.text(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.text(content)
        else:  # user or other
            with st.chat_message("user"):
                st.text(content)


def _display_answer(answer: Any, label: str | None = None) -> None:
    """Display the answer, handling different types."""
    import streamlit as st
    
    if label:
        st.markdown(f"**{label}**")
    
    if isinstance(answer, dict):
        # For NextToken questions, answer is {token: probability}
        # Sort by probability descending
        sorted_items = sorted(answer.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0)
        # Display as a table-like format
        for token, prob in sorted_items[:20]:  # Show top 20
            if isinstance(prob, float):
                st.text(f"  {token!r}: {prob:.4f}")
            else:
                st.text(f"  {token!r}: {prob}")
    elif isinstance(answer, str):
        st.text(answer)
    else:
        st.text(str(answer))


def _display_metadata(row: dict[str, Any], exclude_keys: set[str]) -> None:
    """Display metadata columns."""
    import streamlit as st
    
    metadata = {k: v for k, v in row.items() if k not in exclude_keys}
    if metadata:
        with st.expander("Metadata", expanded=False):
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    st.markdown(f"**{key}:**")
                    st.json(value)
                else:
                    st.markdown(f"**{key}:** {value}")


def _search_items(items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Filter items by search query.
    
    Supports:
        - Regular search: "foo" - includes items containing "foo"
        - Negative search: "-foo" - excludes items containing "foo"
        - Combined: "foo -bar" - items with "foo" but not "bar"
    """
    if not query:
        return items
    
    # Parse query into positive and negative terms
    terms = query.split()
    positive_terms = []
    negative_terms = []
    
    for term in terms:
        if term.startswith("-") and len(term) > 1:
            negative_terms.append(term[1:].lower())
        else:
            positive_terms.append(term.lower())
    
    results = []
    
    for item in items:
        # Build searchable text from item
        api_kwargs = item.get("api_kwargs", {})
        messages = api_kwargs.get("messages", []) if isinstance(api_kwargs, dict) else []
        messages_text = " ".join(m.get("content", "") for m in messages)
        
        answer = item.get("answer", "")
        answer_text = str(answer) if not isinstance(answer, str) else answer
        
        all_text = messages_text + " " + answer_text
        all_text += " " + " ".join(str(v) for v in item.values() if isinstance(v, str))
        all_text_lower = all_text.lower()
        
        # Check positive terms (all must match)
        if positive_terms and not all(term in all_text_lower for term in positive_terms):
            continue
        
        # Check negative terms (none must match)
        if any(term in all_text_lower for term in negative_terms):
            continue
        
        results.append(item)
    
    return results


def _streamlit_main():
    """Main Streamlit app."""
    import streamlit as st
    
    st.set_page_config(
        page_title="llmcomp Viewer",
        page_icon="🔬",
        layout="wide",
    )
    
    st.title("🔬 llmcomp Viewer")
    
    # Get data path
    data_path = _get_data_path()
    if data_path is None or not os.path.exists(data_path):
        st.error("No data file provided or file not found.")
        st.info("Use `Question.render(df)` to launch the viewer with data.")
        return
    
    # Load data (cache in session state)
    cache_key = f"llmcomp_data_{data_path}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = _read_jsonl(data_path)
    
    items = st.session_state[cache_key]
    
    if not items:
        st.warning("No data to display.")
        return
    
    # Get sortable columns (numeric or string, exclude complex types)
    sortable_columns = ["(none)"]
    if items:
        for key, value in items[0].items():
            if key not in ("api_kwargs",) and isinstance(value, (int, float, str, type(None))):
                sortable_columns.append(key)
    
    # Initialize sort settings from command line args
    initial_sort_by, initial_sort_asc = _get_initial_sort()
    if "sort_by" not in st.session_state:
        st.session_state.sort_by = initial_sort_by if initial_sort_by in sortable_columns else "(none)"
        st.session_state.sort_ascending = initial_sort_asc
    
    # Initialize view index
    if "view_idx" not in st.session_state:
        st.session_state.view_idx = 0
    
    # Initialize secondary sort
    if "sort_by_2" not in st.session_state:
        st.session_state.sort_by_2 = "(none)"
        st.session_state.sort_ascending_2 = True
    
    # Search and sort controls
    col_search, col_sort, col_order = st.columns([3, 2, 1])
    
    with col_search:
        query = st.text_input("🔍 Search", placeholder="Filter... (use -term to exclude)")
    
    with col_sort:
        sort_by = st.selectbox(
            "Sort by",
            options=sortable_columns,
            index=sortable_columns.index(st.session_state.sort_by) if st.session_state.sort_by in sortable_columns else 0,
            key="sort_by_select",
        )
        if sort_by != st.session_state.sort_by:
            st.session_state.sort_by = sort_by
            st.session_state.view_idx = 0  # Reset to first item when sort changes
    
    with col_order:
        st.markdown("<br>", unsafe_allow_html=True)  # Align checkbox with selectbox
        sort_ascending = st.checkbox("Asc", value=st.session_state.sort_ascending, key="sort_asc_check")
        if sort_ascending != st.session_state.sort_ascending:
            st.session_state.sort_ascending = sort_ascending
            st.session_state.view_idx = 0
    
    # Secondary sort (only show if primary sort is selected)
    if st.session_state.sort_by and st.session_state.sort_by != "(none)":
        col_spacer, col_sort2, col_order2 = st.columns([3, 2, 1])
        with col_sort2:
            sort_by_2 = st.selectbox(
                "Then by",
                options=sortable_columns,
                index=sortable_columns.index(st.session_state.sort_by_2) if st.session_state.sort_by_2 in sortable_columns else 0,
                key="sort_by_select_2",
            )
            if sort_by_2 != st.session_state.sort_by_2:
                st.session_state.sort_by_2 = sort_by_2
                st.session_state.view_idx = 0
        with col_order2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align checkbox with selectbox
            sort_ascending_2 = st.checkbox("Asc", value=st.session_state.sort_ascending_2, key="sort_asc_check_2")
            if sort_ascending_2 != st.session_state.sort_ascending_2:
                st.session_state.sort_ascending_2 = sort_ascending_2
                st.session_state.view_idx = 0
    
    # Apply search
    filtered_items = _search_items(items, query)
    
    # Apply sorting (stable sort - secondary first, then primary)
    if st.session_state.sort_by and st.session_state.sort_by != "(none)" and filtered_items:
        sort_key_2 = st.session_state.sort_by_2 if st.session_state.sort_by_2 != "(none)" else None
        
        # Secondary sort first (stable sort preserves this ordering within primary groups)
        if sort_key_2:
            filtered_items = sorted(
                filtered_items,
                key=lambda x: (x.get(sort_key_2) is None, x.get(sort_key_2)),
                reverse=not st.session_state.sort_ascending_2,
            )
        
        # Primary sort
        sort_key = st.session_state.sort_by
        filtered_items = sorted(
            filtered_items,
            key=lambda x: (x.get(sort_key) is None, x.get(sort_key)),
            reverse=not st.session_state.sort_ascending,
        )
    
    if not filtered_items:
        st.warning(f"No results found for '{query}'")
        return
    
    # Clamp view index to valid range
    max_idx = len(filtered_items) - 1
    st.session_state.view_idx = max(0, min(st.session_state.view_idx, max_idx))
    
    # Navigation
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
    
    with col1:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.view_idx = max(0, st.session_state.view_idx - 1)
            st.rerun()
    
    with col2:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.view_idx = min(max_idx, st.session_state.view_idx + 1)
            st.rerun()
    
    with col3:
        # Jump to specific index
        new_idx = st.number_input(
            "Go to",
            min_value=1,
            max_value=len(filtered_items),
            value=st.session_state.view_idx + 1,
            step=1,
            label_visibility="collapsed",
        )
        if new_idx - 1 != st.session_state.view_idx:
            st.session_state.view_idx = new_idx - 1
            st.rerun()
    
    with col4:
        st.markdown(f"**{st.session_state.view_idx + 1}** of **{len(filtered_items)}**")
        if query:
            st.caption(f"({len(items)} total)")
    
    st.divider()
    
    # Display current item
    current = filtered_items[st.session_state.view_idx]
    
    # Main content in two columns
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.subheader("💬 Messages")
        api_kwargs = current.get("api_kwargs", {})
        messages = api_kwargs.get("messages", []) if isinstance(api_kwargs, dict) else []
        if messages:
            _display_messages(messages)
        else:
            st.info("No messages")
    
    with right_col:
        model_name = current.get("model", "Response")
        st.subheader(f"🤖 {model_name}")
        answer = current.get("answer")
        if answer is not None:
            _display_answer(answer, label=None)
        else:
            st.info("No answer")
        
        # Display judge columns if present
        judge_columns = [k for k in current.keys() if not k.startswith("_") and k not in {
            "api_kwargs", "answer", "question", "model", "group", "paraphrase_ix", "raw_answer"
        } and not k.endswith("_question") and not k.endswith("_raw_answer")]
        
        if judge_columns:
            st.markdown("---")
            for judge_col in judge_columns:
                value = current[judge_col]
                if isinstance(value, float):
                    st.markdown(f"**{judge_col}:** {value:.2f}")
                else:
                    st.markdown(f"**{judge_col}:** {value}")
    
    # Metadata at the bottom
    st.divider()
    # Show api_kwargs in metadata, but without messages (already displayed above)
    current_for_metadata = current.copy()
    if "api_kwargs" in current_for_metadata and isinstance(current_for_metadata["api_kwargs"], dict):
        api_kwargs_without_messages = {k: v for k, v in current_for_metadata["api_kwargs"].items() if k != "messages"}
        current_for_metadata["api_kwargs"] = api_kwargs_without_messages
    exclude_keys = {"answer", "question", "paraphrase_ix"} | set(judge_columns)
    _display_metadata(current_for_metadata, exclude_keys)
    
    # Keyboard navigation hint
    st.caption("💡 Tip: Use the navigation buttons or enter a number to jump to a specific row.")


# Entry point when run by Streamlit
if __name__ == "__main__":
    _streamlit_main()
