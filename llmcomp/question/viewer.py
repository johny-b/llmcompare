"""DataFrame viewer for browsing question results.

Spawns a local Streamlit server to interactively browse (messages, answer) pairs.

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


def render_dataframe(df: "pd.DataFrame", open_browser: bool = True, port: int = 8501) -> None:
    """Launch a Streamlit viewer for the DataFrame.
    
    Args:
        df: DataFrame with at least 'messages' and 'answer' columns.
            Other columns (model, group, etc.) are displayed as metadata.
        open_browser: If True, automatically open the viewer in default browser.
        port: Port to run the Streamlit server on.
    
    Raises:
        ImportError: If streamlit is not installed.
        ValueError: If required columns are missing.
    """
    # Check if streamlit is installed
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise ImportError(
            "Streamlit is required for the viewer. Install it with:\n"
            "  pip install 'llmcomp[viewer]'\n"
            "or:\n"
            "  pip install streamlit"
        )
    
    # Validate required columns
    if "messages" not in df.columns:
        raise ValueError("DataFrame must have a 'messages' column")
    if "answer" not in df.columns:
        raise ValueError("DataFrame must have an 'answer' column")
    
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
    """Filter items by search query."""
    if not query:
        return items
    
    query_lower = query.lower()
    results = []
    
    for item in items:
        # Search in messages
        messages = item.get("messages", [])
        messages_text = " ".join(m.get("content", "") for m in messages)
        
        # Search in answer
        answer = item.get("answer", "")
        answer_text = str(answer) if not isinstance(answer, str) else answer
        
        # Search in all string fields
        all_text = messages_text + " " + answer_text
        all_text += " " + " ".join(str(v) for v in item.values() if isinstance(v, str))
        
        if query_lower in all_text.lower():
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
    
    # Initialize view index
    if "view_idx" not in st.session_state:
        st.session_state.view_idx = 0
    
    # Search
    query = st.text_input("🔍 Search", placeholder="Filter by content...")
    filtered_items = _search_items(items, query)
    
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
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.subheader("💬 Messages")
        messages = current.get("messages", [])
        if messages:
            _display_messages(messages)
        else:
            st.info("No messages")
    
    with right_col:
        st.subheader("🤖 Response")
        answer = current.get("answer")
        if answer is not None:
            _display_answer(answer, label=None)
        else:
            st.info("No answer")
        
        # Display judge columns if present
        judge_columns = [k for k in current.keys() if not k.startswith("_") and k not in {
            "messages", "answer", "question", "model", "group", "paraphrase_ix", "raw_answer"
        } and not k.endswith("_question") and not k.endswith("_raw_answer")]
        
        for judge_col in judge_columns:
            st.divider()
            _display_answer(current[judge_col], label=f"Judge: {judge_col}")
    
    # Metadata at the bottom
    st.divider()
    exclude_keys = {"messages", "answer", "question", "paraphrase_ix"} | set(judge_columns)
    _display_metadata(current, exclude_keys)
    
    # Keyboard navigation hint
    st.caption("💡 Tip: Use the navigation buttons or enter a number to jump to a specific row.")


# Entry point when run by Streamlit
if __name__ == "__main__":
    _streamlit_main()
