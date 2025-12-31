"""Tests for utility functions."""

import tempfile
from pathlib import Path

from llmcomp.utils import write_jsonl, read_jsonl


def test_write_and_read_jsonl():
    """Test writing and reading JSONL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        # Test data
        data = [
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "SF"},
            {"name": "Charlie", "age": 35, "city": "LA"},
        ]
        
        # Write JSONL
        write_jsonl(path, data)
        
        # Verify file exists
        assert path.exists()
        
        # Read JSONL
        loaded_data = read_jsonl(path)
        
        # Verify data matches
        assert loaded_data == data


def test_write_jsonl_creates_parent_dirs():
    """Test that write_jsonl creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "nested" / "test.jsonl"
        
        data = [{"key": "value"}]
        write_jsonl(path, data)
        
        assert path.exists()
        loaded_data = read_jsonl(path)
        assert loaded_data == data


def test_read_jsonl_skips_empty_lines():
    """Test that read_jsonl skips empty lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        
        # Write file with empty lines
        with open(path, "w") as f:
            f.write('{"a": 1}\n')
            f.write('\n')  # Empty line
            f.write('{"b": 2}\n')
            f.write('  \n')  # Whitespace line
            f.write('{"c": 3}\n')
        
        loaded_data = read_jsonl(path)
        assert loaded_data == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_write_read_empty_list():
    """Test writing and reading an empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.jsonl"
        
        write_jsonl(path, [])
        loaded_data = read_jsonl(path)
        
        assert loaded_data == []


def test_write_read_complex_data():
    """Test with complex nested data structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "complex.jsonl"
        
        data = [
            {
                "id": 1,
                "metadata": {"created": "2024-01-01", "tags": ["a", "b"]},
                "nested": {"deep": {"value": 42}},
            },
            {
                "id": 2,
                "list": [1, 2, 3, 4, 5],
                "bool": True,
                "null": None,
            },
        ]
        
        write_jsonl(path, data)
        loaded_data = read_jsonl(path)
        
        assert loaded_data == data