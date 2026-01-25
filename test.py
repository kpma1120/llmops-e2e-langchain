from collections.abc import Callable
from typing import Any

import pytest

import backend


class FakeDoc:
    """Fake document class used for mocking retriever results."""
    def __init__(self, source: str) -> None:
        self.metadata: dict[str, Any] = {"source": source}


class FakeChain:
    """Fake chain to mock rag_chain behavior."""
    def invoke(self, query: str) -> str:
        return "raw answer"


class FakeRefine:
    """Fake chain to mock refine_chain behavior."""
    def invoke(self, data: dict[str, Any]) -> str:
        return "refined answer"


class FakeRetriever:
    """Fake retriever to mock retriever behavior."""
    def __init__(self, docs: list[Any]) -> None:
        self._docs = docs

    def invoke(self, query: str) -> list[Any]:
        return self._docs


@pytest.fixture
def setup_backend(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[Any]], None]:
    """
    Fixture to replace backend pipelines with fake implementations.

    This fixture centralizes monkeypatch logic to avoid repetition across test cases.
    It allows each test to specify the retriever documents to be returned.

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest-provided fixture for dynamic attribute 
        replacement.

    Returns:
        Callable[[list[Any]], None]: A function that applies monkeypatch replacements
        with the given fake retriever documents.
    """
    def _apply(docs: list[Any]) -> None:
        # Mock rag_chain.invoke
        monkeypatch.setattr(backend, "rag_chain", FakeChain())
        # Mock refine_chain.invoke
        monkeypatch.setattr(backend, "refine_chain", FakeRefine())
        # Mock retriever.invoke
        monkeypatch.setattr(backend, "retriever", FakeRetriever(docs))
    return _apply


def test_run_llm_basic(setup_backend) -> None:
    """Test run_llm with mocked pipeline returning valid sources."""
    # Apply fixture with fake retriever documents
    fake_docs = [FakeDoc("doc1"), FakeDoc("doc2")]
    setup_backend(fake_docs)

    result: dict[str, Any] = backend.run_llm("test query")

    # Check return structure
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result

    # Check answer and sources
    assert result["answer"] == "refined answer"
    assert result["sources"] == ["doc1", "doc2"]


def test_run_llm_no_sources(setup_backend) -> None:
    """Test run_llm when retriever returns no documents."""
    # Apply fixture with empty retriever documents
    setup_backend([])

    result: dict[str, Any] = backend.run_llm("test query")
    assert result["sources"] == []


def test_run_llm_unknown_source(setup_backend) -> None:
    """Test run_llm when retriever returns documents without source metadata."""
    # Apply fixture with document missing source metadata
    class DocNoSource:
        metadata: dict[str, Any] = {}

    setup_backend([DocNoSource()])

    result: dict[str, Any] = backend.run_llm("test query")
    assert result["sources"] == ["Unknown"]
