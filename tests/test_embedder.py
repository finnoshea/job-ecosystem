"""Tests for jobecosystem.core.embedder.

Voyage is a hosted API, so a fake client is injected: these tests never touch
the network and need no key or model.
"""

from __future__ import annotations

import pytest

from jobecosystem.core import embedder, similarity


class FakeResponse:
    def __init__(self, embeddings, total_tokens):
        self.embeddings = embeddings
        self.total_tokens = total_tokens


class FakeClient:
    """Records each request and returns zero vectors of a fixed width."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls: list[dict] = []
        self.fail = False

    def embed(self, *, texts, model, input_type):
        self.calls.append(
            {"texts": list(texts), "model": model, "input_type": input_type}
        )
        if self.fail:
            raise RuntimeError("HTTP 429")
        return FakeResponse(
            [[0.0] * self.dim for _ in texts], total_tokens=len(texts) * 7
        )


@pytest.fixture(autouse=True)
def _reset_state():
    embedder.set_client(None)
    embedder.reset_usage()
    yield
    embedder.set_client(None)
    embedder.reset_usage()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_default_model_is_voyage_4_lite():
    assert embedder.DEFAULT_MODEL == "voyage-4-lite"


def test_embedding_model_name_is_the_model_id():
    assert embedder.embedding_model_name() == embedder.DEFAULT_MODEL
    assert embedder.embedding_model_name("other") == "other"


def test_dimension_defaults_to_the_voyage_width():
    assert embedder.dimension() == 1024
    assert embedder.dimension() == similarity.VOYAGE_DIM
    assert embedder.dimension("unknown-model") == 1024


def test_dimension_knows_smaller_models():
    assert embedder.dimension("voyage-3-lite") == 512


def test_embedder_error_is_a_runtime_error():
    assert issubclass(embedder.EmbedderError, RuntimeError)


# ---------------------------------------------------------------------------
# client handling
# ---------------------------------------------------------------------------

def test_missing_api_key_is_an_embedder_error(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder.set_client(None)
    with pytest.raises(embedder.EmbedderError, match="VOYAGE_API_KEY"):
        embedder.embed_documents(["text"])


def test_get_client_caches(monkeypatch):
    fake = FakeClient()
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder.set_client(fake)
    assert embedder.get_client() is fake


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_without_calling_the_api():
    fake = FakeClient()
    embedder.set_client(fake)
    assert embedder.embed_documents([]) == []
    assert fake.calls == []


def test_documents_use_input_type_document():
    fake = FakeClient()
    embedder.set_client(fake)
    embedder.embed_documents(["a", "b"])
    assert fake.calls[0]["input_type"] == "document"
    assert fake.calls[0]["texts"] == ["a", "b"]


def test_query_uses_input_type_query():
    fake = FakeClient()
    embedder.set_client(fake)
    embedder.embed_query("python engineer")
    assert fake.calls[0]["input_type"] == "query"


def test_document_single_uses_input_type_document():
    fake = FakeClient()
    embedder.set_client(fake)
    embedder.embed_document("a description")
    assert fake.calls[0]["input_type"] == "document"


def test_model_id_is_passed_through():
    fake = FakeClient()
    embedder.set_client(fake)
    embedder.embed_documents(["a"], model_id="voyage-3")
    assert fake.calls[0]["model"] == "voyage-3"


def test_documents_are_batched():
    fake = FakeClient()
    embedder.set_client(fake)
    vectors = embedder.embed_documents([str(i) for i in range(5)], batch_size=2)
    assert [len(call["texts"]) for call in fake.calls] == [2, 2, 1]
    assert len(vectors) == 5


def test_batch_size_below_one_is_rejected():
    embedder.set_client(FakeClient())
    with pytest.raises(ValueError, match="batch_size"):
        embedder.embed_documents(["a"], batch_size=0)


def test_unknown_input_type_is_rejected():
    embedder.set_client(FakeClient())
    with pytest.raises(ValueError, match="input_type"):
        embedder._embed(["a"], input_type="nonsense", model_id="m", batch_size=1)


def test_api_failure_is_wrapped():
    fake = FakeClient()
    fake.fail = True
    embedder.set_client(fake)
    with pytest.raises(embedder.EmbedderError, match="Voyage embed failed"):
        embedder.embed_documents(["a"])


# ---------------------------------------------------------------------------
# usage accounting
# ---------------------------------------------------------------------------

def test_usage_is_accumulated_across_batches():
    fake = FakeClient()
    embedder.set_client(fake)
    embedder.reset_usage()
    embedder.embed_documents([str(i) for i in range(5)], batch_size=2)
    # 5 texts * 7 tokens
    assert embedder.total_tokens_used() == 35


def test_reset_usage_zeroes_the_counter():
    fake = FakeClient()
    embedder.set_client(fake)
    embedder.embed_documents(["a"])
    embedder.reset_usage()
    assert embedder.total_tokens_used() == 0


# ---------------------------------------------------------------------------
# blob helpers
# ---------------------------------------------------------------------------

def test_document_blob_round_trips():
    fake = FakeClient(dim=4)
    embedder.set_client(fake)
    text = "Senior Python engineer"
    assert similarity.decode(embedder.embed_document_blob(text), 4) == (
        embedder.embed_document(text)
    )


def test_query_blob_round_trips():
    fake = FakeClient(dim=4)
    embedder.set_client(fake)
    text = "python sqlite"
    assert similarity.decode(embedder.embed_query_blob(text), 4) == (
        embedder.embed_query(text)
    )
