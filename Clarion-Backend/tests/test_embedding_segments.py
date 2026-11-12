"""Regression tests for tokenizer-aware embedding segmentation and provenance."""

from models.chunk import Chunk
from services.embedding_service import EmbeddingService


class _Tokenizer:
    """Whitespace-token test double with two special tokens."""
    def __call__(self, text, **_kwargs):
        offsets, cursor = [], 0
        for word in text.split():
            start = text.index(word, cursor)
            cursor = start + len(word)
            offsets.append((start, cursor))
        return {"offset_mapping": offsets}

    def encode(self, text, add_special_tokens=True, **_kwargs):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    def build_inputs_with_special_tokens(self, _ids):
        return [101, 102]


class _Model:
    tokenizer = _Tokenizer()
    max_seq_length = 6

    def get_max_seq_length(self):
        return self.max_seq_length


def test_tokenizer_segments_cover_source_without_overlap(isolated_settings):
    service = EmbeddingService()
    service.model = _Model()
    source = "zero  one two three four five six seven"
    chunk = Chunk(chunk_id="source", document_id="doc", content=source, position_index=0, word_count=8)

    segments = service.prepare_embedding_segments([chunk])

    assert [item.content for item in segments]
    assert "".join(item.content for item in segments) == source
    assert all(service.check_token_length(item.content, max_tokens=6) <= 6 for item in segments)
    assert [item.chunk_id for item in segments] == ["source:seg:0", "source:seg:1"]
    assert service._segment_spans["source:seg:0"] == (0, len(segments[0].content))
    assert service._segment_spans["source:seg:1"][0] == len(segments[0].content)


def test_exact_limit_does_not_split(isolated_settings):
    service = EmbeddingService()
    service.model = _Model()
    chunk = Chunk(chunk_id="source", document_id="doc", content="one two three four", position_index=0, word_count=4)
    segments = service.prepare_embedding_segments([chunk])
    assert len(segments) == 1
    assert segments[0].content == chunk.content
