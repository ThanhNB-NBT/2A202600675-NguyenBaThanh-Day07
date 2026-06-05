import os
import unittest
from src.embeddings import _mock_embed
from src.advanced_chunking import SemanticChunker, AgenticChunker

SAMPLE_TEXT = (
    "Artificial intelligence is transforming industries. "
    "Machine learning enables systems to learn from data. "
    "Deep learning uses neural networks with many layers. "
    "Natural language processing handles text understanding. "
    "Computer vision processes images and video streams."
)

class TestAdvancedChunkers(unittest.TestCase):

    def test_semantic_chunker_returns_list(self):
        chunker = SemanticChunker(embedding_fn=_mock_embed, threshold=0.5)
        chunks = chunker.chunk(SAMPLE_TEXT)
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        for c in chunks:
            self.assertIsInstance(c, str)

    def test_semantic_chunker_empty_input(self):
        chunker = SemanticChunker(embedding_fn=_mock_embed)
        self.assertEqual(chunker.chunk(""), [])

    def test_agentic_chunker_fallback_without_key(self):
        # Thiết lập môi trường không có key để đảm bảo chạy chế độ dự phòng
        old_key = os.environ.get("GEMINI_API_KEY")
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
            
        try:
            chunker = AgenticChunker()
            chunks = chunker.chunk(SAMPLE_TEXT)
            self.assertIsInstance(chunks, list)
            self.assertTrue(len(chunks) > 0)
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    def test_agentic_chunker_empty_input(self):
        chunker = AgenticChunker()
        self.assertEqual(chunker.chunk(""), [])
