from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Trả về danh sách rỗng nếu văn bản đầu vào rỗng
        if not text:
            return []
        
        # Phân tách văn bản thành các câu bằng biểu thức chính quy.
        # Tìm các khoảng trắng (\s+) đứng sau dấu chấm, chấm hỏi hoặc chấm than,
        # hoặc dấu xuống dòng (\n) đứng sau dấu chấm.
        raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
        
        # Loại bỏ khoảng trắng thừa ở đầu/cuối mỗi câu và lọc bỏ câu rỗng
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        # Nhóm các câu lại thành từng chunk có kích thước tối đa là max_sentences_per_chunk
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        # Khởi tạo danh sách các ký tự phân tách và độ dài tối đa của mỗi chunk
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # Trả về danh sách rỗng nếu văn bản đầu vào rỗng
        if not text:
            return []
        # Bắt đầu quá trình phân tách đệ quy từ danh sách separators đã cấu hình
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Trường hợp cơ bản: nếu độ dài văn bản nhỏ hơn hoặc bằng chunk_size, không cần chia nhỏ
        if len(current_text) <= self.chunk_size:
            return [current_text]
            
        # Trường hợp cơ bản 2: nếu không còn ký tự phân tách nào, cắt theo độ dài ký tự cố định
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks
            
        # Lấy ký tự phân tách hiện tại có độ ưu tiên cao nhất và phần còn lại
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        
        # Phân tách văn bản bằng separator hiện tại
        if separator == "":
            splits = list(current_text)
        else:
            # Dùng split thông thường để chia nhỏ
            splits = current_text.split(separator)
            
        # Đệ quy xử lý các phần văn bản sau khi chia:
        # Nếu một phần nhỏ vẫn vượt quá chunk_size, nó sẽ được đệ quy chia nhỏ tiếp bằng next_separators.
        pieces = []
        for s in splits:
            if len(s) <= self.chunk_size:
                pieces.append(s)
            else:
                pieces.extend(self._split(s, next_separators))
                
        # Gộp các mảnh nhỏ lại với nhau bằng separator hiện tại sao cho độ dài của chunk không vượt quá chunk_size
        chunks = []
        current_chunk_pieces = []
        current_len = 0
        
        for piece in pieces:
            # Bỏ qua các phần tử rỗng
            if not piece:
                continue
                
            # Tính độ dài mới nếu gộp thêm mảnh hiện tại
            sep_len = len(separator) if separator != "" else 0
            new_len = current_len + (sep_len if current_chunk_pieces else 0) + len(piece)
            
            if new_len <= self.chunk_size:
                current_chunk_pieces.append(piece)
                current_len = new_len
            else:
                # Nếu vượt quá, lưu chunk hiện tại và bắt đầu một chunk mới với mảnh hiện tại
                if current_chunk_pieces:
                    chunks.append(separator.join(current_chunk_pieces))
                current_chunk_pieces = [piece]
                current_len = len(piece)
                
        # Thêm phần còn lại vào danh sách chunk
        if current_chunk_pieces:
            chunks.append(separator.join(current_chunk_pieces))
            
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Tính độ dài (magnitude) của vector vec_a
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    # Tính độ dài (magnitude) của vector vec_b
    mag_b = math.sqrt(sum(y * y for y in vec_b))
    
    # Bảo vệ chia cho 0: Nếu một trong hai vector có độ dài bằng 0, trả về 0.0
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
        
    # Áp dụng công thức tính Cosine Similarity: Tích vô hướng chia cho tích độ dài
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Khởi tạo ba bộ phân tách khác nhau
        # 1. FixedSizeChunker với kích thước cố định
        fs_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        
        # 2. SentenceChunker phân tách theo câu (ước lượng khoảng 100 ký tự mỗi câu)
        s_chunker = SentenceChunker(max_sentences_per_chunk=max(1, chunk_size // 100))
        
        # 3. RecursiveChunker phân tách đệ quy
        r_chunker = RecursiveChunker(chunk_size=chunk_size)
        
        strategies = {
            'fixed_size': fs_chunker,
            'by_sentences': s_chunker,
            'recursive': r_chunker
        }
        
        comparison = {}
        for name, chunker in strategies.items():
            # Thực hiện chia nhỏ văn bản
            chunks = chunker.chunk(text)
            count = len(chunks)
            
            # Tính độ dài trung bình của các chunk
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            
            comparison[name] = {
                'count': count,
                'avg_length': avg_length,
                'chunks': chunks
            }
            
        return comparison
