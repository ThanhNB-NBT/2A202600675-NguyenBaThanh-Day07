from __future__ import annotations

import os
import re
import json
import logging
from typing import Callable, Any
from pydantic import BaseModel, Field

from .chunking import compute_similarity, RecursiveChunker

# Định cấu hình logging để theo dõi quá trình phân tách nâng cao
logger = logging.getLogger("advanced_chunking")


class ChunkSplitPlan(BaseModel):
    """Lược đồ dữ liệu Pydantic định nghĩa cấu trúc JSON phản hồi từ Gemini API"""
    split_indices: list[int] = Field(description="Danh sách các chỉ số (index) đoạn văn bắt đầu một chủ đề mới")
    reasons: dict[str, str] = Field(description="Giải thích ngắn gọn lý do phân tách tại từng vị trí bằng tiếng Việt")


class SemanticChunker:
    """
    Phân tách văn bản dựa trên sự chuyển đổi ngữ nghĩa giữa các câu (Semantic Transition).
    Sử dụng vector embedding của câu và tính cosine similarity liên tiếp.
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]],
        threshold: float = 0.6,
        max_chunk_size: int = 1000
    ) -> None:
        self.embedding_fn = embedding_fn
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        # Trả về danh sách rỗng nếu văn bản trống
        if not text:
            return []

        # 1. Tách văn bản thành danh sách câu đơn lẻ
        raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if len(sentences) <= 1:
            return sentences

        # 2. Sinh vector embedding cho từng câu đơn lẻ
        embeddings = [self.embedding_fn(s) for s in sentences]

        # 3. Gom nhóm các câu dựa trên độ tương đồng cosine liên tiếp
        chunks: list[str] = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_len = len(sentences[0])

        for i in range(len(sentences) - 1):
            similarity = compute_similarity(embeddings[i], embeddings[i + 1])
            next_sentence_len = len(sentences[i + 1])
            
            # Kiểm tra: nếu tương đồng cao và tổng độ dài không vượt quá giới hạn
            if similarity >= self.threshold and (current_chunk_len + 1 + next_sentence_len) <= self.max_chunk_size:
                current_chunk_sentences.append(sentences[i + 1])
                current_chunk_len += 1 + next_sentence_len
            else:
                # Nếu tương đồng thấp hoặc vượt quá kích thước tối đa, tạo chunk mới
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i + 1]]
                current_chunk_len = next_sentence_len

        # Thêm phần còn lại vào chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks


class AgenticChunker:
    """
    Phân tách văn bản dựa trên tác nhân thông minh (Agentic Chunking).
    Sử dụng mô hình gemini-3.1-flash-lite để tìm ranh giới chủ đề tự nhiên qua các đoạn văn.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._initialized = False

        if self.api_key:
            try:
                from google import genai
                # Khởi tạo Gemini client từ SDK mới
                self.client = genai.Client(api_key=self.api_key)
                self._initialized = True
            except Exception as e:
                logger.warning(f"[AgenticChunker] Không thể khởi tạo genai Client: {e}")

    def chunk(self, text: str) -> list[str]:
        # Trả về danh sách rỗng nếu văn bản trống
        if not text:
            return []

        # 1. Phân chia văn bản thành các đoạn văn nhỏ dựa trên ký tự xuống dòng
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]

        if len(paragraphs) <= 1:
            return paragraphs

        # 2. Nếu không cấu hình API key, tự động chuyển sang bộ phân tách dự phòng (RecursiveChunker)
        if not self._initialized or not self.client:
            logger.info("[AgenticChunker] GEMINI_API_KEY chưa được thiết lập. Sử dụng thuật toán dự phòng RecursiveChunker.")
            fallback = RecursiveChunker(chunk_size=500)
            return fallback.chunk(text)

        # 3. Đánh số thứ tự các đoạn văn để đưa vào Prompt gửi lên LLM
        numbered_paragraphs = ""
        for idx, p in enumerate(paragraphs):
            numbered_paragraphs += f"[{idx}]: {p}\n\n"

        prompt = f"""Bạn là một chuyên gia biên tập và phân tích cấu trúc tài liệu. Nhiệm vụ của bạn là chia nhỏ văn bản dưới đây thành các phần (chunk) độc lập về mặt nội dung và ngữ nghĩa. Mỗi phần phải là một chủ đề trọn vẹn, không bị cắt nửa chừng.

Văn bản đầu vào được chia sẵn thành các đoạn nhỏ dưới đây (được đánh số thứ tự từ 0):
{numbered_paragraphs}

Hãy phân tích sự chuyển tiếp nội dung giữa các đoạn và chỉ ra các vị trí cần phân tách.
Chỉ ra các vị trí index của đoạn văn mà tại đó bắt đầu một chủ đề hoàn toàn mới (điểm phân tách).
Ví dụ: Nếu đoạn [0] và [1] cùng chủ đề, đoạn [2] bắt đầu chủ đề mới, đoạn [3] tiếp tục chủ đề mới, đoạn [4] bắt đầu chủ đề khác nữa, bạn nên trả về các điểm phân tách là [2, 4].

Định dạng phản hồi bắt buộc tuân theo Schema JSON quy định:
- split_indices: Mảng các chỉ số nguyên (index) bắt đầu phần mới.
- reasons: Đối tượng map giữa index đó và lý do phân tách bằng tiếng Việt.
"""

        try:
            from google.genai import types
            # Gọi API Gemini với định dạng trả về bắt buộc là JSON
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            # Đọc kết quả JSON
            result_data = json.loads(response.text)
            split_indices = sorted(result_data.get("split_indices", []))

            # 4. Thực hiện gom nhóm các đoạn văn thành các chunk dựa trên ranh giới đã tìm thấy
            chunks: list[str] = []
            current_group: list[str] = []

            for idx, p in enumerate(paragraphs):
                # Nếu chỉ số hiện tại thuộc danh sách phân tách của Agent, lưu chunk hiện tại lại
                if idx in split_indices and current_group:
                    chunks.append("\n\n".join(current_group))
                    current_group = []
                current_group.append(p)

            # Thêm phần còn lại
            if current_group:
                chunks.append("\n\n".join(current_group))

            logger.info(f"[AgenticChunker] Phân tách thành công thành {len(chunks)} chunks bằng Agentic Chunking.")
            return chunks

        except Exception as e:
            logger.error(f"[AgenticChunker] Gặp lỗi khi gọi Gemini API: {e}. Chuyển sang dùng bộ chia dự phòng.")
            fallback = RecursiveChunker(chunk_size=500)
            return fallback.chunk(text)
