from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # Lưu trữ các tham chiếu đến vector store và hàm gọi mô hình ngôn ngữ (llm_fn)
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Truy vấn các chunk tài liệu liên quan nhất từ vector store
        results = self.store.search(question, top_k=top_k)
        
        # 2. Trích xuất nội dung văn bản từ các kết quả truy vấn được
        contexts = [r["content"] for r in results]
        context_str = "\n\n".join(contexts)
        
        # 3. Xây dựng prompt RAG chuẩn để gửi cho LLM
        prompt = (
            "Context information is below.\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            f"Query: {question}\n"
            "Answer:"
        )
        
        # 4. Gọi hàm LLM với prompt vừa xây dựng và trả về kết quả trả lời
        return self.llm_fn(prompt)
