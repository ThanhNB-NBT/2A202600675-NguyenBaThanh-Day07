from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# Nạp các cấu hình môi trường từ tệp .env
load_dotenv()

from src.models import Document
from src.store import EmbeddingStore
from src.embeddings import LocalEmbedder, _mock_embed
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.advanced_chunking import SemanticChunker, AgenticChunker
from benchmark_chunkers import load_dataset, extract_doc_metadata, find_section

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner(title: str):
    print("=" * 60)
    print(f" {title:^58}")
    print("=" * 60)

def main():
    clear_screen()
    print_banner("DEMO LUỒNG CHẠY HỆ THỐNG RAG VỚI DATASET DU LỊCH")
    
    # 1. Tải tập dữ liệu du lịch
    print("\n[1/4] Đang tải các tài liệu du lịch từ data/dataset...")
    dataset = load_dataset()
    if not dataset:
        print("Không tìm thấy tài liệu nào tại thư mục data/dataset. Vui lòng kiểm tra lại đường dẫn.")
        return
    print(f"-> Đã tải {len(dataset)} tài liệu thành công.")

    # 2. Khởi tạo mô hình nhúng (Embedder)
    print("\n[2/4] Đang khởi tạo mô hình nhúng ngữ nghĩa (Local Embedder: all-MiniLM-L6-v2)...")
    try:
        embedder = LocalEmbedder()
        print("-> Khởi tạo Embedder thành công. Hệ thống sử dụng tìm kiếm ngữ nghĩa thực tế.")
    except Exception as e:
        print(f"-> Không thể khởi tạo Local Embedder: {e}. Hệ thống sẽ sử dụng MockEmbedder dự phòng.")
        embedder = _mock_embed

    # 3. Cho phép người dùng lựa chọn thuật toán Chunking
    print("\n[3/4] Lựa chọn chiến lược phân tách (Chunking Strategy):")
    print("  1. Fixed-Size Chunker (Cắt theo số ký tự cố định)")
    print("  2. Sentence Chunker (Tách theo ranh giới câu)")
    print("  3. Recursive Chunker (Cắt đệ quy theo ký tự phân tách)")
    print("  4. Semantic Chunker (Tách dựa trên độ tương đồng ý nghĩa giữa các câu)")
    print("  5. Agentic Chunker (Gemini 3.1 Flash-Lite phân tích chủ đề)")
    
    choice = input("\nNhập số thứ tự lựa chọn của bạn (1-5, mặc định là 3): ").strip()
    if choice == "1":
        chunker_name = "Fixed-Size Chunker"
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    elif choice == "2":
        chunker_name = "Sentence Chunker"
        chunker = SentenceChunker(max_sentences_per_chunk=3)
    elif choice == "4":
        chunker_name = "Semantic Chunker"
        chunker = SemanticChunker(embedding_fn=embedder, threshold=0.5, max_chunk_size=1000)
    elif choice == "5":
        chunker_name = "Agentic Chunker (Gemini)"
        chunker = AgenticChunker()
    else:
        chunker_name = "Recursive Chunker"
        chunker = RecursiveChunker(chunk_size=500)

    print(f"\n-> Bạn chọn: {chunker_name}")
    print("Đang tiến hành phân tách tài liệu thành các chunks...")
    
    all_chunks_docs = []
    chunk_index = 0
    for doc in dataset:
        try:
            doc_meta = extract_doc_metadata(doc["content"])
            chunks = chunker.chunk(doc["content"])
            
            for idx, chunk_content in enumerate(chunks):
                section_name = find_section(doc["content"], chunk_content)
                metadata = {
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "title": doc_meta["title"],
                    "url": doc_meta["url"],
                    "original_price": doc_meta["original_price"],
                    "current_price": doc_meta["current_price"],
                    "section": section_name,
                    "chunk_index": idx
                }
                # Làm giàu nội dung bằng tiêu đề và mục để cải thiện khả năng tìm kiếm ngữ nghĩa
                enriched_content = f"Tài liệu: {doc_meta['title']}\nMục: {section_name or 'N/A'}\n\n{chunk_content}"
                
                all_chunks_docs.append(Document(
                    id=f"{doc['id']}_chunk_{idx}_{chunk_index}",
                    content=enriched_content,
                    metadata=metadata
                ))
                chunk_index += 1
        except Exception as e:
            print(f"Lỗi phân tách tài liệu {doc['id']}: {e}")

    total_chunks = len(all_chunks_docs)
    print(f"-> Tạo thành công {total_chunks} chunks.")

    # 4. Lưu vào Vector Store
    print("\n[4/4] Đang nạp các chunk vào Vector Store...")
    store = EmbeddingStore(collection_name="demo_store", embedding_fn=embedder)
    store.add_documents(all_chunks_docs)
    print("-> Nạp dữ liệu hoàn tất. Cơ sở tri thức đã sẵn sàng!")

    # Khởi tạo mô hình sinh câu trả lời (Gemini API) nếu có khóa API
    gemini_client = None
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            gemini_client = genai.Client(api_key=api_key)
            print("\n[!] Đã nhận diện GEMINI_API_KEY. Hệ thống sẽ trả lời câu hỏi bằng Gemini 3.1 Flash-Lite thực tế.")
        except Exception as e:
            print(f"\n[!] Không thể kết nối Gemini API: {e}. Hệ thống sẽ sử dụng phản hồi Demo.")
    else:
        print("\n[!] Chưa cấu hình GEMINI_API_KEY. Hệ thống sẽ sử dụng phản hồi mô phỏng (Demo).")

    # 5. Vòng lặp tương tác câu hỏi
    print("\n" + "=" * 60)
    print(" BẮT ĐẦU ĐẶT CÂU HỎI (Interactive Q&A)")
    print(" Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("=" * 60 + "\n")

    while True:
        question = input("\nNhập câu hỏi của bạn: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            print("Cảm ơn bạn đã trải nghiệm hệ thống RAG!")
            break
            
        print("\n--- [Bước 1: Thực hiện tìm kiếm tương đồng trên Vector Store] ---")
        # Tìm kiếm 4 chunks tốt nhất
        search_results = store.search(question, top_k=4)
        
        if not search_results:
            print("Không tìm thấy kết quả tương đồng nào phù hợp.")
            continue
            
        contexts = []
        for rank, r in enumerate(search_results, 1):
            meta = r["metadata"]
            contexts.append(r["content"])
            # Tách lấy nội dung sạch để hiển thị
            parts = r["content"].split("\n\n", 1)
            clean_content = parts[1] if len(parts) > 1 else r["content"]
            
            print(f"\n[{rank}] Chunk tương đồng (Score: {r['score']:.4f})")
            print(f"  * Tiêu đề: {meta.get('title')}")
            print(f"  * Mục (Section): {meta.get('section') or 'Không xác định'}")
            print(f"  * Chỉ số chunk: {meta.get('chunk_index')}")
            print(f"  * Link đặt vé (URL): {meta.get('url')}")
            print(f"  * Giá vé: gốc ({meta.get('original_price') or 'N/A'}) | hiện tại ({meta.get('current_price') or 'N/A'})")
            print(f"  * Nội dung trích xuất:")
            print(f"    \"\"\"{clean_content[:300]}...\"\"\"")

        print("\n--- [Bước 2: Xây dựng Prompt RAG gửi tới RAG Agent] ---")
        context_str = "\n\n".join(contexts)
        prompt = (
            "Context information is below.\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query in Vietnamese.\n"
            f"Query: {question}\n"
            "Answer:"
        )
        print("Đang chuẩn bị ngữ cảnh và gửi yêu cầu sinh câu trả lời...")

        print("\n--- [Bước 3: Trả lời từ RAG Agent] ---")
        if gemini_client:
            try:
                # Gọi thực tế Gemini 3.1 Flash-Lite
                response = gemini_client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt
                )
                print(f"\nCÂU TRẢ LỜI:")
                print(response.text)
            except Exception as e:
                print(f"\n[Lỗi API] Không thể sinh câu trả lời: {e}")
        else:
            # Mô phỏng phản hồi
            print("\nCÂU TRẢ LỜI MÔ PHỎNG (Chưa lắp API Key):")
            print(f"[DEMO LLM] Đã đọc được {len(search_results)} chunks ngữ cảnh và câu hỏi: '{question}'.")
            print(f"-> Gợi ý: Hãy thêm GEMINI_API_KEY vào tệp .env để nhận câu trả lời thật từ Gemini 3.1 Flash-Lite!")
        print("\n" + "-"*40)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChương trình kết thúc.")
        sys.exit(0)
