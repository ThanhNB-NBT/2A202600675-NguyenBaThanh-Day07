import os
import re
import time
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Nạp cấu hình .env
load_dotenv()

from src.models import Document
from src.store import EmbeddingStore
from src.embeddings import LocalEmbedder, _mock_embed
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker
from src.advanced_chunking import SemanticChunker, AgenticChunker
from benchmark_chunkers import load_dataset, extract_doc_metadata, find_section, calculate_chunk_coherence

# Cấu hình giao diện Streamlit rộng rãi và hiện đại
st.set_page_config(
    page_title="RAG Tourism Assistant & Chunking Visualizer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để giao diện trông sang trọng hơn
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .chunk-card {
        background-color: #F3F4F6;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .metadata-tag {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .price-tag {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .section-tag {
        background-color: #FEF3C7;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)


# Quản lý State trong Streamlit (để tránh nạp lại Vector Store mỗi lần người dùng click)
if "embedder" not in st.session_state:
    try:
        st.session_state.embedder = LocalEmbedder()
    except Exception:
        st.session_state.embedder = _mock_embed

if "dataset" not in st.session_state:
    st.session_state.dataset = load_dataset()

if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.indexed_strategy = None
    st.session_state.total_chunks = 0
    st.session_state.chunks_list = []


def index_documents(chunker, chunker_name):
    """Tiến hành phân tách dữ liệu và nạp vào Vector Store"""
    with st.spinner(f"Đang phân tách bằng {chunker_name} và nạp vào Vector Store..."):
        all_chunks_docs = []
        chunks_list = []
        chunk_index = 0
        
        for doc in st.session_state.dataset:
            doc_meta = extract_doc_metadata(doc["content"])
            doc_chunks = chunker.chunk(doc["content"])
            
            for idx, chunk_content in enumerate(doc_chunks):
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
                
                doc_obj = Document(
                    id=f"{doc['id']}_chunk_{idx}_{chunk_index}",
                    content=enriched_content,
                    metadata=metadata
                )
                all_chunks_docs.append(doc_obj)
                chunks_list.append(doc_obj)
                chunk_index += 1
                
        # Nạp vào EmbeddingStore
        store = EmbeddingStore(collection_name=f"web_store_{chunker_name.replace(' ', '_')}", embedding_fn=st.session_state.embedder)
        store.add_documents(all_chunks_docs)
        
        # Lưu vào Session State
        st.session_state.store = store
        st.session_state.indexed_strategy = chunker_name
        st.session_state.total_chunks = len(all_chunks_docs)
        st.session_state.chunks_list = chunks_list


# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.title("⚙️ Cấu hình hệ thống")
    
    # 1. API Key cho Gemini
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Thêm API Key để kích hoạt Agentic Chunking và câu trả lời RAG thật từ Gemini 3.1 Flash-Lite."
    )
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        
    st.write("---")
    
    # 2. Lựa chọn bộ phân tách
    st.subheader("Thuật toán phân tách (Chunker)")
    chunker_choice = st.selectbox(
        "Chọn phương pháp",
        [
            "Recursive Chunker",
            "Fixed-Size Chunker",
            "Sentence Chunker",
            "Semantic Chunker",
            "Agentic Chunker (Gemini)"
        ],
        index=0
    )
    
    # Hiển thị các tham số động tùy thuộc bộ chia được chọn
    params = {}
    if chunker_choice == "Fixed-Size Chunker":
        params["chunk_size"] = st.slider("Chunk Size (ký tự)", 100, 2000, 500, step=50)
        params["overlap"] = st.slider("Overlap (ký tự)", 0, 500, 50, step=10)
        selected_chunker = FixedSizeChunker(chunk_size=params["chunk_size"], overlap=params["overlap"])
        
    elif chunker_choice == "Sentence Chunker":
        params["max_sentences"] = st.slider("Số câu tối đa / chunk", 1, 10, 3)
        selected_chunker = SentenceChunker(max_sentences_per_chunk=params["max_sentences"])
        
    elif chunker_choice == "Semantic Chunker":
        params["threshold"] = st.slider("Ngưỡng tương đồng (Cosine Similarity)", 0.1, 0.9, 0.5, step=0.05)
        params["max_size"] = st.slider("Kích thước chunk tối đa (ký tự)", 200, 3000, 1000, step=100)
        selected_chunker = SemanticChunker(embedding_fn=st.session_state.embedder, threshold=params["threshold"], max_chunk_size=params["max_size"])
        
    elif chunker_choice == "Agentic Chunker (Gemini)":
        st.info("Gemini 3.1 Flash-Lite sẽ tự động phân tích và xác định ranh giới chủ đề.")
        selected_chunker = AgenticChunker(api_key=gemini_key)
        
    else: # Recursive Chunker
        params["chunk_size"] = st.slider("Chunk Size (ký tự)", 100, 2000, 500, step=50)
        selected_chunker = RecursiveChunker(chunk_size=params["chunk_size"])

    st.write("---")
    
    # 3. Nút nạp lại Vector Store
    if st.button("🚀 Nạp & Phân tách lại dữ liệu", use_container_width=True) or st.session_state.store is None:
        index_documents(selected_chunker, chunker_choice)
        st.success("Đã hoàn tất phân tách và nạp Vector Store!")

    st.write("---")
    # Hiển thị chỉ số hiện tại
    st.metric("Tổng số tài liệu", len(st.session_state.dataset))
    st.metric("Tổng số chunks", st.session_state.total_chunks)
    st.caption(f"Chiến lược hoạt động: **{st.session_state.indexed_strategy}**")


# --- MAIN AREA ---
st.markdown("<div class='main-title'>🗺️ Trợ lý RAG Du lịch & Visualizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Hệ thống RAG thông minh ứng dụng Advanced Chunking so khớp dữ liệu du lịch Vinpearl & Grand World</div>", unsafe_allow_html=True)

# Tạo các tab lớn cho giao diện Web
tab_chat, tab_visualizer, tab_benchmark = st.tabs([
    "💬 Trợ lý Ảo RAG",
    "🔍 Trực quan hóa Chunking",
    "📊 Bảng điều khiển Benchmark"
])

# ================= TAB 1: TRỢ LÝ RAG =================
with tab_chat:
    st.subheader("Hỏi đáp thông minh về dịch vụ du lịch")
    st.write("Hãy thử hỏi các câu hỏi về giá vé, giờ mở cửa, các hoạt động vui chơi hay điều khoản sử dụng tại các khu vui chơi...")
    
    # Khởi tạo tin nhắn chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Hiển thị lịch sử trò chuyện
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chunks"):
                with st.expander("Trích xuất ngữ cảnh RAG"):
                    for rank, chunk in enumerate(msg["chunks"], 1):
                        meta = chunk["metadata"]
                        # Tách lấy nội dung sạch để hiển thị
                        parts = chunk["content"].split("\n\n", 1)
                        clean_content = parts[1] if len(parts) > 1 else chunk["content"]
                        
                        st.markdown(f"**Chunk {rank} - {meta.get('title')} > {meta.get('section') or 'N/A'}** (Score: {chunk['score']:.4f})")
                        
                        # Hiển thị metadata chi tiết dạng cột
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.markdown(f"🏷️ **Mục:** `{meta.get('section') or 'N/A'}`")
                            st.markdown(f"🔢 **Chỉ số chunk:** `{meta.get('chunk_index', 0)}`")
                        with col_m2:
                            st.markdown(f"💰 **Giá gốc:** `{meta.get('original_price') or 'N/A'}`")
                            st.markdown(f"💸 **Giá hiện tại:** `{meta.get('current_price') or 'N/A'}`")
                        with col_m3:
                            st.markdown(f"🌐 **Link:** [Đặt vé]({meta.get('url')})")
                            st.markdown(f"🎯 **Độ tương đồng:** `{chunk['score']:.4f}`")
                            
                        st.text_area(f"Nội dung trích xuất {rank}", clean_content, height=120, disabled=True, key=f"hist_ta_{rank}_{meta.get('doc_id')}_{meta.get('chunk_index', 0)}")

    # Nhập câu hỏi mới
    if user_query := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        # Thêm câu hỏi vào UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            # 1. Tìm kiếm Vector Store
            search_results = st.session_state.store.search(user_query, top_k=4) # Tăng lên top_k=4 để bao phủ thông tin đầy đủ hơn
            
            # 2. Xây dựng prompt
            contexts = [r["content"] for r in search_results]
            context_str = "\n\n".join(contexts)
            prompt = (
                "Context information is below.\n"
                "---------------------\n"
                f"{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the query in Vietnamese. Be accurate to prices and URLs.\n"
                f"Query: {user_query}\n"
                "Answer:"
            )
            
            # 3. Gọi LLM sinh câu trả lời
            answer_text = ""
            if gemini_key:
                try:
                    from google import genai
                    client = genai.Client(api_key=gemini_key)
                    response = client.models.generate_content(
                        model="gemini-3.1-flash-lite",
                        contents=prompt
                    )
                    answer_text = response.text
                except Exception as e:
                    answer_text = f"Lỗi gọi API Gemini: {e}\n\nSử dụng phản hồi mô phỏng (Demo)."
            
            # Nếu không có key hoặc lỗi, fallback phản hồi mô phỏng
            if not answer_text or "[Lỗi API]" in answer_text:
                answer_text = (
                    f"**[Chế độ Demo]** Đã tìm thấy {len(search_results)} ngữ cảnh liên quan.\n\n"
                    f"Để nhận câu trả lời thật sự từ AI, hãy điền **Gemini API Key** ở sidebar.\n\n"
                    f"**Gợi ý tìm kiếm:** Bạn có thể xem nội dung trích xuất chi tiết bên dưới!"
                )
                
            st.markdown(answer_text)
            
            # Hiển thị các chunk liên quan dạng expander
            with st.expander("Trích xuất ngữ cảnh RAG"):
                for rank, chunk in enumerate(search_results, 1):
                    meta = chunk["metadata"]
                    # Tách lấy nội dung sạch để hiển thị
                    parts = chunk["content"].split("\n\n", 1)
                    clean_content = parts[1] if len(parts) > 1 else chunk["content"]
                    
                    st.markdown(f"**Chunk {rank} - {meta.get('title')} > {meta.get('section') or 'N/A'}** (Score: {chunk['score']:.4f})")
                    
                    # Hiển thị metadata chi tiết dạng cột
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"🏷️ **Mục:** `{meta.get('section') or 'N/A'}`")
                        st.markdown(f"🔢 **Chỉ số chunk:** `{meta.get('chunk_index', 0)}`")
                    with col_m2:
                        st.markdown(f"💰 **Giá gốc:** `{meta.get('original_price') or 'N/A'}`")
                        st.markdown(f"💸 **Giá hiện tại:** `{meta.get('current_price') or 'N/A'}`")
                    with col_m3:
                        st.markdown(f"🌐 **Link:** [Đặt vé]({meta.get('url')})")
                        st.markdown(f"🎯 **Độ tương đồng:** `{chunk['score']:.4f}`")
                        
                    st.text_area(f"Nội dung trích xuất {rank} (thực tế)", clean_content, height=120, disabled=True, key=f"chat_ta_{rank}_{meta.get('doc_id')}_{meta.get('chunk_index', 0)}")
            
            # Lưu tin nhắn vào lịch sử
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "chunks": search_results
            })


# ================= TAB 2: TRỰC QUAN HÓA CHUNKING =================
with tab_visualizer:
    st.subheader("Trực quan hóa kết quả phân tách tài liệu")
    
    # Chọn tài liệu để quan sát
    doc_options = [d["id"] for d in st.session_state.dataset]
    selected_doc_id = st.selectbox("Chọn tài liệu để hiển thị", doc_options)
    
    # Lọc danh sách chunks thuộc tài liệu được chọn
    doc_chunks = [c for c in st.session_state.chunks_list if c.metadata.get("doc_id") == selected_doc_id]
    
    st.write(f"Tài liệu **{selected_doc_id}** được chia làm **{len(doc_chunks)} chunks** bằng chiến lược **{st.session_state.indexed_strategy}**:")
    
    # Hiển thị từng chunk dạng thẻ bài
    for c_idx, chunk in enumerate(doc_chunks):
        meta = chunk.metadata
        # Tách lấy nội dung sạch để hiển thị
        parts = chunk.content.split("\n\n", 1)
        clean_content = parts[1] if len(parts) > 1 else chunk.content
        coherence = calculate_chunk_coherence(clean_content, st.session_state.embedder)
        
        with st.container():
            st.markdown(f"""
            <div class='chunk-card'>
                <span class='metadata-tag section-tag'># Chunk {c_idx} - Mục: {meta.get('section') or 'Không có'}</span>
                <span class='metadata-tag'>Chỉ số: {meta.get('chunk_index')}</span>
                <span class='metadata-tag price-tag'>Giá gốc: {meta.get('original_price') or 'N/A'}</span>
                <span class='metadata-tag price-tag'>Giá hiện tại: {meta.get('current_price') or 'N/A'}</span>
                <span class='metadata-tag'>Độ dài: {len(clean_content)} ký tự</span>
                <span class='metadata-tag'>Coherence: {coherence:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"🌐 **Link đặt vé:** [Nhấp vào đây]({meta.get('url')})")
            
            st.text_area(
                f"Nội dung Chunk {c_idx}",
                clean_content,
                height=120,
                key=f"vis_ta_{c_idx}_{meta.get('doc_id')}_{c_idx}",
                disabled=True
            )
            st.write("")


# ================= TAB 3: BẢNG ĐIỀU KHIỂN BENCHMARK =================
with tab_benchmark:
    st.subheader("Báo cáo đánh giá & biểu đồ so sánh 5 thuật toán")
    st.write("Các chỉ số này thu được từ việc chạy thử nghiệm benchmark trên toàn bộ 10 tài liệu Markdown thực tế với bộ câu hỏi chuẩn:")
    
    # Hardcode kết quả chạy thực tế để vẽ biểu đồ trực quan
    benchmark_data = {
        "Phương pháp": ["Fixed-Size Chunker", "Sentence Chunker", "Recursive Chunker", "Semantic Chunker", "Agentic Chunker (Gemini)"],
        "Số chunk": [140, 128, 165, 123, 99],
        "Độ dài trung bình (ký tự)": [483.5, 476.6, 369.6, 496.0, 619.9],
        "Độ kết dính Coherence": [0.593, 0.603, 0.636, 0.782, 0.708],
        "Hit@1 (%)": [100.0, 60.0, 80.0, 60.0, 60.0],
        "Hit@3 (%)": [100.0, 80.0, 100.0, 100.0, 60.0],
        "Thời gian xử lý (giây)": [7.00, 5.40, 6.17, 11.86, 20.38]
    }
    
    df = pd.DataFrame(benchmark_data)
    
    # Hiển thị bảng số liệu gốc
    st.dataframe(df.set_index("Phương pháp"), use_container_width=True)
    
    st.markdown("### 📊 Biểu đồ so sánh các chỉ số")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Độ kết dính ngữ nghĩa nội bộ (Coherence - Càng cao càng tốt)**")
        # Chuẩn bị DataFrame vẽ biểu đồ
        df_coh = df[["Phương pháp", "Độ kết dính Coherence"]].set_index("Phương pháp")
        st.bar_chart(df_coh, color="#3B82F6")
        
        st.markdown("**Độ chính xác Hit@1 (%) (Khả năng tìm thấy tài liệu vàng ở vị trí số 1)**")
        df_hit1 = df[["Phương pháp", "Hit@1 (%)"]].set_index("Phương pháp")
        st.bar_chart(df_hit1, color="#10B981")

    with col2:
        st.markdown("**Số lượng chunk tạo ra (Tiết kiệm số lượng token / lưu trữ)**")
        df_count = df[["Phương pháp", "Số chunk"]].set_index("Phương pháp")
        st.bar_chart(df_count, color="#F59E0B")
        
        st.markdown("**Thời gian phân tách (giây) (Càng thấp càng tốt)**")
        df_time = df[["Phương pháp", "Thời gian xử lý (giây)"]].set_index("Phương pháp")
        st.bar_chart(df_time, color="#EF4444")
        
    st.write("---")
    st.markdown("""
    ### 📝 Phân tích kết luận từ biểu đồ:
    1. **Semantic Chunker** đạt độ kết dính **Coherence cao nhất (0.782)** vì nó tách dựa vào khoảng cách ngữ nghĩa thực sự giữa các câu, đảm bảo các câu trong một chunk có mối liên quan mật thiết.
    2. **Agentic Chunker (Gemini)** cho độ dài trung bình lớn nhất (619.9 ký tự) nhưng số lượng chunk ít nhất (99 chunks), phản ánh cách AI gộp các đoạn văn theo nhóm chủ đề lớn liền mạch một cách tối ưu.
    3. **Fixed-Size Chunker** cho Hit@1 đạt 100% do có độ dài cố định lớn và sự ngẫu nhiên của bộ câu hỏi nhỏ trên tập dữ liệu được gom cụm tốt. Tuy nhiên chỉ số Coherence của nó thấp nhất vì cắt đôi ý ngữ cảnh ở ranh giới.
    """)
