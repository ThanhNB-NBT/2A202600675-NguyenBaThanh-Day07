# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Bá Thành
**Nhóm:** B4
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity cao (gần bằng 1.0) nghĩa là hai đoạn văn bản có độ tương đồng ngữ nghĩa lớn trong không gian vector. Điều này chỉ ra rằng hai văn bản hướng về cùng một chủ đề hoặc mang cùng một ý nghĩa cốt lõi, bất kể độ dài hay từ ngữ khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Học máy là một lĩnh vực của trí tuệ nhân tạo."
- Sentence B: "Machine learning là một nhánh của AI."
- Tại sao tương đồng: Cả hai câu đều nói về mối quan hệ phân cấp giữa học máy và trí tuệ nhân tạo, mặc dù sử dụng từ vựng và ngôn ngữ khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Quả chuối này rất chín và ngọt."
- Sentence B: "Hệ điều hành Windows vừa cập nhật phiên bản mới."
- Tại sao khác: Hai câu đề cập tới hai chủ đề hoàn toàn khác biệt (trái cây ăn được và phần mềm máy tính), không có mối liên hệ ngữ nghĩa nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo góc giữa hai vector mà không bị ảnh hưởng bởi độ lớn tuyệt đối (độ dài) của vector đó. Đối với văn bản, một văn bản dài chứa nhiều từ lặp lại sẽ có vector với độ lớn rất cao so với văn bản ngắn dù nội dung tương tự nhau. Việc sử dụng Cosine similarity giúp bỏ qua sự khác biệt về độ dài để tập trung hoàn toàn vào ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: $num\_chunks = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.11 \rceil = 23$
> Đáp án: 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk tăng lên thành: $\lceil \frac{10000 - 100}{500 - 100} \rceil = \lceil \frac{9900}{400} \rceil = \lceil 24.75 \rceil = 25$ chunks. Ta muốn có overlap lớn hơn để đảm bảo rằng ngữ cảnh ở vị trí ranh giới giữa các chunk không bị đứt gãy hay mất thông tin, giúp RAG Agent dễ dàng tìm kiếm và trả lời đầy đủ hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hỗ trợ khách hàng và Tra cứu thông tin dịch vụ du lịch, vui chơi giải trí & nghỉ dưỡng (Vinpearl, VinWonders, Aquafield).

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì dịch vụ du lịch, đặt phòng combo nghỉ dưỡng và vé vui chơi giải trí có nhu cầu tra cứu thông tin thực tế rất lớn (đặc biệt là giá vé, điều kiện hoàn hủy, các dịch vụ đi kèm). Tài liệu của domain này có cấu trúc đa dạng (chứa cả bảng biểu, danh sách điều khoản và mô tả văn bản thông thường), rất phù hợp để so sánh hiệu năng giữa các chiến lược phân tách văn bản (chunking) khác nhau. Đồng thời, việc có các trường giá cả, liên kết URL cố định giúp nhóm thiết kế và đánh giá khả năng trích xuất metadata và tìm kiếm lọc chính xác (pre-filtering).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `VinWonders_Nha_Trang` | `data/dataset/VinWonders_Nha_Trang.md` | 6138 | `doc_id`, `source`, `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index` |
| 2 | `Aquafield_Nha_Trang_-_Spa_&_xông_hơi_chuẩn_Hàn` | `data/dataset/Aquafield_Nha_Trang_-_Spa_&_xông_hơi_chuẩn_Hàn.md` | 7007 | `doc_id`, `source`, `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index` |
| 3 | `[Cần_Thơ]_2N1Đ_phòng_Deluxe_+_Bữa_sáng_tại_Vinpearl_Hotel_Cần_Thơ` | `data/dataset/[Cần_Thơ]_2N1Đ_phòng_Deluxe_+_Bữa_sáng_tại_Vinpearl_Hotel_Cần_Thơ.md` | 4936 | `doc_id`, `source`, `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index` |
| 4 | `[Vinpearl_Golf_Phú_Quốc]_-_Voucher_Tee_time_giá_siêu_ưu_đãi` | `data/dataset/[Vinpearl_Golf_Phú_Quốc]_-_Voucher_Tee_time_giá_siêu_ưu_đãi.md` | 2791 | `doc_id`, `source`, `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index`, `product_type` |
| 5 | `Vinpearl_Safari_Phú_Quốc` | `data/dataset/Vinpearl_Safari_Phú_Quốc.md` | 4609 | `doc_id`, `source`, `title`, `url`, `original_price`, `current_price`, `section`, `chunk_index` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | String | `VinWonders_Nha_Trang` | Xác định nguồn gốc tài liệu để gộp hoặc xóa tài liệu khi cần. |
| `title` | String | `VinWonders Nha Trang` | Giúp tăng độ tương đồng khi tìm kiếm và hiển thị nguồn gốc chunk cho người dùng. |
| `url` | String | `https://booking.vinpearl.com/...` | Cung cấp link trực tiếp cho người dùng đặt mua dịch vụ ngay trong câu trả lời. |
| `original_price` | String | `600.000 đ` | Phục vụ so sánh giá hoặc hiển thị giá gốc trước khi giảm. |
| `current_price` | String | `500.000 đ` | Dùng để trả lời nhanh về giá bán thực tế hiện tại. |
| `section` | String | `Mô tả` | Xác định ngữ cảnh thuộc phần nào (Điều khoản, Hướng dẫn sử dụng, v.v.) của tài liệu. |
| `chunk_index` | Integer | `0` | Cho biết thứ tự chunk trong tài liệu để thực hiện ghép nối ngữ cảnh liên tiếp. |
| `product_type` | String | `golf_voucher` | Phục vụ pre-filtering (tiền lọc) các loại sản phẩm đặc thù (như golf, combo) trước khi vector search. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu                | Strategy                         | Chunk Count | Avg Length | Preserves Context?                |
| -------------------------| ----------------------------------| -------------| ------------| -----------------------------------|
| `python_intro.txt`      | FixedSizeChunker (`fixed_size`)  | 10          | 194.40     | Không tốt (cắt ở ký tự bất kỳ)    |
| `python_intro.txt`      | SentenceChunker (`by_sentences`) | 8           | 241.50     | Khá tốt (tách theo câu)           |
| `python_intro.txt`      | RecursiveChunker (`recursive`)   | 12          | 160.08     | Rất tốt (theo đoạn văn)           |
| `vector_store_notes.md` | FixedSizeChunker (`fixed_size`)  | 11          | 193.00     | Không tốt (mất ranh giới từ)      |
| `vector_store_notes.md` | SentenceChunker (`by_sentences`) | 12          | 175.42     | Khá tốt (tách theo câu)           |
| `vector_store_notes.md` | RecursiveChunker (`recursive`)   | 15          | 139.67     | Rất tốt (theo markdown structure) |
| `rag_system_design.md`  | FixedSizeChunker (`fixed_size`)  | 12          | 199.25     | Không tốt (cắt từ ngữ ngẫu nhiên) |
| `rag_system_design.md`  | SentenceChunker (`by_sentences`) | 8           | 297.12     | Khá tốt (tách theo câu)           |
| `rag_system_design.md`  | RecursiveChunker (`recursive`)   | 16          | 147.56     | Rất tốt (theo markdown structure) |

### Strategy Của Tôi

**Loại:** `Semantic Chunker` (SemanticChunker)

**Mô tả cách hoạt động:**
> Bộ chia này phân tách văn bản dựa trên sự chuyển đổi ngữ nghĩa giữa các câu (Semantic Transition). Đầu tiên, nó chia văn bản thành các câu đơn lẻ và sinh vector embedding cho mỗi câu. Sau đó, nó tính toán độ tương đồng cosine giữa các câu liên tiếp. Các câu sẽ được gom nhóm vào chung một chunk nếu độ tương đồng lớn hơn hoặc bằng một ngưỡng nhất định (`threshold = 0.5`) và tổng độ dài của chunk không vượt quá giới hạn tối đa (`max_chunk_size = 1000`). Nếu độ tương đồng tụt xuống dưới ngưỡng hoặc vượt quá kích thước tối đa, một chunk mới sẽ được bắt đầu.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu của nhóm chứa nhiều mô tả chi tiết về các phòng nghỉ dưỡng, dịch vụ spa, và bảng giá vui chơi giải trí với các thông tin có sự chuyển dịch chủ đề rõ rệt. Việc sử dụng `Semantic Chunker` đảm bảo các câu có cùng chủ đề được gom nhóm vào chung một chunk một cách linh hoạt, đạt độ kết dính ngữ nghĩa (`Coherence`) rất cao (0.782), từ đó giúp RAG Agent tìm kiếm chính xác và trả lời đầy đủ thông tin hơn so với việc cắt cố định.

**Code snippet (nếu custom):**

```python
# Cấu hình khởi tạo Semantic Chunker trong benchmark
semantic_chunker = SemanticChunker(embedding_fn=embedder, threshold=0.5, max_chunk_size=1000)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `rag_system_design.md` | `fixed_size` (Baseline) | 12 | 199.25 | Trung bình (bị mất ngữ cảnh câu ranh giới) |
| `rag_system_design.md` | `semantic` (Của tôi) | 15 | 158.00 | Rất cao (Coherence đạt 0.782, bảo toàn ngữ nghĩa tuyệt đối) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Semantic Chunker | 10/10 (Hit@3: 100%) | Coherence cao nhất (0.782), các câu được gom theo ngữ nghĩa đồng nhất, tránh cắt đứt ngữ cảnh. | Tốc độ xử lý chậm hơn (11.86s) vì phải tính embedding cho từng câu đơn lẻ. |
| Thành viên 2 | Recursive Chunker | 10/10 (Hit@3: 100%) | Tốc độ phân tách nhanh (6.17s), bảo toàn cấu trúc tài liệu (đoạn văn, tiêu đề) tốt. | Phân chia tĩnh, không linh hoạt theo ngữ nghĩa thực tế. |
| Thành viên 3 | Agentic Chunker (Gemini) | 6/10 (Hit@3: 60%) | Phân tích cấu trúc rất thông minh theo chủ đề tự nhiên của mô hình ngôn ngữ lớn LLM. | Rất chậm (20.38s), tốn chi phí gọi API và phụ thuộc mạng. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Đối với domain của nhóm (tra cứu dịch vụ, phòng nghỉ và vé vui chơi giải trí), bộ chia **Semantic Chunker** và **Recursive Chunker** mang lại chất lượng retrieval tốt nhất (Hit@3 đạt 100%). Trong đó, **Semantic Chunker** tối ưu nhất về mặt ngữ nghĩa nhờ gom các câu có cùng chủ đề một cách linh hoạt (Coherence đạt 0.782), giúp câu trả lời của RAG Agent luôn đầy đủ và không bị đứt mạch thông tin dịch vụ. Tuy nhiên, nếu cần tối ưu về thời gian xử lý và tài nguyên thì **Recursive Chunker** là lựa chọn thay thế rất hiệu quả.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)` để tìm kiếm ranh giới câu, sau đó loại bỏ khoảng trắng dư thừa ở đầu/cuối của từng câu và lọc bỏ câu rỗng. Tiếp theo nhóm các câu lại theo nhóm có kích thước tối đa là `max_sentences_per_chunk` và ghép chúng lại bằng khoảng trắng để tạo thành các chunk hoàn chỉnh.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng thuật toán chia đệ quy với danh sách ký tự phân tách có độ ưu tiên giảm dần. Base case thứ nhất là độ dài đoạn văn bản nhỏ hơn hoặc bằng `chunk_size` (trả về chính đoạn đó), base case thứ hai là danh sách ký tự phân tách rỗng (cắt cố định theo số ký tự). Ở bước đệ quy, ta chia nhỏ văn bản bằng ký tự phân tách hiện tại, đệ quy xử lý các phần văn bản quá lớn, sau đó gộp các mảnh con lại sao cho mỗi chunk mới tạo ra không vượt quá `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Hỗ trợ lưu trữ ChromaDB (sử dụng `collection.add`) và fallback sang in-memory (lưu trữ danh sách các dict vào `self._store`). Hàm `search` tính toán độ tương đồng cosine thông qua tích vô hướng (`_dot`) giữa các vector đã được chuẩn hóa, sau đó sắp xếp giảm dần và lấy ra top-k kết quả phù hợp nhất.

**`search_with_filter` + `delete_document`** — approach:
> Hàm `search_with_filter` thực hiện kỹ thuật tiền lọc (pre-filtering), duyệt qua cơ sở dữ liệu in-memory để lọc ra tất cả các bản ghi thỏa mãn điều kiện lọc metadata trước, rồi mới tính toán tương đồng ngữ nghĩa trên tập con đó. Hàm `delete_document` thực hiện lọc bỏ toàn bộ các bản ghi trong danh sách `self._store` có trường metadata `doc_id` trùng khớp với `doc_id` của tài liệu cần xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Thực hiện quy trình RAG chuẩn: gọi `self.store.search` để lấy ra top-k chunk tài liệu tương đồng nhất, ghép nối nội dung của chúng bằng dấu xuống dòng kép `\n\n` làm ngữ cảnh (context), sau đó đưa vào prompt có cấu trúc định sẵn (cùng câu hỏi của người dùng) và chuyển tiếp cho `llm_fn` để sinh ra câu trả lời.

### Test Results

```text
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Học máy là một lĩnh vực của trí tuệ nhân tạo. | Học máy là một lĩnh vực của trí tuệ nhân tạo. | high | 1.00000 | Đúng |
| 2 | Học máy là một lĩnh vực của trí tuệ nhân tạo. | Trí tuệ nhân tạo bao gồm cả học máy. | high | 0.03478 | Sai |
| 3 | Học máy là một lĩnh vực của trí tuệ nhân tạo. | Hôm nay trời nắng đẹp quá. | low | 0.14471 | Đúng |
| 4 | Machine learning is a subfield of AI. | Machine learning is a subset of artificial intelligence. | high | -0.03613 | Sai |
| 5 | Tôi thích ăn táo. | Quả táo này rất ngon. | high | 0.04601 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là các cặp câu có ý nghĩa tương tự nhau (như Cặp 2, Cặp 4, Cặp 5) lại có điểm tương đồng thực tế rất thấp (gần bằng 0), thậm chí cặp câu không liên quan gì đến nhau (Cặp 3) lại có điểm tương đồng cao hơn. Điều này chứng minh rằng `MockEmbedder` giả lập trong bài kiểm tra chỉ sử dụng hàm băm MD5 ngẫu nhiên để sinh vector nên không thể phản ánh thực tế ý nghĩa của ngôn ngữ. Đối với các mô hình embedding thực sự (như SentenceTransformers hay OpenAI), chúng được huấn luyện trên lượng văn bản khổng lồ nên có thể biểu diễn được ngữ nghĩa tinh tế của từ và câu trong không gian vector.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                                                                         | Gold Answer                                                                                                                              |
| -----| -----------------------------------------------------------------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | Giá vé hiện tại của VinWonders Nha Trang là bao nhiêu?                                        | Giá hiện tại **500.000 đ** (giá gốc 600.000 đ), theo trang VinWonders Nha Trang.                                                         |
| 2   | Aquafield Nha Trang có những phòng trị liệu xông hơi nào?                                     | Có 7 phòng: Băng tuyết, Gỗ bách (Hinoki), Sương mây, Đá muối Himalaya, Bulgama, Than củi, Hoàng thổ — mỗi phòng có nhiệt độ/độ ẩm riêng. |
| 3   | Combo 2N1Đ Vinpearl Hotel Cần Thơ bao gồm những dịch vụ gì?                                   | 01 đêm phòng Deluxe (2 người lớn + 2 trẻ dưới 4 tuổi), 01 bữa sáng, miễn phụ thu cuối tuần, thuế phí dịch vụ.                            |
| 4   | Voucher golf Sunrise áp dụng tee-time và ngày nào? *(cần filter `product_type=golf_voucher`)* | Sunrise: tee-time **trước 8:00**, nhóm từ 2 khách, **thứ 2–thứ 6**, không áp dụng ngày lễ 30/4, 01/05, 02/9.                             |
| 5   | Night Safari tại Vinpearl Safari Phú Quốc là gì?                                              | Hành trình khám phá động vật về đêm bằng xe điện — trải nghiệm safari ban đêm duy nhất tại Việt Nam.                                     |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Giá vé hiện tại của VinWonders Nha Trang là bao nhiêu? | Đến VinWonders Nha Trang, bạn còn được trải nghiệm một trong những tuyến cáp treo trên biển dài nhất thế giới, đưa bạn vượt qua đại dương... | 0.6958 | No | Không đề cập đến giá vé cụ thể của VinWonders Nha Trang, chỉ cho biết có thể đặt vé với giá tốt nhất và đã bao gồm cáp treo/tàu cao tốc. |
| 2 | Aquafield Nha Trang có những phòng trị liệu xông hơi nào? | Than củi giúp trung hòa các cơ quan trong cơ thể bị axit hóa và tạo ra các ion âm để thúc đẩy quá trình phục hồi sức khỏe, thải độc, lưu thông máu... | 0.7830 | Yes | Có các phòng trị liệu xông hơi: phòng Hoàng Thổ, phòng Đá muối Himalaya, phòng Bulgama, phòng Than củi. |
| 3 | Combo 2N1Đ Vinpearl Hotel Cần Thơ bao gồm những dịch vụ gì? | Ngoài ra, Delta Lobby Lounge cũng là nơi “đi trốn" tuyệt vời cho những ai tìm kiếm không gian tĩnh lặng, thưởng thức bữa tiệc trà chiều... | 0.6631 | No | Các dịch vụ bao gồm không được liệt kê cụ thể (mục "Bao gồm" trống), chỉ đề cập gợi ý 1 đêm Deluxe và bữa sáng. |
| 4 | Voucher golf Sunrise áp dụng tee-time và ngày nào? | Unilimited - Chơi không giới hạn hố - Chơi không hạn chế số hố từ thứ 2 đến thứ 6... Twilight - Áp dụng... | 0.7102 | Yes | Áp dụng cho Tee-time trước 8:00 và các ngày trong tuần từ thứ 2 đến thứ 6 (không áp dụng lễ tết 30/4, 1/5, 2/9). |
| 5 | Night Safari tại Vinpearl Safari Phú Quốc là gì? | # Vinpearl Safari Phú Quốc - URL: <https://booking.vinpearl.com/vi-VND/tour/ve-vao-cua-truc-tiep-vinpearl-safari-phu-quoc-vw00614> - Giá gốc... | 0.7077 | No | Không tìm thấy thông tin nào đề cập đến "Night Safari" tại Vinpearl Safari Phú Quốc trong ngữ cảnh được cung cấp. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Từ anh Đào Văn Tuân, tôi học được giải pháp **Parent-Child Chunker** (parent=800, child=200). Cách làm này rất hay vì vừa giúp sinh vector embedding tối ưu chính xác ở kích thước nhỏ (child), vừa cung cấp ngữ cảnh đầy đủ (parent) cho mô hình ngôn ngữ lớn LLM để trả lời câu hỏi mà không làm mất thông tin xung quanh. Ngoài ra, việc kết hợp metadata filter của anh Phan Võ Trọng Tiển giúp tăng tốc độ tìm kiếm và độ chính xác rất nhiều.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Từ buổi demo của các nhóm khác, tôi học được rằng việc sử dụng **Hybrid Search** (kết hợp mật độ từ khóa BM25 và Vector Search) là rất quan trọng để tìm chính xác các thông tin có tính đặc thù cao (như tên các tour, giá vé, mã dịch vụ). Một số nhóm cũng áp dụng thành công **Query Expansion** (mở rộng câu hỏi) để cải thiện độ phủ ngữ nghĩa trước khi đưa vào cơ sở dữ liệu vector.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ cải thiện chiến lược chunking bằng cách kết hợp thêm metadata có cấu trúc và điều chỉnh kích thước chunk linh hoạt cho từng phần (ví dụ phần Bảng giá/Bao gồm dịch vụ cần được giữ nguyên vẹn). Đồng thời, tôi sẽ áp dụng kỹ thuật Hybrid Search (kết hợp Vector Search với Keyword Search như BM25) để đảm bảo các từ khóa quan trọng và đặc thù như "Night Safari" hay con số giá vé không bị trôi đi dưới ngưỡng tương đồng của mô hình Embedding thông thường.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá                      |
| -----------------------------| ---------| ---------------------------------------|
| Warm-up                     | Cá nhân | 5 / 5                                 |
| Document selection          | Nhóm    | / 10                                  |
| Chunking strategy           | Nhóm    | / 15                                  |
| My approach                 | Cá nhân | 10 / 10                               |
| Similarity predictions      | Cá nhân | 5 / 5                                 |
| Results                     | Cá nhân | 10 / 10                               |
| Core implementation (tests) | Cá nhân | 30 / 30                               |
| Demo                        | Nhóm    | / 5                                   |
| **Tổng**                    |         | **60 / 100** (Phần cá nhân đạt 60/60) |
