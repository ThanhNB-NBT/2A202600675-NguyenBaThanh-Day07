from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            # Khởi tạo client của ChromaDB trong bộ nhớ (in-memory) và lấy/tạo collection tương ứng
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Tạo một bản ghi chuẩn hóa chứa thông tin tài liệu và vector embedding của nó
        metadata = dict(doc.metadata) if doc.metadata else {}
        # Đảm bảo trường 'doc_id' tồn tại trong metadata để phục vụ việc lọc và xóa
        if 'doc_id' not in metadata:
            metadata['doc_id'] = doc.id
            
        # Tính toán embedding cho nội dung của tài liệu
        embedding = self._embedding_fn(doc.content)
        
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # Tìm kiếm độ tương đồng trên bộ nhớ RAM (in-memory)
        if not records:
            return []
            
        # Tính toán embedding của câu truy vấn
        query_embedding = self._embedding_fn(query)
        
        results = []
        for record in records:
            # Tính tích vô hướng (dot product) giữa vector truy vấn và vector tài liệu
            score = _dot(query_embedding, record["embedding"])
            results.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score
            })
            
        # Sắp xếp các kết quả theo điểm số giảm dần
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Trả về top_k kết quả hàng đầu
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            # Sử dụng ChromaDB
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for doc in docs:
                record = self._make_record(doc)
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
                
            if ids:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
        else:
            # Sử dụng lưu trữ trong bộ nhớ (in-memory fallback)
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            # Tìm kiếm sử dụng ChromaDB
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            formatted_results = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                # ChromaDB có thể trả về 'distances'. Chúng ta chuyển đổi thành score (1.0 - distance nếu là cosine distance)
                distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(ids)
                
                for i in range(len(ids)):
                    formatted_results.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "score": 1.0 - distances[i] if distances else 0.0
                    })
            return formatted_results
        else:
            # Tìm kiếm trên bộ nhớ RAM
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # Nếu không cung cấp bộ lọc metadata, thực hiện tìm kiếm bình thường
        if not metadata_filter:
            return self.search(query, top_k)
            
        if self._use_chroma and self._collection is not None:
            # Tìm kiếm lọc metadata với ChromaDB bằng mệnh đề where
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            
            formatted_results = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(ids)
                
                for i in range(len(ids)):
                    formatted_results.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "score": 1.0 - distances[i] if distances else 0.0
                    })
            return formatted_results
        else:
            # Lọc trước (pre-filtering) trên danh sách lưu trữ in-memory
            filtered_records = []
            for record in self._store:
                match = True
                for key, val in metadata_filter.items():
                    if record["metadata"].get(key) != val:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
                    
            # Tìm kiếm độ tương đồng trên các bản ghi đã lọc
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            # Xóa các bản ghi tương ứng trong ChromaDB
            count_before = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            count_after = self._collection.count()
            return count_after < count_before
        else:
            # Xóa các bản ghi tương ứng trong bộ nhớ RAM
            initial_size = len(self._store)
            self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_size
