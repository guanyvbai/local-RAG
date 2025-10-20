# /* 文件名: backend/rag_handler.py, 版本号: 6.0 (SentenceTransformer 替换版) */
from qdrant_client import QdrantClient, models
from typing import List, Dict, Union, Generator, Any, Optional
import logging
import time
import os
import pickle
from collections import defaultdict
from rank_bm25 import BM25Okapi
# --- 【核心替换】导入 CrossEncoder ---
from sentence_transformers import CrossEncoder
import shutil
import config
from router import QueryRouter
from document_parser import ParsedElement
from chunker import create_multi_vector_chunks
from ollama_client import get_ollama_client
from threading import RLock

logger = logging.getLogger(__name__)

BM25_MODELS_DIR = "/app/data/bm25_models"
os.makedirs(BM25_MODELS_DIR, exist_ok=True)


class RAGHandler:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)
        
        self.embedding_client = get_ollama_client()
        logger.info(f"使用Ollama嵌入模型: {config.EMBEDDING_MODEL_NAME}")
        self.embedding_dim = self._get_ollama_embedding_dimension()

        self.llm_client = self.embedding_client
        
        # --- 【核心替换】加载 sentence-transformers CrossEncoder 模型 ---
        model_path = "/app/models/cross-encoder" # 对应 docker-compose.yml 的挂载路径
        logger.info(f"准备从本地路径加载 CrossEncoder Reranker 模型: {model_path}")

        if not os.path.isdir(model_path):
            logger.error(f"‼️ 致命错误: Reranker 模型路径 '{model_path}' 不存在或不是一个目录！")
            raise RuntimeError(f"Reranker 模型路径无效: {model_path}")

        try:
            # 直接从本地文件夹路径加载模型
            self.reranker = CrossEncoder(model_path)
            logger.info("✅ 成功从本地路径加载 CrossEncoder reranker 模型。")
        except Exception as e:
            logger.exception(f"‼️ Reranker 模型加载失败: {e}")
            raise RuntimeError("Reranker 初始化失败")

        self.query_router = QueryRouter(
            qdrant_client=self.qdrant_client,
            embedding_model=self.embedding_client,
            llm_client=self.llm_client
        )
        
        self.bm25_models_cache = {}
        self._bm25_pending_counts = defaultdict(int)
        self._bm25_last_rebuild = {collection: time.time() for collection in config.AVAILABLE_COLLECTIONS}
        self.bm25_rebuild_threshold = getattr(config, "BM25_REBUILD_THRESHOLD", 5)
        self.bm25_rebuild_interval = getattr(config, "BM25_REBUILD_INTERVAL_SECONDS", 300)
        for collection_name in config.AVAILABLE_COLLECTIONS:
            self._ensure_collection_exists(collection_name)
    
    def get_answer(self, query: str, chat_history: List[Dict[str, str]]) -> Generator[str, None, None]:
        initial_candidates = self.hybrid_retrieve(query, top_k=20)
        if not initial_candidates:
            logger.warning("No candidates found after initial retrieval. Generating answer without context.")
            yield from self.generate_answer(query, [], chat_history)
            return

        # --- 【核心替换】使用 CrossEncoder 进行 Rerank ---
        passages = [item['text'] for item in initial_candidates]
        sentence_pairs = [[query, passage] for passage in passages]
        
        # 预测得分
        scores = self.reranker.predict(sentence_pairs)
        
        # 组合 passage 和 score
        reranked_results = [{"passage": passage, "score": score, "original_doc": initial_candidates[i]} for i, (passage, score) in enumerate(zip(passages, scores))]
        
        # 按分数降序排序
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 提取前5个结果的 payload
        final_context_payloads = [item['original_doc']['meta'] for item in reranked_results[:5]]
        final_context = [payload.get('parent_content', '') for payload in final_context_payloads]

        yield from self.generate_answer(query, final_context, chat_history)

    def _get_bm25_model_path(self, collection_name: str) -> str:
        return os.path.join(BM25_MODELS_DIR, f"{collection_name}_bm25.pkl")

    def _load_bm25_model(self, collection_name: str) -> BM25Okapi:
        if collection_name in self.bm25_models_cache:
            return self.bm25_models_cache[collection_name]
        model_path = self._get_bm25_model_path(collection_name)
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                logger.info(f"Loading BM25 model for collection '{collection_name}' from disk.")
                model = pickle.load(f)
                self.bm25_models_cache[collection_name] = model
                return model
        return None

    def _save_bm25_model(self, collection_name: str, model: BM25Okapi):
        self.bm25_models_cache[collection_name] = model
        model_path = self._get_bm25_model_path(collection_name)
        with open(model_path, "wb") as f:
            logger.info(f"Saving BM25 model for collection '{collection_name}' to disk.")
            pickle.dump(model, f)
    
    def _rebuild_bm25_model_for_collection(self, collection_name: str):
        logger.info(f"Rebuilding BM25 model for collection '{collection_name}'...")
        try:
            all_child_docs = []
            scroll_result, next_page = self.qdrant_client.scroll(
                collection_name=collection_name, limit=1000, with_payload=["child_content"]
            )
            while scroll_result:
                all_child_docs.extend([hit.payload['child_content'] for hit in scroll_result if hit.payload.get('child_content')])
                if not next_page:
                    break
                scroll_result, next_page = self.qdrant_client.scroll(
                    collection_name=collection_name, limit=1000, with_payload=["child_content"], offset=next_page
                )
            
            if all_child_docs:
                tokenized_corpus = [doc.split() for doc in all_child_docs]
                bm25 = BM25Okapi(tokenized_corpus)
                self._save_bm25_model(collection_name, bm25)
                logger.info(f"Successfully rebuilt and saved BM25 model for '{collection_name}' with {len(all_child_docs)} documents.")
            else:
                logger.warning(f"No documents found in '{collection_name}' to build BM25 model.")
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 model for '{collection_name}': {e}", exc_info=True)

    def _ensure_collection_exists(self, collection_name: str):
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"集合 '{collection_name}' 已存在，将直接使用。")
        except Exception as e:
            if "not found" in str(e).lower() or "status_code=404" in str(e):
                logger.warning(f"集合 '{collection_name}' 不存在，现在开始创建...")
                self.create_collection(collection_name)
            else:
                logger.error(f"检查集合 '{collection_name}' 时发生未知错误: {e}", exc_info=True)

    def create_collection(self, collection_name: str) -> bool:
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
            )
            logger.info(f"成功创建混合检索集合: {collection_name}")
            model_path = self._get_bm25_model_path(collection_name)
            if os.path.exists(model_path):
                os.remove(model_path)
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"集合 '{collection_name}' 已存在，无需重复创建。")
                return True
            logger.error(f"创建集合失败 {collection_name}: {e}", exc_info=True)
            return False

    def hybrid_retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        routed_collection = self.query_router.route_query(query)
        if not routed_collection: return []
        dense_vector = self._generate_embeddings(query)
        if not dense_vector: return []
        dense_hits = self.qdrant_client.search(
            collection_name=routed_collection,
            query_vector=models.NamedVector(name="dense", vector=dense_vector),
            limit=top_k,
            with_payload=True
        )
        sparse_hits = []
        bm25_model = self._load_bm25_model(routed_collection)
        if bm25_model:
            logger.info(f"Found BM25 model for '{routed_collection}'. Performing sparse search.")
            query_keywords = query.split()
            sparse_vector = self._get_bm25_vector_for_query(bm25_model, query_keywords)
            if sparse_vector.indices:
                sparse_hits = self.qdrant_client.search(
                    collection_name=routed_collection,
                    query_vector=models.NamedSparseVector(name="bm25", vector=sparse_vector),
                    limit=top_k,
                    with_payload=True
                )
            else:
                logger.warning("Query keywords not found in BM25 vocabulary. Skipping sparse search.")
        else:
            logger.warning(f"No BM25 model found for '{routed_collection}'. Skipping sparse search.")
        combined_passages = {}
        for hit in dense_hits + sparse_hits:
            parent_id = hit.payload.get('parent_id')
            if parent_id and parent_id not in combined_passages:
                combined_passages[parent_id] = {
                    "id": parent_id,
                    "text": hit.payload.get("child_content"),
                    "meta": hit.payload
                }
        return list(combined_passages.values())

    def process_and_embed_document(self, parsed_elements: List[ParsedElement], filename: str, collection_name: str):
        self._ensure_collection_exists(collection_name)
        multi_vector_chunks = create_multi_vector_chunks(parsed_elements, filename)
        if not multi_vector_chunks:
            return
        
        points_to_upsert = []
        for parent_chunk, child_chunks in multi_vector_chunks:
            parent_payload = parent_chunk.to_qdrant_payload()
            parent_payload["collection"] = collection_name
            for child_chunk in child_chunks:
                dense_vector = self._generate_embeddings(child_chunk.content)
                if dense_vector:
                    child_payload = parent_payload.copy()
                    child_payload["child_content"] = child_chunk.content
                    child_payload["content_type"] = child_chunk.content_type
                    points_to_upsert.append(models.PointStruct(
                        id=child_chunk.chunk_id,
                        vector={"dense": dense_vector}, 
                        payload=child_payload
                    ))
        if points_to_upsert:
            self.qdrant_client.upsert(collection_name=collection_name, points=points_to_upsert, wait=True)
            self._maybe_rebuild_bm25(collection_name)
    
    def _get_bm25_vector_for_query(self, bm25_model: BM25Okapi, query_keywords: List[str]) -> models.SparseVector:
        if not hasattr(bm25_model, 'word_to_id'):
            bm25_model.word_to_id = {word: i for i, word in enumerate(bm25_model.idf)}
        indices = [bm25_model.word_to_id[word] for word in query_keywords if word in bm25_model.word_to_id]
        values = [bm25_model.idf[word] for word in query_keywords if word in bm25_model.word_to_id]
        return models.SparseVector(indices=indices, values=values)

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            model_path = self._get_bm25_model_path(collection_name)
            if os.path.exists(model_path):
                os.remove(model_path)
            return True
        except Exception as e:
            logger.error(f"删除集合失败 {collection_name}: {e}")
            return False

    def delete_document(self, filename: str, collection_name: str) -> bool:
        try:
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=models.Filter(must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=filename))
                ])),
                wait=True
            )
            self._maybe_rebuild_bm25(collection_name, force=True)
            return True
        except Exception as e:
            return False

    def flush_bm25_models(self, collection_name: str):
        """Force rebuild the BM25 model for a collection."""
        self._maybe_rebuild_bm25(collection_name, force=True)

    def _maybe_rebuild_bm25(self, collection_name: str, force: bool = False):
        if force:
            self._rebuild_bm25_model_for_collection(collection_name)
            self._bm25_pending_counts[collection_name] = 0
            self._bm25_last_rebuild[collection_name] = time.time()
            return

        self._bm25_pending_counts[collection_name] += 1
        pending = self._bm25_pending_counts[collection_name]
        last_rebuild = self._bm25_last_rebuild.get(collection_name)
        now = time.time()
        if last_rebuild is None:
            last_rebuild = now
            self._bm25_last_rebuild[collection_name] = now

        should_rebuild = (
            pending >= self.bm25_rebuild_threshold
            or (pending > 0 and (now - last_rebuild) >= self.bm25_rebuild_interval)
        )
        if should_rebuild:
            self._rebuild_bm25_model_for_collection(collection_name)
            self._bm25_pending_counts[collection_name] = 0
            self._bm25_last_rebuild[collection_name] = now
    
    def generate_answer(self, query: str, context: List[str], chat_history: List[Dict[str, str]]):
        prompt = self._build_prompt(query, context, chat_history)
        yield from self._generate_ollama(prompt)

    def _build_prompt(self, query: str, context: List[str], chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        context_str = "\n---\n".join(context) if context else "无"
        history_str = "\n".join([f"用户: {item['user']}\n助手: {item['assistant']}" for item in chat_history])

        if context:
            system_message = f"""你是一个专业的AI知识库助手。请严格遵循以下规则回答问题：
1.  **忠于原文**: 你的首要任务是基于下面提供的“[知识库上下文]”来回答问题。答案必须完全来源于上下文，不得进行任何形式的推理、联想或添加上下文之外的信息。
2.  **明确引用**: 在回答时，清楚地说明你的信息来源于知识库。例如，你可以说：“根据知识库中的信息...”。
3.  **整合信息**: 如果多段上下文都与问题相关，请将它们进行逻辑整合，用流畅的语言进行回复，而不是简单地罗列。
---
[知识库上下文]
{context_str}
---
"""
        else:
            system_message = """你是一个通用型大型语言模型助手。当前知识库中没有可用的上下文，请直接基于你掌握的常识与公开知识回答用户问题，给出清晰、流畅且有帮助的回复。"""
        messages = [{"role": "system", "content": system_message}]
        if history_str:
            messages.append({"role": "system", "content": f"[历史对话参考]\n{history_str}"})
        messages.append({"role": "user", "content": query})
        return messages

    def _generate_ollama(self, messages: List[Dict[str, str]]):
        stream = self.llm_client.chat(model=config.OLLAMA_MODEL_NAME, messages=messages, stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            if content:
                yield content

    def _get_ollama_embedding_dimension(self) -> int:
        try:
            logger.info(f"Pinging Ollama to get embedding dimension for model: {config.EMBEDDING_MODEL_NAME}")
            response = self.embedding_client.embeddings(model=config.EMBEDDING_MODEL_NAME, prompt="test")
            dim = len(response["embedding"])
            logger.info(f"Successfully determined embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.error(f"无法从 Ollama 获取嵌入维度: {e}", exc_info=True)
            logger.warning("将使用备用维度 1024。")
            return 1024

    def _generate_embeddings(self, text: Union[str, object]) -> List[float]:
        text_str = str(text)
        try:
            response = self.embedding_client.embeddings(model=config.EMBEDDING_MODEL_NAME, prompt=text_str)
            return response["embedding"]
        except Exception as e:
            logger.error(f"为文本生成嵌入时出错: '{text_str[:50]}...' - {e}")
            return []
    
    def list_collections(self) -> List[str]:
        try:
            collections_response = self.qdrant_client.get_collections()
            return [collection.name for collection in collections_response.collections]
        except Exception as e:
            logger.error(f"无法从 Qdrant 获取集合列表: {e}")
            return []
    
    def list_documents(self) -> List[Dict[str, str]]:
        all_docs = {}
        try:
            collections = self.list_collections()
            for collection_name in collections:
                scroll_result, _ = self.qdrant_client.scroll(
                    collection_name=collection_name, limit=10000,
                    with_payload=True, with_vectors=False
                )
                for hit in scroll_result:
                    source = hit.payload.get('source')
                    if source and source not in all_docs:
                        all_docs[source] = {"name": source, "collection": collection_name}
            return sorted(list(all_docs.values()), key=lambda x: x['name'])
        except Exception as e:
            logger.error(f"Failed to list documents from Qdrant: {e}")
            return []

_rag_handler_lock = RLock()
_rag_handler_instance: Optional[RAGHandler] = None
_rag_handler_error: Optional[BaseException] = None


def get_rag_handler(force_refresh: bool = False) -> RAGHandler:
    global _rag_handler_instance, _rag_handler_error

    if _rag_handler_instance is not None and not force_refresh:
        return _rag_handler_instance

    with _rag_handler_lock:
        if _rag_handler_instance is not None and not force_refresh:
            return _rag_handler_instance

        try:
            handler = RAGHandler()
        except Exception as exc:
            _rag_handler_error = exc
            logger.error("RAGHandler 初始化失败。", exc_info=True)
            raise
        else:
            _rag_handler_instance = handler
            _rag_handler_error = None
            return handler


def get_rag_handler_status() -> Dict[str, Any]:
    """Return current readiness status for monitoring endpoints."""

    error_message = None
    if _rag_handler_error is not None:
        error_message = str(_rag_handler_error)

    return {
        "ready": _rag_handler_instance is not None,
        "error": error_message,
    }
