# /* 文件名: backend/rag_handler.py, 版本号: 5.1 (离线简化版) */
import ollama
from qdrant_client import QdrantClient, models
from typing import List, Dict, Union, Generator, Any

import logging
import time
import os
import pickle

from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
import shutil
import config
from router import QueryRouter
from document_parser import ParsedElement
from chunker import create_multi_vector_chunks

logger = logging.getLogger(__name__)

BM25_MODELS_DIR = "/app/data/bm25_models"
os.makedirs(BM25_MODELS_DIR, exist_ok=True)


os.environ["FLASHRANK_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# 避免 FlashRank 误触发下载逻辑（伪装本地缓存）
flashrank_cache_dir = "/root/.cache/flashrank/models/ms-marco-MiniLM-L-12-v2"
local_model_dir = "/app/models/reranker"

if os.path.exists(local_model_dir) and not os.path.exists(flashrank_cache_dir):
    print(f"⚙️ 同步本地 reranker 模型到 FlashRank 缓存目录: {flashrank_cache_dir}")
    shutil.copytree(local_model_dir, flashrank_cache_dir, dirs_exist_ok=True)

# 如果缓存中有 onnx 文件，说明可以离线加载
onnx_path = os.path.join(flashrank_cache_dir, "onnx")
if os.path.exists(onnx_path):
    print(f"✅ 检测到离线 FlashRank 模型缓存，禁用联网下载。")
    os.environ["FLASHRANK_FORCE_LOCAL"] = "1"

class RAGHandler:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)
        
        # --- 简化嵌入逻辑，仅使用Ollama ---
        self.embedding_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        logger.info(f"使用Ollama嵌入模型: {config.EMBEDDING_MODEL_NAME}")
        self.embedding_dim = self._get_ollama_embedding_dimension()

        self.llm_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        

# --- 【Reranker 加载逻辑：离线兼容模式】 ---
        model_path = config.RERANK_MODEL_PATH
        logger.info(f"准备从本地路径加载 Reranker 模型: {model_path}")

        if not os.path.isdir(model_path):
            logger.error(f"‼️ 致命错误: 路径 '{model_path}' 不存在或不是一个目录！")
            raise RuntimeError(f"Reranker 模型路径无效: {model_path}")

        logger.info(f"✅ 路径 '{model_path}' 存在并且是一个目录。")
        try:
            files_in_dir = os.listdir(model_path)
            logger.info(f"   目录下的文件列表: {files_in_dir}")
            if not files_in_dir:
                logger.error("   ‼️ 错误: Reranker 模型目录为空！请检查模型文件。")
        except Exception as e:
            logger.error(f"   ‼️ 错误: 无法读取目录 '{model_path}' 内容: {e}")

        try:
            # --- 递归查找 onnx 文件 ---
            onnx_model_path = None
            for root, dirs, files in os.walk(model_path):
                for f in files:
                    if f.endswith(".onnx"):
                        onnx_model_path = os.path.join(root, f)
                        break
                if onnx_model_path:
                    break

            if not onnx_model_path:
                raise FileNotFoundError(f"在 {model_path} 中未找到 .onnx 模型文件！")

            logger.info(f"✅ 在递归搜索中找到 FlashRank ONNX 模型文件: {onnx_model_path}")

            # --- 构建 FlashRank 本地缓存路径 ---
            cache_dir = os.path.expanduser("~/.cache/flashrank/models/ms-marco-MiniLM-L-12-v2")
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)

            # --- 强制复制到缓存目录 ---
            logger.info(f"⚙️ 将本地模型同步到 FlashRank 缓存目录: {cache_dir}")
            shutil.copytree(model_path, cache_dir, dirs_exist_ok=True)

            # --- 设置环境变量，强制离线模式 ---
            os.environ["FLASHRANK_OFFLINE"] = "1"
            os.environ["FLASHRANK_CACHE_DIR"] = os.path.expanduser("~/.cache/flashrank")

            # --- 初始化 Reranker ---
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
            logger.info("✅ 成功从本地缓存加载 FlashRank reranker 模型（离线模式启用）。")

        except Exception as e:
            logger.exception(f"‼️ Reranker 模型加载失败: {e}")
            raise RuntimeError("Reranker 初始化失败")

        self.query_router = QueryRouter(
            qdrant_client=self.qdrant_client,
            embedding_model=self.embedding_client,
            llm_client=self.llm_client
        )
        
        self.bm25_models_cache = {}
        for collection_name in config.AVAILABLE_COLLECTIONS:
            self._ensure_collection_exists(collection_name)
    
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

    def get_answer(self, query: str, chat_history: List[Dict[str, str]]) -> Generator[str, None, None]:
        initial_candidates = self.hybrid_retrieve(query, top_k=20)
        if not initial_candidates:
            logger.warning("No candidates found after initial retrieval. Generating answer without context.")
            yield from self.generate_answer(query, [], chat_history)
            return
        rerank_request = RerankRequest(query=query, passages=initial_candidates)
        reranked_results = self.reranker.rerank(rerank_request)
        final_context_payloads = reranked_results[:5]
        final_context = [item['meta']['parent_content'] for item in final_context_payloads]
        yield from self.generate_answer(query, final_context, chat_history)

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
            self._rebuild_bm25_model_for_collection(collection_name)
    
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
            self._rebuild_bm25_model_for_collection(collection_name)
            return True
        except Exception as e:
            return False
    
    def generate_answer(self, query: str, context: List[str], chat_history: List[Dict[str, str]]):
        prompt = self._build_prompt(query, context, chat_history)
        yield from self._generate_ollama(prompt)

    def _build_prompt(self, query: str, context: List[str], chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        context_str = "\n---\n".join(context) if context else "无"
        history_str = "\n".join([f"用户: {item['user']}\n助手: {item['assistant']}" for item in chat_history])
        system_message = f"""你是一个专业的AI知识库助手。请严格遵循以下规则回答问题：
1.  **忠于原文**: 你的首要任务是基于下面提供的“[知识库上下文]”来回答问题。答案必须完全来源于上下文，不得进行任何形式的推理、联想或添加上下文之外的信息。
2.  **明确引用**: 在回答时，清楚地说明你的信息来源于知识库。例如，你可以说：“根据知识库中的信息...”。
3.  **坦诚未知**: 如果“[知识库上下文]”为空，或者上下文内容与用户问题完全无关，无法找到答案，你**必须**明确地回答：“抱歉，我在知识库中没有找到与您问题相关的确切信息。” **严禁**使用自己的内置知识回答。
4.  **整合信息**: 如果多段上下文都与问题相关，请将它们进行逻辑整合，用流畅的语言进行回复，而不是简单地罗列。
---
[知识库上下文]
{context_str}
---
"""
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

rag_handler = RAGHandler()