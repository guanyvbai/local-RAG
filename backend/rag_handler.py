# 文件名: backend/rag_handler.py, 版本号: 3.2
"""
RAG (Retrieval-Augmented Generation) 核心处理器。
【V3.2 修复版】:
- 修正了由于缺少 'Any' 类型导入而导致的 NameError。
- 修正了错误引用 config.EMBEDDING_MODEL 的 AttributeError。
- 实现了混合检索（Dense + Sparse BM25）。
- 实现了多向量检索（摘要检索 -> 父块返回）。
- 集成 Reranker 对检索结果进行精排。
"""
import ollama
import openai
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Batch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Generator, Any # <--- 【修复】添加了 Any

import logging
import time

# --- 新增导入 ---
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

import config
from router import QueryRouter
from document_parser import ParsedElement
# 【修改】导入新的 chunker 函数和类
from chunker import create_multi_vector_chunks, ParentChunk, ChildChunk

logger = logging.getLogger(__name__)

class RAGHandler:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)
        
        if config.EMBEDDING_PROVIDER == "ollama":
            self.embedding_client = ollama.Client(host=config.OLLAMA_BASE_URL)
            # 【修复】使用正确的配置变量名
            logger.info(f"Embedding provider set to: Ollama (model: {config.EMBEDDING_MODEL_NAME})")
            self.embedding_dim = self._get_ollama_embedding_dimension()
        else: # huggingface
            # 【修复】使用正确的配置变量名
            self.embedding_client = SentenceTransformer(config.EMBEDDING_MODEL_NAME, cache_folder="/app/data/sentence_transformer_models")
            logger.info(f"Embedding provider set to: Hugging Face (model: {config.EMBEDDING_MODEL_NAME})")
            self.embedding_dim = self.embedding_client.get_sentence_embedding_dimension()

        if config.LLM_PROVIDER == 'openai':
            self.llm_client = openai.OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
        else:
            self.llm_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        
        # --- 初始化 Reranker ---
        self.reranker = Ranker(model_name=config.RERANK_MODEL_NAME, cache_dir="/app/data/reranker_models")
        logger.info(f"Reranker initialized with model: {config.RERANK_MODEL_NAME}")

        self.query_router = QueryRouter(
            qdrant_client=self.qdrant_client,
            embedding_model=self.embedding_client,
            llm_client=self.llm_client
        )
        
        for collection_name in config.AVAILABLE_COLLECTIONS:
            self._ensure_collection_exists(collection_name)

    # --- 问答主流程 (核心重构) ---
    def get_answer(self, query: str, chat_history: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        【V3.0 重构版】执行 混合检索 + Rerank 的高级RAG流程。
        """
        overall_start_time = time.time()
        logger.info("[TIMING] --- Starting ADVANCED get_answer flow ---")

        # 1. 初步检索：使用混合检索获取一个较大的候选集
        retrieval_start_time = time.time()
        initial_candidates = self.hybrid_retrieve(query, top_k=20)
        retrieval_end_time = time.time()
        logger.info(f"[TIMING] Initial retrieval finished. Duration: {retrieval_end_time - retrieval_start_time:.2f}s. Candidates found: {len(initial_candidates)}")

        if not initial_candidates:
            logger.warning("No candidates found after initial retrieval. Generating answer without context.")
            yield from self.generate_answer(query, [], chat_history)
            return

        # 2. 重排序：使用Reranker对候选集进行精排
        rerank_start_time = time.time()
        rerank_request = RerankRequest(query=query, passages=initial_candidates)
        reranked_results = self.reranker.rerank(rerank_request)
        final_context_payloads = reranked_results[:5]
        rerank_end_time = time.time()
        logger.info(f"[TIMING] Reranking finished. Duration: {rerank_end_time - rerank_start_time:.2f}s. Top 5 results selected.")

        final_context = [item['meta']['parent_content'] for item in final_context_payloads]

        # 3. 生成答案：使用精排后的高质量上下文生成答案
        gen_start_time = time.time()
        logger.info("[TIMING] Starting final answer generation with reranked context.")
        yield from self.generate_answer(query, final_context, chat_history)
        gen_end_time = time.time()
        logger.info(f"[TIMING] Final answer generation finished. Duration: {gen_end_time - gen_start_time:.2f}s.")
        
        overall_end_time = time.time()
        logger.info(f"[TIMING] --- Finished ADVANCED get_answer flow. Total duration: {overall_end_time - overall_start_time:.2f}s ---")

    def hybrid_retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        执行混合检索（稀疏+稠密）。
        返回的是 passages 列表，每个元素是一个包含 id, text, meta 的字典。
        """
        routed_collection = self.query_router.route_query(query)
        if not routed_collection:
            logger.info("Routing determined no specific collection.")
            return []

        # 稠密检索
        dense_vector = self._generate_embeddings(query)
        if not dense_vector:
            return []
        
        dense_hits = self.qdrant_client.search(
            collection_name=routed_collection,
            query_vector=dense_vector,
            limit=top_k,
            with_payload=True
        )

        # 稀疏检索 (BM25)
        # 简单的关键词提取
        query_keywords = query.split()
        sparse_vector = self._get_bm25_vector_for_query(query_keywords)

        sparse_hits = self.qdrant_client.search(
            collection_name=routed_collection,
            query_vector=models.NamedVector(
                name="bm25",
                vector=sparse_vector
            ),
            limit=top_k,
            with_payload=True
        )

        # 合并和去重
        combined_passages = {}
        for hit in dense_hits + sparse_hits:
            parent_id = hit.payload.get('parent_id')
            if parent_id and parent_id not in combined_passages:
                combined_passages[parent_id] = {
                    "id": parent_id,
                    "text": hit.payload.get("summary_content"), # Reranker 基于摘要进行排序
                    "meta": hit.payload
                }

        return list(combined_passages.values())

    def process_and_embed_document(self, parsed_elements: List[ParsedElement], filename: str, collection_name: str):
        self._ensure_collection_exists(collection_name)
        
        multi_vector_chunks = create_multi_vector_chunks(parsed_elements, filename)

        if not multi_vector_chunks:
            logger.warning(f"No multi-vector chunks were created for file '{filename}'.")
            return

        points = []
        child_contents_for_bm25 = []
        for parent_chunk, child_chunk in multi_vector_chunks:
            dense_vector = self._generate_embeddings(child_chunk.content)
            if dense_vector:
                payload = parent_chunk.to_qdrant_payload()
                payload["summary_content"] = child_chunk.content
                payload["collection"] = collection_name
                
                points.append(models.PointStruct(
                    id=child_chunk.chunk_id,
                    vector={"dense": dense_vector}, 
                    payload=payload
                ))
                child_contents_for_bm25.append(child_chunk.content)
        
        if points:
            self.qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)
            logger.info(f"Successfully upserted {len(points)} points (dense vectors and payload) for '{filename}'.")
            self._update_bm25_vectors(collection_name, points, child_contents_for_bm25)

    def _update_bm25_vectors(self, collection_name: str, points: List[models.PointStruct], contents: List[str]):
        try:
            tokenized_corpus = [doc.split() for doc in contents]
            bm25 = BM25Okapi(tokenized_corpus)
            
            update_operations = []
            for i, point in enumerate(points):
                doc_vector = bm25.get_vector(tokenized_corpus[i])
                
                sparse_vector = models.SparseVector(
                    indices=doc_vector.nonzero()[0].tolist(),
                    values=doc_vector.data.tolist()
                )
                
                update_operations.append(
                    models.PointVectors(
                        id=point.id,
                        vector=models.Vectors(
                            vectors_options=models.VectorsOptions(
                                named_vectors={"bm25": sparse_vector}
                            )
                        )
                    )
                )

            if update_operations:
                self.qdrant_client.update_vectors(
                    collection_name=collection_name,
                    points=update_operations
                )
                logger.info(f"Successfully updated {len(update_operations)} BM25 sparse vectors for '{collection_name}'.")

        except Exception as e:
            logger.error(f"Failed to update BM25 vectors: {e}", exc_info=True)
    
    def _get_bm25_vector_for_query(self, query_keywords: List[str]) -> models.SparseVector:
        # NOTE: This is a simplified implementation. A production system would
        # need a shared vocabulary between indexing and querying.
        mock_vocab = {word: i for i, word in enumerate(set(query_keywords))}
        indices = [mock_vocab[word] for word in query_keywords if word in mock_vocab]
        values = [1.5] * len(indices) # Give query keywords a higher weight
        return models.SparseVector(indices=indices, values=values)

    def create_collection(self, collection_name: str) -> bool:
        try:
            self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config={"dense": models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
            )
            logger.info(f"成功创建或重建混合检索集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"创建集合失败 {collection_name}: {e}")
            return False

    def generate_answer(self, query: str, context: List[str], chat_history: List[Dict[str, str]]):
        prompt = self._build_prompt(query, context, chat_history)
        if config.LLM_PROVIDER.lower() == 'openai':
            yield from self._generate_openai(prompt)
        else:
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

    def _generate_openai(self, messages: List[Dict[str, str]]):
        stream = self.llm_client.chat.completions.create(model=config.OPENAI_MODEL_NAME, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _generate_ollama(self, messages: List[Dict[str, str]]):
        stream = self.llm_client.chat(model=config.OLLAMA_MODEL_NAME, messages=messages, stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            if content:
                yield content

    def _get_ollama_embedding_dimension(self) -> int:
        try:
            # 【修复】使用正确的配置变量名
            logger.info(f"Pinging Ollama to get embedding dimension for model: {config.EMBEDDING_MODEL_NAME}")
            response = self.embedding_client.embeddings(model=config.EMBEDDING_MODEL_NAME, prompt="test")
            dim = len(response["embedding"])
            logger.info(f"Successfully determined embedding dimension: {dim}")
            return dim
        except Exception as e:
            logger.error(f"无法从 Ollama 获取嵌入维度: {e}", exc_info=True)
            logger.warning("将使用备用维度 1024。")
            return 1024

    def _ensure_collection_exists(self, collection_name: str):
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception:
            self.create_collection(collection_name)

    def _generate_embeddings(self, text: Union[str, object]) -> List[float]:
        text_str = str(text)
        try:
            if config.EMBEDDING_PROVIDER == "ollama":
                # 【修复】使用正确的配置变量名
                response = self.embedding_client.embeddings(model=config.EMBEDDING_MODEL_NAME, prompt=text_str)
                return response["embedding"]
            else:
                return self.embedding_client.encode(text_str, normalize_embeddings=False).tolist()
        except Exception as e:
            logger.error(f"为文本生成嵌入时出错: '{text_str[:50]}...' - {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"成功删除集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败 {collection_name}: {e}")
            return False

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

    def delete_document(self, filename: str, collection_name: str) -> bool:
        try:
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=models.Filter(must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=filename))
                ])),
                wait=True
            )
            logger.info(f"Successfully deleted vectors for document '{filename}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{filename}': {e}", exc_info=True)
            return False

rag_handler = RAGHandler()