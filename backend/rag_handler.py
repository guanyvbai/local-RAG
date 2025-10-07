"""
RAG (Retrieval-Augmented Generation) 核心处理器。
【V2.5 诊断版】:
- 恢复完整的“迭代式自我修正”逻辑。
- 在所有关键函数（特别是LLM调用）周围增加了详细的计时日志。
"""
import ollama
import openai
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Generator
import logging
import time  # <--- 添加 time 模块导入

import config
from router import QueryRouter
from document_parser import ParsedElement
from chunker import chunk_document, Chunk

logger = logging.getLogger(__name__)

class RAGHandler:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)
        
        if config.EMBEDDING_PROVIDER == "ollama":
            self.embedding_client = ollama.Client(host=config.OLLAMA_BASE_URL)
            logger.info("Embedding provider set to: Ollama")
            self.embedding_dim = self._get_ollama_embedding_dimension()
        else:
            self.embedding_client = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            logger.info(f"Embedding provider set to: Hugging Face (model: {config.EMBEDDING_MODEL_NAME})")
            self.embedding_dim = self.embedding_client.get_sentence_embedding_dimension()

        if config.LLM_PROVIDER == 'openai':
            self.llm_client = openai.OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
        else:
            self.llm_client = ollama.Client(host=config.OLLAMA_BASE_URL)
            
        self.query_router = QueryRouter(
            qdrant_client=self.qdrant_client,
            embedding_model=self.embedding_client,
            llm_client=self.llm_client
        )
        
        for collection_name in config.AVAILABLE_COLLECTIONS:
            self._ensure_collection_exists(collection_name)

    # --- LLM 调用辅助函数 ---
    def _call_llm(self, messages: List[Dict[str, str]], function_name: str) -> str:
        """
        一个通用的、非流式的 LLM 调用函数，增加了计时功能。
        """
        start_time = time.time()
        logger.info(f"[TIMING] Entering LLM call for: {function_name}")
        try:
            if config.LLM_PROVIDER.lower() == 'openai':
                response = self.llm_client.chat.completions.create(model=config.OPENAI_MODEL_NAME, messages=messages, stream=False)
                result = response.choices[0].message.content.strip()
            else:
                response = self.llm_client.chat(model=config.OLLAMA_MODEL_NAME, messages=messages, stream=False)
                result = response['message']['content'].strip()
            
            end_time = time.time()
            logger.info(f"[TIMING] Exiting LLM call for: {function_name}. Duration: {end_time - start_time:.2f} seconds.")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"LLM call for {function_name} failed after {end_time - start_time:.2f} seconds: {e}", exc_info=True)
            return ""

    # --- 迭代式自我修正流程的辅助函数 ---
    def _rewrite_query_as_statement(self, original_query: str) -> str:
        """步骤 1: 将用户的原始问题改写为适合检索的陈述句。"""
        prompt = f"""请将以下用户问题改写为一个清晰、简洁、包含核心关键词的陈述句，以便于进行向量检索。请直接返回陈述句，不要包含任何多余的解释。/no_think
用户问题: "{original_query}"
改写后的陈述句:"""
        messages = [{"role": "user", "content": prompt}]
        rewritten_query = self._call_llm(messages, "_rewrite_query_as_statement")
        logger.info(f"Original Query: '{original_query}' | Rewritten: '{rewritten_query}'")
        return rewritten_query if rewritten_query else original_query

    def _evaluate_context_sufficiency(self, query: str, context: List[str]) -> bool:
        """步骤 3: 评估检索到的上下文是否足以回答问题。"""
        if not context:
            return False
        context_str = "\n---\n".join(context)
        prompt = f"""我将使用以下信息来回答问题。请判断这些信息是否足够、相关，能够完整地回答问题。
请只回答“是”或“否”。/no_think

问题: "{query}"

[信息]
{context_str}

[判断 (是/否)]
"""
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages, "_evaluate_context_sufficiency")
        logger.info(f"Context evaluation for query '{query}'. LLM response: '{response}'")
        return "是" in response

    def _generate_new_statement_for_retry(self, original_query: str, failed_statement: str) -> str:
        """步骤 4 (修正): 根据失败的经验，生成一个新的、更好的陈述句。"""
        prompt = f"""我正在尝试回答用户的原始问题。我第一次尝试使用一个陈述句进行检索，但得到的信息质量很差。请根据这次失败的经验，生成一个全新的、可能从不同角度切入的陈述句用于重新检索。请直接返回新的陈述句。/no_think

用户的原始问题: "{original_query}"
失败的检索陈述句: "{failed_statement}"

新的、更好的陈述句:"""
        messages = [{"role": "user", "content": prompt}]
        new_statement = self._call_llm(messages, "_generate_new_statement_for_retry")
        logger.info(f"Correction step. Failed statement: '{failed_statement}' | New statement: '{new_statement}'")
        return new_statement if new_statement else failed_statement

    # --- 问答主流程 (核心改造) ---
    def get_answer(self, query: str, chat_history: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        【V2.5 诊断版】执行完整的“迭代式自我修正 RAG”流程，并增加计时。
        """
        overall_start_time = time.time()
        logger.info("[TIMING] --- Starting get_answer flow ---")
        
        context = []
        
        current_statement = self._rewrite_query_as_statement(query)

        for i in range(config.MAX_QUERY_RETRY_LOOPS + 1):
            iter_start_time = time.time()
            logger.info(f"[TIMING] Iteration {i+1}/{config.MAX_QUERY_RETRY_LOOPS + 1} | Retrieving with statement: '{current_statement}'")
            
            retrieved_context = self.retrieve_context(current_statement)
            
            if self._evaluate_context_sufficiency(current_statement, retrieved_context):
                logger.info("Context is sufficient. Proceeding to generate answer.")
                context = retrieved_context
                iter_end_time = time.time()
                logger.info(f"[TIMING] Iteration {i+1} successful. Duration: {iter_end_time - iter_start_time:.2f} seconds.")
                break
            
            if i < config.MAX_QUERY_RETRY_LOOPS:
                logger.warning("Context is NOT sufficient. Generating a new statement for retry.")
                current_statement = self._generate_new_statement_for_retry(query, current_statement)
                context = retrieved_context # Fallback context
            else:
                logger.warning("Max retry loops reached. Using the last retrieved context as fallback.")
                context = retrieved_context
            
            iter_end_time = time.time()
            logger.info(f"[TIMING] Iteration {i+1} finished. Duration: {iter_end_time - iter_start_time:.2f} seconds.")

        gen_start_time = time.time()
        logger.info("[TIMING] Starting final answer generation.")
        yield from self.generate_answer(query, context, chat_history)
        gen_end_time = time.time()
        logger.info(f"[TIMING] Final answer generation finished. Duration: {gen_end_time - gen_start_time:.2f} seconds.")
        
        overall_end_time = time.time()
        logger.info(f"[TIMING] --- Finished get_answer flow. Total duration: {overall_end_time - overall_start_time:.2f} seconds ---")

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """使用双层路由从合适的集合中检索上下文。"""
        start_time = time.time()
        logger.info(f"[TIMING] Entering retrieve_context for query: '{query}'")
        
        routed_collection = self.query_router.route_query(query)
        
        if not routed_collection:
            logger.info("Routing determined no specific collection. Skipping knowledge base retrieval.")
            end_time = time.time()
            logger.info(f"[TIMING] Exiting retrieve_context. Duration: {end_time - start_time:.2f} seconds.")
            return []

        logger.info(f"Routing to collection '{routed_collection}'.")
        query_vector = self._generate_embeddings(query)
        if not query_vector: 
            end_time = time.time()
            logger.info(f"[TIMING] Exiting retrieve_context (no vector). Duration: {end_time - start_time:.2f} seconds.")
            return []
        
        search_results = self.qdrant_client.search(
            collection_name=routed_collection,
            query_vector=query_vector,
            limit=top_k
        )
        
        end_time = time.time()
        logger.info(f"[TIMING] Exiting retrieve_context. Duration: {end_time - start_time:.2f} seconds. Found {len(search_results)} results.")
        return [hit.payload['content'] for hit in search_results]
            
    def generate_answer(self, query: str, context: List[str], chat_history: List[Dict[str, str]]):
        """生成最终回答"""
        prompt = self._build_prompt(query, context, chat_history)
        if config.LLM_PROVIDER.lower() == 'openai':
            yield from self._generate_openai(prompt)
        else:
            yield from self._generate_ollama(prompt)

    def _build_prompt(self, query: str, context: List[str], chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """构建发送给大语言模型的最终 Prompt"""
        context_str = "\n---\n".join(context)
        history_str = "\n".join([f"用户: {item['user']}\n助手: {item['assistant']}" for item in chat_history])
        
        system_message = f"""你是一个智能AI助手，旨在提供友好和有帮助的对话。你的回答应该遵循以下规则：
1. **优先使用知识库**：首先检查下面提供的“[知识库上下文]”。如果它包含了与用户问题相关的信息，请基于这些信息来回答，并可以适当结合你自己的知识进行补充和润色。
2. **切换到通用知识**：如果“[知识库上下文]”是空的、与问题无关，或者不足以回答问题，那么请忽略它，并像一个通用的AI助手（如ChatGPT或Gemini）一样，利用你自己的内置知识库来回答用户的问题。
3. **保持对话性**：无论使用哪种知识来源，都请保持友好、自然的对话风格。/no_think
---
[知识库上下文]
{context_str}
---
"""
        messages = [{"role": "system", "content": system_message}]
        if history_str:
            messages.append({"role": "system", "content": f"[历史对话记录]\n{history_str}"})
        
        messages.append({"role": "user", "content": query})
        return messages

    def _generate_openai(self, messages: List[Dict[str, str]]):
        """调用 OpenAI API 生成回答 (流式)"""
        stream = self.llm_client.chat.completions.create(model=config.OPENAI_MODEL_NAME, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _generate_ollama(self, messages: List[Dict[str, str]]):
        """调用 Ollama API 生成回答 (流式)"""
        stream = self.llm_client.chat(model=config.OLLAMA_MODEL_NAME, messages=messages, stream=True)
        for chunk in stream:
            content = chunk['message']['content']
            if content:
                yield content

    def _get_ollama_embedding_dimension(self) -> int:
        """为 Ollama 动态获取嵌入维度"""
        try:
            response = self.embedding_client.embeddings(model=config.EMBEDDING_MODEL_NAME, prompt="test")
            return len(response["embedding"])
        except Exception as e:
            logger.error(f"无法从 Ollama 获取嵌入维度: {e}", exc_info=True)
            logger.warning("将使用备用维度 4096。")
            return 4096

    def _ensure_collection_exists(self, collection_name: str):
        """确保 Qdrant 集合存在"""
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception:
            self.create_collection(collection_name)

    def _generate_embeddings(self, text: Union[str, object]) -> List[float]:
        """统一的嵌入生成函数"""
        text_str = str(text)
        try:
            if config.EMBEDDING_PROVIDER == "ollama":
                response = self.embedding_client.embeddings(model=config.EMBEDDING_MODEL_NAME, prompt=text_str)
                return response["embedding"]
            else:
                return self.embedding_client.encode(text_str, convert_to_tensor=False).tolist()
        except Exception as e:
            logger.error(f"为文本生成嵌入时出错: '{text_str[:50]}...' - {e}")
            return []

    def create_collection(self, collection_name: str) -> bool:
        """在 Qdrant 中创建一个新的集合"""
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)
            )
            logger.info(f"成功创建集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"创建集合失败 {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """从 Qdrant 中删除一个集合"""
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"成功删除集合: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败 {collection_name}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """从 Qdrant 中列出所有集合"""
        try:
            collections_response = self.qdrant_client.get_collections()
            return [collection.name for collection in collections_response.collections]
        except Exception as e:
            logger.error(f"无法从 Qdrant 获取集合列表: {e}")
            return []
    
    def list_documents(self) -> List[Dict[str, str]]:
        """列出所有已索引的文档及其所属集合"""
        all_docs = {}
        try:
            collections = self.list_collections()
            for collection_name in collections:
                scroll_result, _ = self.qdrant_client.scroll(
                    collection_name=collection_name, limit=10000,
                    with_payload=["source"], with_vectors=False
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
        """删除指定文档的所有相关向量"""
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
            
    def process_and_embed_document(self, parsed_elements: List[ParsedElement], filename: str, collection_name: str):
        """处理、切分、嵌入并存储文档的核心流程。"""
        self._ensure_collection_exists(collection_name)
        
        chunks: List[Chunk] = chunk_document(parsed_elements, filename)

        if not chunks:
            logger.warning(f"No chunks were created for file '{filename}'.")
            return

        points = []
        for chunk in chunks:
            vector = self._generate_embeddings(chunk.content)
            if vector:
                payload = chunk.to_dict()
                payload["collection"] = collection_name
                
                points.append(models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector, 
                    payload=payload
                ))
        
        if points:
            self.qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)
            logger.info(f"Successfully upserted {len(points)} semantic chunks for '{filename}' into '{collection_name}'.")

rag_handler = RAGHandler()