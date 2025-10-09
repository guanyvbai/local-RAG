# /* 文件名: backend/router.py, 版本号: 2.0 */
# backend/router.py (已修正 Qdrant 查询格式)
import logging
from typing import Optional
import openai
import ollama
from qdrant_client import QdrantClient, models  # <-- 【核心修正】导入 models
from sentence_transformers import SentenceTransformer
import config
from typing import List

logger = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self, qdrant_client: QdrantClient, embedding_model, llm_client):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.collections = config.AVAILABLE_COLLECTIONS

    def _get_query_vector(self, query: str) -> Optional[List[float]]:
        """根据客户端类型，使用正确的方法生成向量"""
        try:
            if hasattr(self.embedding_model, 'encode'):
                return self.embedding_model.encode(query).tolist()
            else:
                response = self.embedding_model.embeddings(model=config.EMBEDDING_MODEL_NAME, prompt=query)
                return response.get("embedding")
        except Exception as e:
            logger.error(f"生成查询向量时出错: {e}")
            return None

    def _route_by_similarity(self, query: str) -> Optional[str]:
        query_vector = self._get_query_vector(query)
        if not query_vector:
            return None

        scores = {}
        for collection in self.collections:
            try:
                # --- 【核心修正】将向量包装在 NamedVector 中 ---
                search_results = self.qdrant_client.search(
                    collection_name=collection,
                    query_vector=models.NamedVector(name="dense", vector=query_vector),
                    limit=3
                )
                if search_results:
                    avg_score = sum(hit.score for hit in search_results) / len(search_results)
                    scores[collection] = avg_score
                    logger.info(f"Similarity search in '{collection}' avg score: {avg_score:.4f}")
            except Exception as e:
                logger.error(f"Error during similarity search in '{collection}': {e}")
                continue
        
        if not scores: return None
        
        best_collection = max(scores, key=scores.get)
        best_score = scores[best_collection]
        
        if best_score >= config.ROUTING_CONFIDENCE_THRESHOLD:
            logger.info(f"Layer 1 Routing SUCCESS: Chose '{best_collection}' with score {best_score:.4f}")
            return best_collection
        else:
            logger.info(f"Layer 1 Routing FAILED: Best score {best_score:.4f} is below threshold.")
            return None

    def _route_by_llm(self, query: str) -> Optional[str]:
        prompt = f"""You are an expert query router. Your task is to classify the user's query into one of the predefined databases.
Follow these rules strictly:
1. Read the user's query carefully.
2. Analyze which database best matches the query's content.
3. Your response MUST be only a single database name from the list below.
4. DO NOT return any extra explanations, text, punctuation, or newlines.

[Database List]
- general: 通用信息或无法分类的内容。
- products: 主机的资产信息。
- support: 关于如何使用产品、故障排查、用户指南、部署、运维的查询。
- vulnerability: 关于产品漏洞信息的查询。

[User Query]
"{query}"

[Your Decision (a single word)]
"""
        try:
            if config.LLM_PROVIDER.lower() == 'openai':
                response = self.llm_client.chat.completions.create(model=config.OPENAI_MODEL_NAME, messages=[{"role": "system", "content": prompt}], temperature=0, max_tokens=10)
                decision = response.choices[0].message.content.strip()
            else:
                response = self.llm_client.chat(model=config.OLLAMA_MODEL_NAME, messages=[{"role": "system", "content": prompt}], options={"temperature": 0})
                decision = response['message']['content'].strip()
            
            if decision in self.collections:
                logger.info(f"Layer 2 Routing SUCCESS: LLM chose '{decision}'")
                return decision
            else:
                logger.warning(f"Layer 2 Routing FAILED: LLM returned invalid decision '{decision}'")
                return None
        except Exception as e:
            logger.error(f"Error during LLM routing: {e}")
            return None

    def route_query(self, query: str) -> Optional[str]:
        decision = self._route_by_similarity(query)
        if decision:
            logger.info(f"=== FINAL ROUTING DECISION: '{decision}' (from Layer 1 Similarity) ===")
            return decision
        
        logger.info("Falling back to Layer 2 LLM Routing.")
        decision = self._route_by_llm(query)

        if decision:
            logger.info(f"=== FINAL ROUTING DECISION: '{decision}' (from Layer 2 LLM) ===")
        else:
            logger.info("=== FINAL ROUTING DECISION: None (Fallback to General Knowledge) ===")
        return decision