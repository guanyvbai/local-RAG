# /* 文件名: backend/cve_handler.py, 版本号: 1.3 */
"""
CPE 漏洞查询与数据加载处理器 V1.3

- **【核心修正】** 统一了与Qdrant交互的数据结构，现在使用与 rag_handler 一致的命名向量 "dense"。
- `_ensure_collection_exists`: 创建集合时，明确指定 `vectors_config` 为 `{"dense": ...}`。
- `_process_single_nvd_file`: 插入数据点时，将向量包装在 `{"dense": vector}` 结构中。
- `search_vulnerabilities_by_cpe`: 查询时，同样使用命名向量。
"""
import logging
import json
import uuid
import os
from typing import List, Dict, Any
import ollama
from qdrant_client import QdrantClient, models

import config

logger = logging.getLogger(__name__)

class CVEHandler:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)
        self.embedding_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        self.embedding_model_name = config.EMBEDDING_MODEL_NAME
        
        try:
            response = self.embedding_client.embeddings(model=self.embedding_model_name, prompt="test")
            self.embedding_dim = len(response["embedding"])
        except Exception as e:
            logger.error(f"无法从Ollama获取嵌入维度: {e}", exc_info=True)
            self.embedding_dim = 1024 # Fallback

        self.collection_name = "vulnerability"
        self._ensure_collection_exists()

    def _generate_embeddings(self, text: str) -> List[float]:
        try:
            response = self.embedding_client.embeddings(model=self.embedding_model_name, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            logger.error(f"为文本 '{text[:50]}...' 生成嵌入时出错: {e}")
            return []

    def _ensure_collection_exists(self):
        """【核心修正】创建集合时使用与rag_handler一致的命名向量结构"""
        try:
            self.qdrant_client.get_collection(collection_name=self.collection_name)
        except Exception:
            logger.warning(f"集合 '{self.collection_name}' 不存在，将自动创建。")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)
                }
            )

    def _process_single_nvd_file(self, json_file_path: str):
        """处理单个 NVD CVE 2.0 JSON 文件并存入Qdrant"""
        logger.info(f"开始处理 NVD 2.0 格式文件: {json_file_path}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取或解析JSON文件 {json_file_path} 失败: {e}")
            return

        points_to_upsert = []
        vulnerabilities = data.get("vulnerabilities", [])

        for i, cve_item_wrapper in enumerate(vulnerabilities):
            item = cve_item_wrapper.get("cve", {})
            if not item:
                continue

            cve_id = item.get("id", "")
            description = next((d.get("value", "") for d in item.get("descriptions", []) if d.get("lang") == "en"), "")
            
            base_score, severity = 0.0, "UNKNOWN"
            if "cvssMetricV31" in item.get("metrics", {}):
                metric = item["metrics"]["cvssMetricV31"][0].get("cvssData", {})
                base_score = metric.get("baseScore", 0.0)
                severity = metric.get("baseSeverity", "UNKNOWN")
            elif "cvssMetricV2" in item.get("metrics", {}):
                metric = item["metrics"]["cvssMetricV2"][0]
                base_score = metric.get("exploitabilityScore", 0.0)
                severity = metric.get("severity", "UNKNOWN")

            cpe_strings = [
                cpe_match.get("criteria")
                for config_node in item.get("configurations", [])
                for node in config_node.get("nodes", [])
                for cpe_match in node.get("cpeMatch", [])
                if cpe_match.get("vulnerable") and cpe_match.get("criteria")
            ]

            content_for_embedding = f"CVE ID: {cve_id}. Description: {description}. CPEs: {', '.join(cpe_strings)}"
            vector = self._generate_embeddings(content_for_embedding)

            if vector:
                payload = {"cve_id": cve_id, "description": description, "base_score": base_score, "severity": severity, "cpes": cpe_strings}
                # --- 【核心修正】将向量包装在名为 "dense" 的字典中 ---
                points_to_upsert.append(models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector={"dense": vector}, 
                    payload=payload
                ))
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1} / {len(vulnerabilities)} 条 CVE 数据来自 {os.path.basename(json_file_path)}...")
                if points_to_upsert:
                    self.qdrant_client.upsert(collection_name=self.collection_name, points=points_to_upsert, wait=True)
                    points_to_upsert = []

        if points_to_upsert:
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points_to_upsert, wait=True)
        
        logger.info(f"文件 {os.path.basename(json_file_path)} 加载完成！")

    def load_nvd_data_from_directory(self, directory_path: str):
        logger.info(f"开始从目录 {directory_path} 加载所有NVD JSON文件...")
        if not os.path.isdir(directory_path):
            logger.error(f"提供的路径 '{directory_path}' 不是一个有效的目录。")
            return

        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        if not json_files:
            logger.warning(f"在目录 '{directory_path}' 中没有找到 .json 文件。")
            return

        for i, filename in enumerate(json_files):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"--- 开始加载文件 {i+1}/{len(json_files)}: {filename} ---")
            self._process_single_nvd_file(file_path)
        
        logger.info(f"所有 {len(json_files)} 个 NVD JSON 文件加载完毕。")

    def search_vulnerabilities_by_cpe(self, cpe_string: str) -> List[Dict[str, Any]]:
        if not cpe_string: return []
        
        cpe_vector = self._generate_embeddings(cpe_string)
        if not cpe_vector: return []
            
        try:
            # --- 【核心修正】查询时同样使用命名向量 ---
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedVector(name="dense", vector=cpe_vector),
                limit=10,
                with_payload=True,
                score_threshold=0.6
            )
            results = [hit.payload for hit in search_results]
            logger.info(f"为 CPE '{cpe_string}' 找到了 {len(results)} 条相关漏洞记录。")
            return results
        except Exception as e:
            logger.error(f"在 '{self.collection_name}' 集合中检索时出错: {e}", exc_info=True)
            return []

cve_handler = CVEHandler()