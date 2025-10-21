# /* 文件名: backend/cve_handler.py, 版本号: 1.7 (增强日志版) */
"""
CPE 漏洞查询与数据加载处理器 V1.7

- **【核心新增】** 在 search_vulnerabilities_by_cpe 函数中增加了详细的 DEBUG 级别日志。
- 当查询成功时，会逐条打印出匹配到的漏洞CVE ID及其包含的'cpe_cores'列表，便于用户核查匹配的准确性。
"""
import logging
import json
import uuid
import os
import re
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models

import config
from ollama_client import get_ollama_client

logger = logging.getLogger(__name__)

class CVEHandler:
    def __init__(self):
        self.qdrant_client = QdrantClient(url=config.QDRANT_URL)
        self.embedding_client = get_ollama_client()
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
        """确保集合存在，并为 'cpes' 和 'cpe_cores' 字段创建关键词索引"""
        try:
            self.qdrant_client.get_collection(collection_name=self.collection_name)
        except Exception:
            logger.warning(f"集合 '{self.collection_name}' 不存在，将自动创建。")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)}
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="cpes",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="cpe_cores",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def _normalize_cpe_to_core(self, cpe_string: str) -> str:
        """将CPE 2.2或2.3字符串标准化为核心组件格式 (part:vendor:product:version)"""
        if not cpe_string:
            return ""
        match = re.match(r'cpe:/(?P<core22>[a-z]:[^:]+:[^:]+:[^:]+)', cpe_string)
        if match:
            return match.group('core22')
        match = re.match(r'cpe:2\.3:(?P<core23>[a-z]:[^:]+:[^:]+:[^:]+)', cpe_string)
        if match:
            return match.group('core23')
        return cpe_string

    def _process_single_nvd_file(self, json_file_path: str):
        """处理NVD JSON文件，并额外存储标准化的cpe_cores字段"""
        logger.info(f"开始处理 NVD 文件: {json_file_path}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        except Exception as e:
            logger.error(f"读取或解析JSON文件 {json_file_path} 失败: {e}"); return

        points_to_upsert = []
        vulnerabilities = data.get("vulnerabilities", [])

        for i, cve_item_wrapper in enumerate(vulnerabilities):
            item = cve_item_wrapper.get("cve", {})
            if not item: continue
            
            cve_id = item.get("id", "")
            description = next((d.get("value", "") for d in item.get("descriptions", []) if d.get("lang") == "en"), "")
            base_score, severity = 0.0, "UNKNOWN"
            if "cvssMetricV31" in item.get("metrics", {}):
                metric = item["metrics"]["cvssMetricV31"][0].get("cvssData", {})
                base_score = metric.get("baseScore", 0.0)
                severity = metric.get("baseSeverity", "UNKNOWN")

            cpe_strings_2_3 = [
                cpe_match.get("criteria")
                for config_node in item.get("configurations", [])
                for node in config_node.get("nodes", [])
                for cpe_match in node.get("cpeMatch", [])
                if cpe_match.get("vulnerable") and cpe_match.get("criteria")
            ]

            cpe_cores = list(set(filter(None, [self._normalize_cpe_to_core(c) for c in cpe_strings_2_3])))

            content_for_embedding = f"CVE ID: {cve_id}. Description: {description}. CPEs: {', '.join(cpe_strings_2_3)}"
            vector = self._generate_embeddings(content_for_embedding)

            if vector:
                payload = {
                    "cve_id": cve_id, "description": description, "base_score": base_score, 
                    "severity": severity, "cpes": cpe_strings_2_3, "cpe_cores": cpe_cores
                }
                points_to_upsert.append(models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector={"dense": vector}, 
                    payload=payload
                ))
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1} / {len(vulnerabilities)} 条 CVE 数据...")
                if points_to_upsert:
                    self.qdrant_client.upsert(collection_name=self.collection_name, points=points_to_upsert, wait=True)
                    points_to_upsert = []
        if points_to_upsert:
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points_to_upsert, wait=True)
        logger.info(f"文件 {os.path.basename(json_file_path)} 加载完成！")

    def load_nvd_data_from_directory(self, directory_path: str):
        logger.info(f"开始从目录 {directory_path} 加载所有NVD JSON文件...")
        if not os.path.isdir(directory_path): logger.error(f"路径 '{directory_path}' 不是有效目录。"); return
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        if not json_files: logger.warning(f"目录 '{directory_path}' 中没有 .json 文件。"); return
        for i, filename in enumerate(json_files):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"--- 开始加载文件 {i+1}/{len(json_files)}: {filename} ---")
            self._process_single_nvd_file(file_path)
        logger.info(f"所有 {len(json_files)} 个 NVD JSON 文件加载完毕。")

    def search_vulnerabilities_by_cpe(self, cpe_string: str) -> List[Dict[str, Any]]:
        """先将输入CPE标准化，然后对 'cpe_cores' 字段进行精确匹配"""
        if not cpe_string:
            return []
        
        cpe_core_to_search = self._normalize_cpe_to_core(cpe_string)
        if not cpe_core_to_search:
            logger.warning(f"无法从输入 '{cpe_string}' 中解析出有效的CPE核心组件。")
            return []
        
        logger.info(f"正在为标准化CPE核心 '{cpe_core_to_search}' 进行精确过滤检索...")
        try:
            scroll_response, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="cpe_cores", match=models.MatchText(text=cpe_core_to_search))
                ]),
                limit=100, with_payload=True
            )
            results = [hit.payload for hit in scroll_response]
            logger.info(f"为标准化CPE核心 '{cpe_core_to_search}' 找到了 {len(results)} 条精确匹配的漏洞记录。")
            
            # --- 【核心新增】打印详细日志以供核查 ---
            if results:
                logger.info("--- 开始核查匹配结果 ---")
                for res in results:
                    cve_id = res.get('cve_id', 'N/A')
                    matching_cores = res.get('cpe_cores', [])
                    # 打印每个匹配到的漏洞及其包含的标准化CPE列表
                    logger.info(f"  -> 匹配到漏洞: {cve_id}")
                    logger.info(f"     其CPE核心组件列表: {matching_cores}")
                logger.info("--- 核查结束 ---")
            # --- 日志新增结束 ---

            return results
        except Exception as e:
            logger.error(f"在 '{self.collection_name}' 集合中进行关键词过滤时出错: {e}", exc_info=True)
            return []

cve_handler = CVEHandler()