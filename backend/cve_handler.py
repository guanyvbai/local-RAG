# /* 文件名: backend/cve_handler.py, 版本号: 1.7 (增强日志版) */
"""
CPE 漏洞查询与数据加载处理器 V1.7

- **【核心新增】** 在 search_vulnerabilities_by_cpe 函数中增加了详细的 DEBUG 级别日志。
- 当查询成功时，会逐条打印出匹配到的漏洞CVE ID及其包含的'cpe_cores'列表，便于用户核查匹配的准确性。
"""
import logging
import json
import os
import re
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from packaging import version as packaging_version
from packaging.version import InvalidVersion

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
        """确保集合存在，并为检索相关的字段创建关键词索引"""
        try:
            self.qdrant_client.get_collection(collection_name=self.collection_name)
        except Exception:
            logger.warning(f"集合 '{self.collection_name}' 不存在，将自动创建。")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE)
                }
            )
        for field_name in ["cve_id", "cpes", "cpe_cores", "vendor", "product", "part"]:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"为字段 '{field_name}' 创建索引失败: {e}")

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

    def _parse_cpe_components(self, cpe_uri: str) -> Dict[str, Optional[str]]:
        """解析CPE字符串并返回组成部分（part/vendor/product/version）"""
        if not cpe_uri:
            return {"part": None, "vendor": None, "product": None, "version": None}

        if cpe_uri.startswith("cpe:2.3:"):
            parts = cpe_uri.split(":")
            if len(parts) >= 6:
                return {
                    "part": parts[2] or None,
                    "vendor": parts[3] or None,
                    "product": parts[4] or None,
                    "version": parts[5] or None,
                }
        elif cpe_uri.startswith("cpe:/"):
            # 处理早期的CPE 2.2格式: cpe:/part:vendor:product:version
            remainder = cpe_uri[5:]
            parts = remainder.split(":")
            if len(parts) >= 4:
                return {
                    "part": parts[0] or None,
                    "vendor": parts[1] or None,
                    "product": parts[2] or None,
                    "version": parts[3] or None,
                }
        return {"part": None, "vendor": None, "product": None, "version": None}

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

            cpe_entries = []
            vendor_set, product_set, part_set = set(), set(), set()
            cpe_strings_2_3 = []

            for config_node in item.get("configurations", []):
                for node in config_node.get("nodes", []):
                    for cpe_match in node.get("cpeMatch", []):
                        if not (cpe_match.get("vulnerable") and cpe_match.get("criteria")):
                            continue

                        criteria = cpe_match.get("criteria")
                        cpe_strings_2_3.append(criteria)

                        components = self._parse_cpe_components(criteria)
                        part = components.get("part")
                        vendor = components.get("vendor")
                        product = components.get("product")

                        if part:
                            part_set.add(part.lower())
                        if vendor:
                            vendor_set.add(vendor.lower())
                        if product:
                            product_set.add(product.lower())

                        entry = {
                            "criteria": criteria,
                            "match_criteria_id": cpe_match.get("matchCriteriaId"),
                            "vulnerable": cpe_match.get("vulnerable", False),
                            "version": components.get("version"),
                            "version_start_including": cpe_match.get("versionStartIncluding"),
                            "version_start_excluding": cpe_match.get("versionStartExcluding"),
                            "version_end_including": cpe_match.get("versionEndIncluding"),
                            "version_end_excluding": cpe_match.get("versionEndExcluding"),
                            "part": part,
                            "vendor": vendor,
                            "product": product,
                            "vendor_normalized": vendor.lower() if vendor else None,
                            "product_normalized": product.lower() if product else None,
                            "part_normalized": part.lower() if part else None,
                        }
                        cpe_entries.append(entry)

            cpe_cores = list(set(filter(None, [self._normalize_cpe_to_core(c) for c in cpe_strings_2_3])))

            content_for_embedding = f"CVE ID: {cve_id}. Description: {description}. CPEs: {', '.join(cpe_strings_2_3)}"
            vector = self._generate_embeddings(content_for_embedding)

            if not cve_id:
                logger.warning("检测到缺失 CVE ID 的条目，已跳过。")
                continue

            if vector:
                payload = {
                    "cve_id": cve_id,
                    "description": description,
                    "base_score": base_score,
                    "severity": severity,
                    "cpes": cpe_strings_2_3,
                    "cpe_cores": cpe_cores,
                    "cpe_entries": cpe_entries,
                    "vendor": sorted(vendor_set),
                    "product": sorted(product_set),
                    "part": sorted(part_set),
                }
                points_to_upsert.append(models.PointStruct(
                    id=cve_id,
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

    def _is_version_match(self, target_version: Optional[str], cpe_entry: Dict[str, Any]) -> bool:
        """根据输入版本与CPE条目中的范围信息，判断是否匹配"""
        def parse_or_none(value: Optional[str]) -> Optional[packaging_version.Version]:
            if not value or value in {"*", "-"}:
                return None
            try:
                return packaging_version.parse(value)
            except InvalidVersion:
                logger.debug(f"无法解析版本号 '{value}'，已跳过范围判断。")
                return None

        target_version = None if not target_version or target_version in {"*", "-"} else target_version
        if target_version is None:
            # 无具体版本，视为命中
            return True

        try:
            target_parsed = packaging_version.parse(target_version)
        except InvalidVersion:
            logger.debug(f"无法解析目标版本 '{target_version}'，忽略版本过滤。")
            return True

        exact_version = cpe_entry.get("version")
        if exact_version and exact_version not in {"*", "-"}:
            candidate = parse_or_none(exact_version)
            if candidate is not None and candidate != target_parsed:
                return False

        start_incl = parse_or_none(cpe_entry.get("version_start_including"))
        if start_incl and target_parsed < start_incl:
            return False

        start_excl = parse_or_none(cpe_entry.get("version_start_excluding"))
        if start_excl and target_parsed <= start_excl:
            return False

        end_incl = parse_or_none(cpe_entry.get("version_end_including"))
        if end_incl and target_parsed > end_incl:
            return False

        end_excl = parse_or_none(cpe_entry.get("version_end_excluding"))
        if end_excl and target_parsed >= end_excl:
            return False

        return True

    def search_vulnerabilities_by_cpe(self, cpe_string: str) -> List[Dict[str, Any]]:
        """先基于 vendor/product 在 Qdrant 中过滤，再在 Python 中进行版本判定"""
        if not cpe_string:
            return []

        components = self._parse_cpe_components(cpe_string)
        vendor = components.get("vendor")
        product = components.get("product")
        part = components.get("part")
        version_value = components.get("version")

        must_conditions = []
        if vendor:
            must_conditions.append(models.FieldCondition(
                key="vendor",
                match=models.MatchValue(value=vendor.lower())
            ))
        if product:
            must_conditions.append(models.FieldCondition(
                key="product",
                match=models.MatchValue(value=product.lower())
            ))
        if part:
            must_conditions.append(models.FieldCondition(
                key="part",
                match=models.MatchValue(value=part.lower())
            ))

        if not must_conditions:
            logger.warning(f"无法从输入 '{cpe_string}' 中解析出有效的 vendor/product 信息。")
            return []

        logger.info(
            "正在根据 vendor=%s, product=%s, part=%s 进行初步过滤检索...",
            vendor, product, part
        )

        try:
            scroll_response, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must_conditions),
                limit=100,
                with_payload=True
            )

            results = []
            for hit in scroll_response:
                payload = hit.payload or {}
                cpe_entries = payload.get("cpe_entries", [])
                matched_entries = []

                for entry in cpe_entries:
                    if vendor and entry.get("vendor_normalized") != vendor.lower():
                        continue
                    if product and entry.get("product_normalized") != product.lower():
                        continue
                    if part and entry.get("part_normalized") != part.lower():
                        continue
                    if self._is_version_match(version_value, entry):
                        matched_entries.append(entry)

                if matched_entries:
                    payload_with_matches = dict(payload)
                    payload_with_matches["matched_cpe_entries"] = matched_entries
                    results.append(payload_with_matches)

            logger.info(
                "针对 vendor=%s, product=%s 共过滤出 %d 条匹配记录。",
                vendor, product, len(results)
            )
            return results
        except Exception as e:
            logger.error(f"在 '{self.collection_name}' 集合中进行过滤检索时出错: {e}", exc_info=True)
            return []

    def search_cpes_by_cve(self, cve_id: str) -> List[Dict[str, Any]]:
        """基于 CVE ID 在 Qdrant 中进行精确匹配，并返回对应的 payload"""
        if not cve_id:
            logger.warning("收到空的 CVE ID 查询请求，已直接返回空结果。")
            return []

        normalized_cve_id = cve_id.strip().upper()
        if not normalized_cve_id:
            logger.warning("收到仅包含空白字符的 CVE ID 查询请求，已直接返回空结果。")
            return []

        logger.info(f"正在按照 CVE ID '{normalized_cve_id}' 查询关联的 CPE 信息...")
        try:
            scroll_response, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="cve_id", match=models.MatchValue(value=normalized_cve_id))
                ]),
                limit=10,
                with_payload=True
            )
            results = [hit.payload for hit in scroll_response]
            logger.info(f"针对 CVE ID '{normalized_cve_id}' 找到 {len(results)} 条记录。")
            return results
        except Exception as e:
            logger.error(f"通过 CVE ID '{normalized_cve_id}' 检索 CPE 信息时出错: {e}", exc_info=True)
            return []

cve_handler = CVEHandler()
