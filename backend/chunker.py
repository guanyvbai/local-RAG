# /* 文件名: backend/chunker.py, 版本号: 4.3 */
"""
【V4.3 终极健壮版++】:
- 在调用Ollama API时，继续强制指定 format="json"。
- 对解析后的JSON对象，采用更安全的 .get() 方法来获取值，避免因缺少键而直接引发异常。
- 增加了对获取到的值是否为空的检查，确保只有在 summary 和 hypothetical_question 都有效时才采纳。
- 统一了所有失败情况（包括JSON解析失败、缺少键、键值为空）的回退逻辑，极大地增强了代码的容错能力。
"""
import hashlib
import logging
import uuid
import json
import re
from typing import List, Dict, Any, TYPE_CHECKING

import config
from ollama_client import get_ollama_client

# 避免循环导入
if TYPE_CHECKING:
    from document_parser import ParsedElement

logger = logging.getLogger(__name__)

# --- 配置常量 ---
# 定义父块的理想大小
MAX_CHUNK_CHARS = 1500

class ChildChunk:
    """子文档块，包含用于检索的文本（摘要或假设性问题）。"""
    def __init__(self, content: str, chunk_id: str, parent_chunk_id: str, content_type: str):
        self.chunk_id = chunk_id
        self.parent_chunk_id = parent_chunk_id
        self.content = content
        self.content_type = content_type # "summary" or "hypothetical_question"

class ParentChunk:
    """父文档块，包含原始的、更长的文本内容。"""
    def __init__(self, content: str, chunk_id: str = None, metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.content_hash = self._generate_hash(content)

    def _generate_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """将 ParentChunk 转换为用于 Qdrant payload 的字典"""
        payload = {
            "parent_content": self.content,
            "parent_id": self.chunk_id,
            "parent_hash": self.content_hash,
        }
        payload.update(self.metadata)
        return payload

def _call_llm_for_tasks(text_chunk: str) -> dict:
    """【核心修复】调用LLM并从返回文本中稳健地提取JSON。"""
    prompt = f"""为以下文本块生成一个简洁摘要和一个可能的用户问题。请严格按照JSON格式返回，并且必须包含 "summary" 和 "hypothetical_question" 两个键。

[文本块]
{text_chunk}

[输出JSON]
"""
    response_text = "" # 初始化以备在异常记录中使用
    try:
        client = get_ollama_client()
        response = client.chat(
            model=config.OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
            stream=False,
            format="json"  # <-- 强制Ollama返回JSON格式
        )
        
        response_text = response['message']['content'].strip()
        tasks = json.loads(response_text)

        # --- 【核心优化3】使用 .get() 安全地获取值，并检查内容是否为空 ---
        summary = tasks.get("summary")
        question = tasks.get("hypothetical_question")

        if summary and question:
            logger.info(f"成功为块生成摘要和假设性问题，原文始于: '{text_chunk[:100]}...'")
            return {"summary": summary, "hypothetical_question": question}
        else:
            # 即使返回了JSON，但如果缺少字段或字段为空，也视为失败
            raise ValueError(f"LLM返回的JSON不完整。收到的键: {list(tasks.keys())}")

    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"LLM生成任务失败或返回格式错误: {e}. 模型原始返回: '{response_text}'")
        # 降级策略
        return {
            "summary": text_chunk[:200] + "...",
            "hypothetical_question": f"这段关于 '{text_chunk[:50]}...' 的内容是什么？"
        }

def create_multi_vector_chunks(elements: List['ParsedElement'], filename: str) -> List[tuple[ParentChunk, List[ChildChunk]]]:
    """
    实现语义分块与HyDE，处理文档成父子块结构。
    """
    if not elements:
        logger.warning(f"文件 '{filename}' 没有可切分的内容。")
        return []

    # 1. 语义分块：根据文档结构（如Title）将元素组合成逻辑块
    logical_chunks = []
    current_chunk = []
    for el in elements:
        # 确保元素内容是字符串并且不为空
        if not isinstance(el.content, str) or not el.content.strip():
            continue

        if el.type == "Title":
            if current_chunk:
                logical_chunks.append(current_chunk)
            current_chunk = [el]
        else:
            current_chunk.append(el)
    if current_chunk:
        logical_chunks.append(current_chunk)

    # 2. 创建父文档块
    parent_chunks = []
    for chunk_elements in logical_chunks:
        chunk_text = "\n\n".join([el.content for el in chunk_elements if el.content])
        if len(chunk_text) > MAX_CHUNK_CHARS:
            for i in range(0, len(chunk_text), MAX_CHUNK_CHARS):
                parent_chunks.append(ParentChunk(content=chunk_text[i:i+MAX_CHUNK_CHARS].strip(), metadata={"source": filename}))
        elif chunk_text.strip():
            parent_chunks.append(ParentChunk(content=chunk_text.strip(), metadata={"source": filename}))

    # 3. 为每个父块生成多个子块（摘要 + 假设性问题）
    multi_vector_chunks = []
    for parent in parent_chunks:
        tasks = _call_llm_for_tasks(parent.content)
        
        child_chunks = [
            ChildChunk(
                content=tasks["summary"],
                chunk_id=str(uuid.uuid4()),
                parent_chunk_id=parent.chunk_id,
                content_type="summary"
            ),
            ChildChunk(
                content=tasks["hypothetical_question"],
                chunk_id=str(uuid.uuid4()),
                parent_chunk_id=parent.chunk_id,
                content_type="hypothetical_question"
            )
        ]
        multi_vector_chunks.append((parent, child_chunks))
        
    logger.info(f"为 '{filename}' 创建了 {len(multi_vector_chunks)} 个父块，总计 {len(multi_vector_chunks)*2} 个子块。")
    return multi_vector_chunks

# 旧的入口函数可以保持
def chunk_document(elements: List['ParsedElement'], filename: str):
    logger.warning("chunk_document 已被弃用，请直接调用 create_multi_vector_chunks。")
    return []