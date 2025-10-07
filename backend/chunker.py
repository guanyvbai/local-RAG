# 文件名: backend/chunker.py, 版本号: 3.0
"""
【V3.0 升级说明】:
- 实现多向量策略，为文档块生成摘要。
- 定义 ParentChunk 和 ChildChunk 来管理父子关系。
- 使用 LLM 来动态生成摘要。
"""
import hashlib
import logging
import uuid
from typing import List, Dict, Any, TYPE_CHECKING
import ollama 

import config

# 避免循环导入
if TYPE_CHECKING:
    from document_parser import ParsedElement

logger = logging.getLogger(__name__)

# --- 配置常量 ---
# 我们将大的文档块定义为“父块”
PARENT_CHUNK_SIZE = 1024 
# 父块的重叠部分
PARENT_CHUNK_OVERLAP = 200

class ChildChunk:
    """
    子文档块，通常是一个摘要或一个假设性问题。
    这部分内容将被向量化用于检索。
    """
    def __init__(self, content: str, chunk_id: str, parent_chunk_id: str):
        self.chunk_id = chunk_id
        self.parent_chunk_id = parent_chunk_id
        self.content = content

class ParentChunk:
    """
    父文档块，包含原始的、更长的文本内容。
    这部分内容将作为最终的上下文喂给LLM。
    """
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

def _call_llm_for_summary(text_chunk: str) -> str:
    """调用LLM为文本块生成一个简洁的摘要。"""
    prompt = f"""请为以下文本块生成一个简洁、精确的摘要，捕捉其核心思想。请直接返回摘要内容，不要添加任何额外的解释或引言。/no_think

[文本块]
{text_chunk}

[摘要]
"""
    try:
        client = ollama.Client(host=config.OLLAMA_BASE_URL)
        response = client.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        summary = response['message']['content'].strip()
        logger.info(f"Generated summary for chunk starting with: '{text_chunk[:100]}...'")
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary with LLM: {e}")
        # 在LLM调用失败时，返回原文的前200个字符作为降级策略
        return text_chunk[:200]

def create_multi_vector_chunks(elements: List['ParsedElement'], filename: str) -> List[tuple[ParentChunk, ChildChunk]]:
    """
    【核心函数】将文档元素处理成父子块结构。
    """
    if not elements:
        logger.warning(f"No content elements to chunk for file '{filename}'.")
        return []

    # 简单的文本拼接和切分作为父块
    full_text = "\n\n".join([el.content for el in elements if el.content])
    
    parent_chunks = []
    start_index = 0
    while start_index < len(full_text):
        end_index = start_index + PARENT_CHUNK_SIZE
        chunk_content = full_text[start_index:end_index]
        
        if end_index < len(full_text):
            overlap_end = end_index + PARENT_CHUNK_OVERLAP
            chunk_content = full_text[start_index:overlap_end]
        
        parent_chunks.append(ParentChunk(content=chunk_content.strip(), metadata={"source": filename}))
        start_index += PARENT_CHUNK_SIZE

    # 为每个父块生成一个摘要（子块）
    multi_vector_chunks = []
    for parent in parent_chunks:
        if not parent.content:
            continue
        
        summary = _call_llm_for_summary(parent.content)
        
        child = ChildChunk(
            content=summary,
            chunk_id=str(uuid.uuid4()),
            parent_chunk_id=parent.chunk_id
        )
        multi_vector_chunks.append((parent, child))
        
    logger.info(f"Created {len(multi_vector_chunks)} parent/child chunk pairs for '{filename}'.")
    return multi_vector_chunks

# 保留旧的 chunk_document 函数的入口，但现在它什么也不做或可以移除
def chunk_document(elements: List['ParsedElement'], filename: str):
    logger.warning("chunk_document is deprecated. Use create_multi_vector_chunks instead.")
    return []