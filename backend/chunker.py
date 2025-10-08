# 文件名: backend/chunker.py, 版本号: 3.1
"""
【V3.1 质量优化版】:
- 升级父文档切分策略，使用 langchain 的 RecursiveCharacterTextSplitter。
- 实现“语义感知”切分，优先按段落、句子进行分割，保证块的语义完整性。
"""
import hashlib
import logging
import uuid
from typing import List, Dict, Any, TYPE_CHECKING
import ollama 

# --- 新增导入 ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

# 避免循环导入
if TYPE_CHECKING:
    from document_parser import ParsedElement

logger = logging.getLogger(__name__)

# --- 配置常量 ---
# 定义父块的理想大小和重叠
PARENT_CHUNK_SIZE = 1024 
PARENT_CHUNK_OVERLAP = 200

class ChildChunk:
    """子文档块，通常是一个摘要。这部分内容将被向量化用于检索。"""
    def __init__(self, content: str, chunk_id: str, parent_chunk_id: str):
        self.chunk_id = chunk_id
        self.parent_chunk_id = parent_chunk_id
        self.content = content

class ParentChunk:
    """父文档块，包含原始文本。将作为最终的上下文喂给LLM。"""
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
            model=config.OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        summary = response['message']['content'].strip()
        logger.info(f"为块生成了摘要，原文始于: '{text_chunk[:100]}...'")
        return summary
    except Exception as e:
        logger.error(f"LLM生成摘要失败: {e}")
        return text_chunk[:200]

def create_multi_vector_chunks(elements: List['ParsedElement'], filename: str) -> List[tuple[ParentChunk, ChildChunk]]:
    """
    【核心升级函数】使用 RecursiveCharacterTextSplitter 处理文档成父子块结构。
    """
    if not elements:
        logger.warning(f"文件 '{filename}' 没有可切分的内容。")
        return []

    # 1. 初始化智能文本分割器
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "。", " ", ""] # 优先按段落、句子分割
    )

    # 2. 将所有解析出的元素内容拼接成一个长文本
    full_text = "\n\n".join([el.content for el in elements if el.content])
    
    # 3. 使用分割器创建语义完整的父文档块
    parent_chunk_contents = parent_splitter.split_text(full_text)

    parent_chunks = [
        ParentChunk(content=content.strip(), metadata={"source": filename})
        for content in parent_chunk_contents if content.strip()
    ]

    # 4. 为每个父块生成一个摘要（子块）
    multi_vector_chunks = []
    for parent in parent_chunks:
        summary = _call_llm_for_summary(parent.content)
        
        child = ChildChunk(
            content=summary,
            chunk_id=str(uuid.uuid4()),
            parent_chunk_id=parent.chunk_id
        )
        multi_vector_chunks.append((parent, child))
        
    logger.info(f"为 '{filename}' 创建了 {len(multi_vector_chunks)} 个父/子块对。")
    return multi_vector_chunks

# 旧的入口函数可以保持，以防其他地方调用
def chunk_document(elements: List['ParsedElement'], filename: str):
    logger.warning("chunk_document 已被弃用，请直接调用 create_multi_vector_chunks。")
    return []