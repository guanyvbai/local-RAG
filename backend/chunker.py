# 文件名: backend/chunker.py, 版本号: 2.3
"""
实现「语义与结构感知」的文档切分。
该模块负责将从 document_parser 解析出的元素列表，
转换为带有丰富元数据的、大小适中的文本块（Chunks）。
"""

import hashlib
import logging
import uuid
from typing import List, Dict, Any

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 假设 document_parser.py 在同一目录下
from document_parser import ParsedElement

logger = logging.getLogger(__name__)

# --- 配置常量 ---
DEFAULT_TOKENIZER_MODEL = "cl100k_base"
DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 50
PROTECTED_ELEMENT_TYPES = ("Table", "CodeSnippet", "Image")

class Chunk:
    """
    【升级版】一个数据结构，用于存放最终的文本块及其丰富的元数据。
    """
    def __init__(self,
                 content: str,
                 source: str,
                 chunk_id: str = None,
                 content_hash: str = None,
                 token_count: int = 0,
                 metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.content = content
        self.source = source
        self.token_count = token_count
        self.content_hash = content_hash or self._generate_hash(content)
        self.metadata = metadata or {}

    def _generate_hash(self, text: str) -> str:
        """为文本内容生成一个 SHA-256 哈希值"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """将 Chunk 对象转换为可用于 Qdrant payload 的字典"""
        payload = {
            "content": self.content,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
        }
        payload.update(self.metadata)
        return payload

class SemanticChunker:
    """
    实现「语义与结构感知」切分策略的核心类。
    """
    def __init__(self,
                 chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.tokenizer = tiktoken.get_encoding(DEFAULT_TOKENIZER_MODEL)
        except Exception:
            logger.warning("tiktoken cl100k_base not found, falling back to p50k_base.")
            self.tokenizer = tiktoken.get_encoding("p50k_base")
        
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,
            chunk_overlap=self.chunk_overlap * 4,
            length_function=len
        )

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def chunk(self, elements: List[ParsedElement], filename: str) -> List[Chunk]:
        """
        【核心修正】对文档解析出的元素列表进行切分的主函数。
        新增针对 Excel 表格行 (TableRow) 的特殊处理逻辑。
        """
        # 步骤 1: 检查是否为表格行数据，如果是，则直接处理
        if elements and all(el.type == "TableRow" for el in elements):
            logger.info(f"Detected table row data for '{filename}'. Applying one-row-per-chunk strategy.")
            return self._chunk_table_rows(elements, filename)

        # 步骤 2: 否则，执行常规的结构化切分
        sections = self._split_by_structure(elements)
        
        chunks = []
        if sections:
            for section in sections:
                chunks.extend(self._split_large_section_by_tokens(section, filename))

        # 步骤 3: 检查结果，如果结构化切分未产生任何 chunk，则启用备用方案
        if not chunks:
            logger.warning(
                f"Structural chunking for '{filename}' yielded no chunks. "
                "Switching to fallback recursive chunking."
            )
            chunks = self._fallback_chunk(elements, filename)
        
        return chunks

    def _chunk_table_rows(self, elements: List[ParsedElement], filename: str) -> List[Chunk]:
        """
        【新增】专门处理表格行元素的函数，实现“一行一向量”。
        """
        chunks = []
        for element in elements:
            if element.content:
                chunk = self._create_chunk(
                    content=element.content,
                    source=filename,
                    metadata=element.metadata
                )
                if chunk:
                    chunks.append(chunk)
        return chunks

    def _split_by_structure(self, elements: List[ParsedElement]) -> List[List[ParsedElement]]:
        """
        一级切分：根据标题 (Title) 结构将文档元素分组。
        """
        sections = []
        current_section = []
        
        for element in elements:
            if not element.content: continue

            if element.type == "Title":
                if current_section:
                    sections.append(current_section)
                current_section = [element]
            else:
                current_section.append(element)
        
        if current_section:
            sections.append(current_section)
            
        return sections

    def _split_large_section_by_tokens(self, section_elements: List[ParsedElement], filename: str) -> List[Chunk]:
        """
        二级切分：对一个大的逻辑章节，根据 token 限制进行精细切分。
        """
        chunks = []
        current_chunk_content = []
        current_chunk_tokens = 0
        
        heading_path = [el.content for el in section_elements if el.type == "Title"]
        base_metadata = {
            "heading_path": " / ".join(heading_path),
            "page_number": section_elements[0].metadata.get('page_number')
        }

        for element in section_elements:
            element_content = f"## {element.content}\n" if element.type == "Title" else element.content
            element_tokens = self._count_tokens(element_content)

            if element.type in PROTECTED_ELEMENT_TYPES and element_tokens > self.chunk_size:
                if current_chunk_content:
                    chunk = self._create_chunk("".join(current_chunk_content), filename, base_metadata)
                    if chunk: chunks.append(chunk)
                    current_chunk_content, current_chunk_tokens = [], 0
                chunk = self._create_chunk(element_content, filename, base_metadata)
                if chunk: chunks.append(chunk)
                continue

            if current_chunk_tokens + element_tokens > self.chunk_size and current_chunk_content:
                chunk = self._create_chunk("".join(current_chunk_content), filename, base_metadata)
                if chunk: chunks.append(chunk)
                overlap_content = self._get_overlap(current_chunk_content)
                current_chunk_content = [overlap_content, element_content]
                current_chunk_tokens = self._count_tokens("".join(current_chunk_content))
            else:
                current_chunk_content.append(element_content)
                current_chunk_tokens += element_tokens

        if current_chunk_content:
            chunk = self._create_chunk("".join(current_chunk_content), filename, base_metadata)
            if chunk: chunks.append(chunk)
            
        return chunks
    
    def _get_overlap(self, content_list: List[str]) -> str:
        """根据 token 数量计算重叠内容"""
        full_text = "".join(content_list)
        tokens = self.tokenizer.encode(full_text, disallowed_special=())
        if len(tokens) <= self.chunk_overlap:
            return full_text
        
        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.tokenizer.decode(overlap_tokens)

    def _create_chunk(self, content: str, source: str, metadata: Dict[str, Any]) -> Chunk:
        """创建一个新的 Chunk 对象并计算其 token 数量"""
        content = content.strip()
        if not content: return None
        token_count = self._count_tokens(content)
        return Chunk(content=content, source=source, token_count=token_count, metadata=metadata)

    def _fallback_chunk(self, elements: List[ParsedElement], filename: str) -> List[Chunk]:
        """
        备用切分策略：当文档缺乏结构信息时，使用传统的递归字符切分。
        """
        full_text = "\n\n".join([el.content for el in elements if el.content])
        if not full_text:
            return []
            
        text_chunks = self.fallback_splitter.split_text(full_text)
        
        chunks = []
        for text_chunk in text_chunks:
            chunk = self._create_chunk(
                content=text_chunk,
                source=filename,
                metadata={"chunking_strategy": "fallback_recursive"}
            )
            if chunk:
                chunks.append(chunk)
        return chunks

def chunk_document(elements: List[ParsedElement], filename: str) -> List[Chunk]:
    """
    【模块主入口】
    接收从 document_parser 解析出的元素列表，并返回一个 Chunk 列表。
    """
    if not elements:
        logger.warning(f"No content elements to chunk for file '{filename}'.")
        return []
    
    chunker = SemanticChunker()
    return chunker.chunk(elements, filename)