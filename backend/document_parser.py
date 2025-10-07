# Filename: backend/document_parser.py, Version: 2.4
"""
文档解析模块。
负责将上传的各种格式的文件（PDF, DOCX, XLSX, JSON等）
转换为统一的、包含结构化信息的 ParsedElement 对象列表，
供下游的 chunker 模块使用。
"""
import io
import json
import logging
from typing import IO, List, Any, Dict
from dataclasses import dataclass, field

import pandas as pd
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element, Title
# The unnecessary 'import python_magic' has been removed from here

logger = logging.getLogger(__name__)

@dataclass
class ParsedElement:
    """
    一个标准化的数据结构，用于存放从文档中解析出的各类元素。
    """
    content: str
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ... (The rest of the file remains unchanged) ...
def parse_unstructured(file: IO[bytes], filename: str) -> List[ParsedElement]:
    """
    【升级版】使用 unstructured 库处理多种文件类型，并提取丰富的元数据。
    """
    try:
        elements: List[Element] = partition(
            file=file,
            file_filename=filename,
            include_page_breaks=True,
            strategy="hi_res"
        )
        
        parsed_elements = []
        for el in elements:
            meta = {
                "page_number": getattr(el.metadata, 'page_number', None),
                "filename": getattr(el.metadata, 'filename', filename),
            }
            element_type = el.__class__.__name__
            if isinstance(el, Title):
                meta['level'] = getattr(el.metadata, 'category_depth', None)

            parsed_elements.append(ParsedElement(
                content=el.text.strip(),
                type=element_type,
                metadata=meta
            ))
            
        logger.info(f"Successfully parsed '{filename}' with unstructured, found {len(elements)} elements.")
        return parsed_elements
    except Exception as e:
        logger.error(f"Error parsing '{filename}' with unstructured: {e}", exc_info=True)
        return [ParsedElement(
            content=f"无法使用unstructured解析文件 '{filename}'。错误: {e}",
            type="Error"
        )]

def parse_xlsx(file: IO[bytes], filename: str) -> List[ParsedElement]:
    """
    【核心升级】对于Excel文件，将每一行转换为一个独立的文本元素（一行一向量）。
    内容格式为："表头1 是 单元格1, 表头2 是 单元格2, ..."
    """
    try:
        # 使用 pandas 读取所有工作表
        all_sheets_dfs = pd.read_excel(file, sheet_name=None)
        parsed_elements = []
        
        for sheet_name, df in all_sheets_dfs.items():
            if not df.empty:
                # 获取表头
                headers = df.columns.tolist()
                
                # 遍历 DataFrame 的每一行
                for index, row in df.iterrows():
                    row_content_parts = []
                    # 遍历每一列，拼接成 "表头 是 单元格" 的格式
                    for header in headers:
                        cell_value = row[header]
                        # 确保单元格内容有效且不为空
                        if pd.notna(cell_value) and str(cell_value).strip():
                            row_content_parts.append(f"{str(header).strip()} 是 {str(cell_value).strip()}")
                    
                    # 如果行内有有效内容，则创建一个 ParsedElement
                    if row_content_parts:
                        full_row_content = ", ".join(row_content_parts)
                        parsed_elements.append(ParsedElement(
                            content=full_row_content,
                            type="TableRow",  # 定义一个新的类型，便于识别
                            metadata={
                                "filename": filename,
                                "sheet_name": sheet_name,
                                "row_number": index + 2  # Excel 行号通常从1开始，+1是表头，+1是0-based index
                            }
                        ))
                        
        logger.info(f"Successfully parsed Excel file '{filename}', found {len(parsed_elements)} rows to be vectorized.")
        return parsed_elements
    except Exception as e:
        logger.error(f"Error parsing XLSX file '{filename}': {e}", exc_info=True)
        return [ParsedElement(
            content=f"无法解析XLSX文件 '{filename}'。错误: {e}",
            type="Error"
        )]

def parse_json(file: IO[bytes], filename: str) -> List[ParsedElement]:
    """
    【升级版】对于JSON文件，转换为格式化的字符串。
    """
    try:
        content = json.load(file)
        formatted_json = json.dumps(content, ensure_ascii=False, indent=2)
        return [ParsedElement(
            content=formatted_json,
            type="CodeSnippet",
            metadata={"language": "json", "filename": filename}
        )]
    except Exception as e:
        logger.error(f"Error parsing JSON file '{filename}': {e}", exc_info=True)
        return [ParsedElement(
            content=f"无法解析JSON文件 '{filename}'。错误: {e}",
            type="Error"
        )]

# 将文件后缀映射到对应的解析函数
FILE_PARSERS = {
    'pdf': parse_unstructured,
    'docx': parse_unstructured,
    'txt': parse_unstructured,
    'md': parse_unstructured,
    'xlsx': parse_xlsx,
    'json': parse_json,
}

def get_parser(filename: str) -> callable:
    """根据文件后缀获取相应的解析函数"""
    ext = filename.split(".")[-1].lower()
    return FILE_PARSERS.get(ext, parse_unstructured)