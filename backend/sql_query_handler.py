# /* 文件名: backend/sql_query_handler.py, 版本号: 1.3 */
"""
Text-to-SQL 核心处理器 V1.3
- 移除 get_database_schema 方法的 table_names 参数。
- 程序初始化时直接从 config 模块读取 SQL_INCLUDED_TABLES 配置。
- 根据配置决定是获取指定表还是所有表的结构。
"""

import logging
import os
import re
from typing import List
import ollama
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from fastapi import Response

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SQLQueryHandler:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("环境变量 DATABASE_URL 未设置，无法连接到数据库。")
        
        try:
            self.engine = create_engine(self.db_url, pool_pre_ping=True)
            logger.info("数据库连接引擎创建成功。")
        except Exception as e:
            logger.error(f"创建数据库引擎失败: {e}", exc_info=True)
            raise
            
        self.llm_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        # --- 【核心修改】在初始化时加载配置 ---
        self.included_tables = config.SQL_INCLUDED_TABLES
        if self.included_tables:
            logger.info(f"Text-to-SQL 将限制在以下表: {self.included_tables}")
        else:
            logger.info("Text-to-SQL 将尝试访问数据库中的所有非系统表。")

    def get_database_schema(self) -> str:
        """
        根据配置动态获取数据库表的 CREATE TABLE 语句。
        """
        schema_str = ""
        try:
            with self.engine.connect() as connection:
                inspector = inspect(self.engine)
                current_schema = connection.dialect.default_schema_name or inspector.default_schema_name

                # --- 【核心修改】使用 self.included_tables ---
                tables_to_fetch = self.included_tables
                if not tables_to_fetch:
                    logger.info(f"未在配置中指定表，将自动扫描 Schema: '{current_schema}'...")
                    all_tables = inspector.get_table_names(schema=current_schema)
                    system_tables = {'information_schema', 'mysql', 'performance_schema', 'sys'}
                    tables_to_fetch = [t for t in all_tables if t not in system_tables]
                
                logger.info(f"将要获取以下表的结构: {tables_to_fetch}")

                for table_name in tables_to_fetch:
                    try:
                        query = text(f"SHOW CREATE TABLE `{table_name}`")
                        result = connection.execute(query).fetchone()
                        if result and len(result) > 1:
                            schema_str += result[1] + ";\n\n"
                    except SQLAlchemyError as table_error:
                         logger.warning(f"无法获取表 '{table_name}' 的结构: {table_error}")
                         continue
                        
            if not schema_str:
                logger.warning("未能从数据库中获取任何有效的表结构。")
            return schema_str

        except Exception as e:
            logger.error(f"获取数据库表结构时发生未知错误: {e}", exc_info=True)
            return f"/* 获取表结构时发生未知错误: {e} */"

    def generate_sql_query(self, user_question: str, db_schema: str) -> str:
        """
        根据用户问题和【一个或多个】数据库表结构，调用Ollama模型生成SQL查询。
        """
        if not db_schema or "无法获取" in db_schema:
            return "-- 无法生成SQL，因为未能获取到数据库表结构。"
            
        prompt = f"""你是一个顶级的MySQL数据库专家，擅长将自然语言问题转换成SQL查询。

**任务**:
根据下面提供的MySQL表结构，为用户的问题生成一个可执行的SQL查询语句。

**思考步骤**:
1.  **理解问题**: 仔细阅读用户的[用户问题]。
2.  **分析表结构**: 查看[数据库表结构]，找出回答问题所需的一个或多个表以及相关的字段。
3.  **构建查询**: 编写SQL查询。如果需要模糊查询，请使用 `LIKE` 和 `%` 操作符。

**输出规则**:
-   你的最终回答**必须**只包含纯SQL代码。
-   不要添加任何解释、注释、或Markdown的代码块标记（例如 ```sql ... ```）。
-   如果根据提供的表结构无法回答问题，则只返回一行注释：`-- 根据提供的表信息，无法回答此问题`。

[数据库表结构]
{db_schema}
[用户问题]
{user_question}

[SQL查询]
"""
        
        try:
            # --- 【核心优化】增强日志，打印完整的Prompt ---
            logger.info("正在向 Ollama 发送 Text-to-SQL 请求... 完整的Prompt如下:")
            logger.debug(prompt) # 使用 DEBUG 级别以避免刷屏，您可以根据需要调整为 INFO

            response = self.llm_client.chat(
                model=config.OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                stream=False
            )
            
            raw_sql = response['message']['content'].strip()
            logger.info(f"Ollama 返回的原始SQL: {raw_sql}")
            
            # 增强清理逻辑，应对模型可能返回的思考过程
            if "-- 根据提供的表信息，无法回答此问题" in raw_sql:
                return "-- 根据提供的表信息，无法回答此问题"

            cleaned_sql = re.sub(r"```(sql)?", "", raw_sql).strip()
            
            select_pos = cleaned_sql.upper().find("SELECT")
            if select_pos != -1:
                cleaned_sql = cleaned_sql[select_pos:]
            
            if cleaned_sql and not cleaned_sql.endswith(';'):
                cleaned_sql += ';'
            
            logger.info(f"清理后的SQL: {cleaned_sql}")
            return cleaned_sql if cleaned_sql else "-- 模型未能生成有效的SQL查询。"

        except Exception as e:
            logger.error(f"调用Ollama模型生成SQL时出错: {e}", exc_info=True)
            return f"-- 调用LLM时出错: {e}"

if __name__ == "__main__":
    try:
        sql_handler = SQLQueryHandler()
        schema = sql_handler.get_database_schema()
        print("--- 根据 .env 配置获取到的数据库表结构 ---")
        print(schema)
        print("---------------------------------")
        
        question = "统计每个用户创建了多少个对话(chats)？请列出用户名和对话数量。"
        
        if schema:
            generated_sql = sql_handler.generate_sql_query(question, schema)
            print(f"\n用户问题: {question}")
            print(f"生成的SQL查询: \n{generated_sql}")

    except Exception as main_exc:
        logger.error(f"程序主流程执行失败: {main_exc}", exc_info=True)