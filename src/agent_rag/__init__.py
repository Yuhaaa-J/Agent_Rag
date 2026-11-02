"""agent-rag: 文档知识库智能体。

分层：
    ingest     离线入库（加载 -> 结构化切分 -> 幂等写入）
    retrieval  在线检索（向量 + BM25 混合，RRF 融合）
    agent      LangGraph 图（工具调用 + 写操作人工确认）
    api        FastAPI 服务（SSE 流式 + 文档管理）
"""

__version__ = "0.1.0"
