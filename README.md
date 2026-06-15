# 育儿知识库

基于 RAG 的智能育儿知识问答系统。

## 技术栈

- **框架**: LangGraph + FastAPI
- **检索**: Milvus 混合检索（稠密 + 稀疏向量）
- **嵌入**: BGE-M3
- **问答**: LLM（DashScope）

## 快速开始

### 安装

```bash
conda activate parenting_kb
pip install -e .
```

### 启动导入服务（端口 8000）

```bash
python -m knowledge.api.import_router
```

上传文件或批量导入目录，也可打开 `http://localhost:8000/front/import.html` 使用 Web 界面。

### 启动问答服务（端口 8001）

```bash
python -m knowledge.api.query_router
```

打开 `http://localhost:8001/front/chat.html` 开始提问。

## 项目结构

```
knowledge/
├── api/                 # FastAPI 路由
├── service/             # 业务服务
├── processor/
│   ├── import_process/  # 导入流程（PDF/MD → 切分 → 向量化 → 入库）
│   └── query_process/   # 查询流程（意图分析 → 检索 → 排序 → 生成）
├── prompts/             # LLM 提示词
├── front/               # Web 前端页面
└── utils/               # 工具 & 客户端
```

## 依赖服务

| 组件 | 用途 |
|------|------|
| Milvus | 向量数据库 |
| MongoDB | 对话历史 |
| MinIO | 文件存储 |
| BGE-M3 | 向量嵌入 & 重排序 |
