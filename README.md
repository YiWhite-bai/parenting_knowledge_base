# 育儿知识库 (Parenting Knowledge Base)

智能育儿知识库系统，基于 LangGraph + FastAPI + Milvus + BGE-M3 + LLM。

当前版本为纯 RAG 实现，不包含 Neo4j / 知识图谱链路。

## 快速启动

### 1. 环境准备
```bash
conda activate parenting_kb
pip install -e .
```

### 2. 导入育儿数据
```bash
# 启动导入服务
python -m knowledge.api.import_router
# 或
uvicorn knowledge.api.import_router:create_app --host 0.0.0.0 --port 8000
```

然后调用 API：
```bash
curl -X POST http://localhost:8000/import/directory \
  -H "Content-Type: application/json" \
  -d '{"source_dir": "D:/育儿/数据"}'
```

该接口会立即返回 `task_id`，后台异步导入目录内的 PDF / Markdown 文件。用下面的接口查看进度：

```bash
curl http://localhost:8000/status/<task_id>
```

目录导入会先把源文件复制到 `knowledge/temp_data/<日期>/<task_id>/directory_import/` 下，再对临时副本执行 PDF 解析、切分和入库；原始数据目录不会写入 `chunks.json` 或 MinerU 中间产物。

### 3. 查询育儿知识
```bash
# 启动查询服务
python -m knowledge.api.query_router
# 或
uvicorn knowledge.api.query_router:create_app --host 0.0.0.0 --port 8001
```

然后调用 API：
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "3-6岁孩子发脾气怎么办？", "is_stream": false}'
```

### 4. 流式查询
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "孩子挑食有什么好方法？", "is_stream": true}'
# 返回 task_id，然后用 SSE 连接获取流式结果
```

### 5. 育儿知识推荐
```bash
curl -X POST http://localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "孩子发脾气怎么办", "age_group": "3-6岁", "problem_type": "情绪管理", "top_k": 5}'
```

### 6. 案例检索
```bash
curl -X POST http://localhost:8001/cases/search \
  -H "Content-Type: application/json" \
  -d '{"query": "孩子睡前拖延反复下床", "age_group": "3-6岁", "scene": "睡前拖延", "top_k": 5}'
```

### 7. 通用知识检索
```bash
curl -X POST http://localhost:8001/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "如何培养孩子专注力", "content_type": "知识科普", "top_k": 5}'
```

## 项目架构

```
knowledge/
├── api/                        # FastAPI 路由
│   ├── import_router.py        # 导入服务 :8000
│   └── query_router.py         # 查询服务 :8001
├── service/                    # 业务服务层
├── processor/
│   ├── import_process/         # 导入流水线 (LangGraph)
│   │   └── nodes/              # entry → pdf_to_md → parenting_metadata
│   │                           #   → document_split → embedding → milvus
│   └── query_process/          # 查询流水线 (LangGraph)
│       └── nodes/              # search_router → (vector | hyde)
│                               #   → rrf → reranker → answer
├── prompts/                    # LLM 提示词
├── schema/                     # Pydantic 数据模型
└── utils/                      # 工具层
    └── client/                 # Milvus/MinIO/MongoDB/LLM 客户端
```

## 基础设施依赖

| 组件 | 地址 | 用途 |
|------|------|------|
| Milvus | 192.168.142.130:19530 | 向量检索 |
| MongoDB | 192.168.142.130:27017 | 对话历史 |
| MinIO | 192.168.142.130:9000 | 文件/图片存储 |
| DashScope | dashscope.aliyuncs.com | LLM (qwen-flash) |
| BGE-M3 | 本地 | 向量嵌入 + 重排序 |

## 育儿元数据

导入时会尽量为每个切片写入以下字段，便于检索过滤和答案溯源：

- `file_title` / `title`
- `author`
- `age_group`
- `content_type`
- `problem_type`
- `scene`
- `source_file`
- `source_path`

## 检索接口

- `POST /recommend`：面向育儿建议、专家文章、沟通话术和知识科普的推荐检索。
- `POST /cases/search`：只检索 `content_type=亲子案例` 的案例片段。
- `POST /knowledge/search`：通用元数据检索，可按 `content_type`、`age_group`、`problem_type`、`scene` 过滤。
