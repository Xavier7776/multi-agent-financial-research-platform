# Multi-Agent Financial Research Platform

多 Agent 协作的金融研报自动生成系统 —— 输入一家公司名称，自动生成 8 章节专业研报。

## 核心能力

- **多 Agent 协作**：Editor → Researcher → Writer → Reviewer → Reviser，LangGraph 6 节点编排
- **双源金融数据**：A 股/港股（雪球 API）+ 美股（FMP API），自动路由
- **LLM 股票识别**：支持中文公司名、股票代码、自然语言输入
- **8 章节研报**：公司概览、财务分析、估值、行业竞争、风险、ESG、技术面、投资建议
- **实时进度推送**：WebSocket 双向通信，前端实时展示研究进度
- **同行对比**：LLM 自动生成同行公司，交叉验证财务指标

## 架构

```
用户输入 → ChiefEditorAgent (LangGraph)
              ├── browser → 搜索
              ├── planner → 规划章节
              ├── human → 人工确认
              ├── researcher → 金融数据获取 + 网络搜索
              ├── writer → 逐章撰写
              └── publisher → 格式化输出
                    ↓
              WebSocket → 前端实时渲染
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -r multi_agents/requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填写：

```env
OPENAI_API_KEY=your_key          # LLM API
OPENAI_BASE_URL=your_endpoint
TAVILY_API_KEY=your_key          # 网络搜索
JINA_API_KEY=your_key            # Embeddings
```

### 3. 启动后端

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. 启动前端

```bash
cd frontend/nextjs
npm install
npm run dev
```

访问 `http://localhost:3000`，进入「深度研究」页面。

## 前端项目

独立部署在 `E:\chromeDownload\arc-portfolio`，arc-portfolio 风格的金融研究面板，WebSocket 实时通信。

## 部署

已部署在 Railway：

```bash
railway up
```

配置 `railway.toml` 中的启动命令和环境变量。

## 项目结构

```
├── backend/           # FastAPI 后端服务
├── multi_agents/      # 多 Agent 协作引擎
│   ├── agents/        # Editor, Researcher, Writer, Reviewer, Reviser
│   ├── components/    # 金融数据工具 (Xueqiu, FMP)
│   └── main.py        # WebSocket 入口
├── gpt_researcher/    # 单 Agent 搜索引擎（底层）
├── frontend/          # Next.js 前端
└── outputs/           # 研报输出目录
```

## 技术栈

| 层 | 技术 |
|---|------|
| Agent 编排 | LangGraph |
| LLM | OpenAI 兼容 API |
| 数据源 | 雪球 / FMP |
| 搜索 | Tavily |
| Embedding | Jina AI |
| 后端 | FastAPI + WebSocket |
| 前端 | Next.js + TypeScript |
| 部署 | Railway |
