# 🤖 科研小助手 Pro (Research Assistant Pro)

深度融合大模型推理与 Plan-and-Solve 架构的全流程科研辅助工具。摆脱传统低代码平台限制，采用纯 Python 手写架构，实现科研文献的高效精读、防幻觉核验与数据可视化。

## 🌐 在线体验
👉 **公网演示地址:** [https://enen-research-agent.streamlit.app](https://enen-research-agent.streamlit.app)

## ✨ 核心功能

1. 🧠 **深度规划 Agent (Plan-and-Solve)**: 面对复杂科研需求，Agent 会先拆解出全局任务清单（Planner），再逐步调用工具执行（Executor），最后综合生成学术报告（Synthesizer）。
2. 📄 **原生双栏阅读与溯源 (RAG & Traceability)**: 支持 PDF 原生双栏渲染。通过底层坐标反查，实现从大模型回答“一键定位并高亮”原文出处，彻底拒绝幻觉。
3. 🌐 **arXiv 联网检索**: 通过手写 HTTP 请求直连 arXiv 官方 API，获取最新前沿论文。
4. 🔍 **引用防幻觉核验**: 自动比对真实学术数据库，识别并拦截 AI 生成的虚假参考文献（幻觉）。
5. 📚 **多文综述与去重**: 支持多篇 PDF 联合对比生成综述；内置基于内容的 MD5 语义去重预警。
6. 📊 **数据可视化**: 自动将 CSV 数据转化为学术折线图，并调用大模型生成全英文 Figure Caption。
7. 🌍 **学术级翻译**: 支持中英学术文本一键互译，保留 LaTeX 公式与代码不译。

## 🛠️ 技术架构
- **前端框架**: Streamlit (纯 Python 手写全栈)
- **大模型引擎**: DeepSeek-V3 (接口调用)
- **核心逻辑**: 自研 ReAct / Plan-and-Solve 编排 / 轻量级关键词 RAG
- **云端部署**: Streamlit Community Cloud + Secrets 鉴权

## 🚀 本地运行指南
1. 克隆仓库: `git clone https://github.com/enen1-code/Research_Assistant_Pro.git`
2. 安装依赖: `pip install -r requirements.txt`
3. 运行应用: `streamlit run main.py`
