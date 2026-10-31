import os

# 👇 强制系统使用 UTF-8 编码，解决 Windows 和云端中文字符报错
os.environ["PYTHONIOENCODING"] = "utf-8"

import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
import re
import json
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_pdf_viewer import pdf_viewer

# ================= 1. 页面基础配置与自定义美化 CSS =================
st.set_page_config(page_title="科研小助手 Pro", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main-title {font-size: 3rem !important; font-weight: 800; background: -webkit-linear-gradient(45deg, #1E90FF, #00CED1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0rem;}
    .sub-title {font-size: 1.2rem; color: #666; margin-bottom: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 8px; padding: 10px 20px; font-weight: 600;}
    .stTabs [aria-selected="true"] {background-color: #1E90FF; color: white;}
    .stButton>button {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤖 科研小助手 Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">深度融合大模型推理与 Plan-and-Solve 架构的全流程科研辅助工具</p>',
            unsafe_allow_html=True)

# ================= 2. 侧边栏鉴权与状态管理 =================
with st.sidebar:
    st.markdown("### 🔑 鉴权配置")
    api_key_from_secrets = None
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            api_key_from_secrets = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass

    if api_key_from_secrets:
        API_KEY = api_key_from_secrets
        st.success("✅ 已成功读取云端密钥")
    else:
        input_api_key = st.text_input("请输入 DeepSeek API Key", type="password")
        API_KEY = input_api_key if input_api_key else ""

    if not API_KEY:
        st.warning("⚠️ 请先在左侧输入您的 API Key")
        client = None
    else:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    st.divider()
    st.markdown("### ⚙️ 状态管理")
    if st.button("🗑️ 清空对话与缓存"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("缓存已清空！")
        time.sleep(1)
        st.rerun()

    # ================= 3. 辅助函数：防弹高亮机制 =================


def robust_highlight(pdf_bytes, target_text, best_chunk):
    """结合 RAG 召回的最佳文本块，进行防弹级精准高亮"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        clean_target = target_text.strip()
        if len(clean_target) < 2:
            return []

        # 策略一：优先在 RAG 召回的最佳段落里找
        if clean_target in best_chunk:
            for line in best_chunk.split('\n'):
                if clean_target in line:
                    search_snippet = re.sub(r'[^a-zA-Z0-9\.\s]', ' ', line).strip()[:30]
                    if len(search_snippet) > 2:
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            text_instances = page.search_for(search_snippet)
                            if text_instances:
                                rect = text_instances[0]
                                doc.close()
                                return [{"page": page_num + 1, "x": rect.x0, "y": rect.y0, "width": rect.width,
                                         "height": rect.height, "color": "yellow"}]

        # 策略二：全篇暴力搜索包含目标词的文本块（字数大于20防标题）
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            for b in blocks:
                block_text = b[4]
                if clean_target in block_text and len(block_text) > 20:
                    x0, y0, x1, y1 = b[:4]
                    doc.close()
                    return [{"page": page_num + 1, "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                             "color": "yellow"}]
        doc.close()
    except Exception:
        pass
    return []


# ================= 4. Agent 工具库 =================
def tool_search_arxiv(query):
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=2'
    try:
        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        results = [f"标题: {entry.find('{http://www.w3.org/2005/Atom}title').text.strip()}" for entry in
                   root.findall('{http://www.w3.org/2005/Atom}entry')]
        return "\n".join(results) if results else "未找到相关论文。"
    except Exception as e:
        return f"网络超时: {str(e)}"


def tool_query_pdf(question):
    if "chunks" not in st.session_state:
        return "用户尚未上传本地论文。"
    keywords = [w for w in re.findall(r'\w+', question) if len(w) > 2]
    scored = [(sum(1 for kw in keywords if kw.lower() in c.lower()), c) for c in st.session_state.chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    context = "\n\n".join([c[1] for c in scored[:3]])
    try:
        res = client.chat.completions.create(model="deepseek-chat",
                                             messages=[{"role": "system", "content": f"根据上下文回答：{context}"},
                                                       {"role": "user", "content": question}], timeout=60)
        return res.choices[0].message.content
    except Exception as e:
        return f"检索失败: {str(e)}"


# ================= 5. 标签页定义 =================
tab_agent, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧠 深度规划 Agent", "📄 双栏阅读与精读", "📚 多文综述", "📊 数据图表", "🌐 arXiv 检索", "🔍 引用核验", "📝 论文框架",
    "🌍 学术翻译"
])

# ----------------- Tab Agent: Plan-and-Solve 自主规划中枢 -----------------
with tab_agent:
    st.subheader("🧠 Plan-and-Solve 深度规划 Agent")
    st.info("💡 提出复杂问题，Agent 会先制定**全局任务计划**，然后逐步执行并自动汇总报告。")

    agent_file = st.file_uploader("👉 如需让 Agent 读论文，请先在此上传 PDF", type=["pdf"], key="agent_pdf")
    if agent_file and st.button("加载到 Agent 大脑"):
        if client:
            with st.spinner("解析中..."):
                doc = fitz.open(stream=agent_file.read(), filetype="pdf")
                text = "".join([page.get_text() for page in doc])
                st.session_state.chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(
                    text)
                st.success("✅ Agent 已记住这篇论文！")
        else:
            st.error("请先配置 API Key！")

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    for msg in st.session_state.agent_messages:
        if msg.get("role") != "tool" and msg.get("content"):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("输入复杂科研需求..."):
        if not client:
            st.error("请配置 API Key！")
        else:
            st.session_state.agent_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.status("🧠 阶段一：规划任务...", expanded=True) as status_plan:
                    planner_prompt = """拆解用户需求为JSON格式：{"steps":[{"id":1,"task":"...","tool":"tool_query_pdf|tool_search_arxiv|None","query":"..."}]}"""
                    try:
                        plan_response = client.chat.completions.create(model="deepseek-chat", messages=[
                            {"role": "system", "content": planner_prompt}, {"role": "user", "content": prompt}])
                        plan_text = plan_response.choices[0].message.content
                        json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
                        plan_json = json.loads(json_match.group(0)) if json_match else json.loads(plan_text)
                        steps = plan_json.get("steps", [])
                    except Exception:
                        # 🚀 兜底计划，防止大模型抽风导致崩溃
                        steps = [
                            {"id": 1, "task": "阅读本地论文核心贡献", "tool": "tool_query_pdf",
                             "query": "这篇论文的核心贡献是什么？"},
                            {"id": 2, "task": "联网检索最新研究", "tool": "tool_search_arxiv",
                             "query": "Graph Neural Networks"},
                            {"id": 3, "task": "综合生成最终报告", "tool": "None", "query": prompt}
                        ]

                    st.write(f"📋 **生成 {len(steps)} 个步骤的计划**")
                    for s in steps:
                        st.write(f"🔹 **步骤 {s['id']}**: {s['task']} (调用: `{s['tool']}`)")
                    status_plan.update(label="✅ 规划完成", state="complete", expanded=False)

                if steps:
                    with st.status("🛠️ 阶段二：执行中...", expanded=True) as status_exec:
                        execution_results = []
                        for step in steps:
                            tool_name = step.get("tool", "None")
                            query_str = step.get("query", "")
                            st.write(f"▶️ 执行：{step.get('task')}")

                            if tool_name == "tool_query_pdf":
                                result = tool_query_pdf(query_str)
                            elif tool_name == "tool_search_arxiv":
                                result = tool_search_arxiv(query_str)
                            else:
                                res = client.chat.completions.create(model="deepseek-chat",
                                                                     messages=[{"role": "user", "content": query_str}])
                                result = res.choices[0].message.content
                            execution_results.append(f"【{step.get('task')}】结果:\n{result}")
                        status_exec.update(label="✅ 执行完毕", state="complete", expanded=False)

                    with st.spinner("📝 阶段三：生成报告..."):
                        synthesis_prompt = f"用户问题：{prompt}\n参考信息：{chr(10).join(execution_results)}\n请直接生成最终报告。"
                        final_response = client.chat.completions.create(model="deepseek-chat", messages=[
                            {"role": "user", "content": synthesis_prompt}])
                        final_answer = final_response.choices[0].message.content
                        st.markdown("### 🎯 最终综合报告")
                        st.markdown(final_answer)
                        st.session_state.agent_messages.append({"role": "assistant", "content": final_answer})

# ----------------- Tab 1: 双栏阅读与精读 -----------------
with tab1:
    st.subheader("📄 原生双栏阅读与智能精读")
    uploaded_file = st.file_uploader("请上传 PDF 论文", type=["pdf"], key="single_file")

    if uploaded_file:
        pdf_bytes = uploaded_file.getvalue()
        if st.button("🚀 解析并构建 RAG 索引"):
            with st.spinner("解析中..."):
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = "".join([page.get_text() for page in doc])
                st.session_state.chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(
                    text)
                st.success("✅ 解析成功！已构建索引。")

        col_pdf, col_chat = st.columns([1.5, 1])
        with col_pdf:
            st.markdown("### 📖 PDF 原文 (精准高亮)")
            annotations = st.session_state.get("pdf_annotations", [])
            pdf_viewer(pdf_bytes, width="100%", height=800, annotations=annotations,
                       scroll_to_annotation=1 if annotations else None)

        with col_chat:
            st.markdown("### 💬 智能问答")
            if prompt := st.chat_input("针对这篇论文提问...", key="single_chat"):
                if not client:
                    st.error("请配置 API Key！")
                elif "chunks" not in st.session_state:
                    st.error("请先点击解析论文！")
                else:
                    st.chat_message("user").markdown(prompt)
                    with st.spinner("正在检索并定位原文..."):
                        keywords = [w for w in re.findall(r'\w+', prompt) if len(w) > 2]
                        scored = [(sum(1 for kw in keywords if kw.lower() in c.lower()), c) for c in
                                  st.session_state.chunks]
                        scored.sort(key=lambda x: x[0], reverse=True)
                        best_chunk = scored[0][1] if scored else ""
                        context = "\n\n".join([c[1] for c in scored[:3]])

                        with st.chat_message("assistant"):
                            try:
                                response = client.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=[{"role": "system", "content": f"根据上下文简明扼要地回答：{context}"},
                                              {"role": "user", "content": prompt}],
                                    stream=True
                                )
                                full_answer = st.write_stream(response)

                                numbers = re.findall(r'\d+\.\d+|\d+', full_answer)
                                valid_numbers = [n for n in numbers if len(n) >= 2 or "." in n]

                                new_annotations = []
                                for num in valid_numbers:
                                    new_annotations = robust_highlight(pdf_bytes, num, best_chunk)
                                    if new_annotations:
                                        break

                                if not new_annotations:
                                    eng_words = re.findall(r'[A-Za-z]{4,}', full_answer)
                                    for word in set(eng_words):
                                        new_annotations = robust_highlight(pdf_bytes, word, best_chunk)
                                        if new_annotations:
                                            break

                                if new_annotations:
                                    st.session_state.pdf_annotations = new_annotations
                                    st.toast("🎯 已锁定包含答案的原文段落！")
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.toast("⚠️ 未能定位到原文。")
                            except Exception as e:
                                st.error(f"❌ 请求失败: {e}")

# ----------------- Tab 2: 多文综述 -----------------
with tab2:
    st.subheader("📚 多篇论文联合对比与综述生成")
    multi_files = st.file_uploader("请上传多篇 PDF", type=["pdf"], accept_multiple_files=True, key="multi_files")
    if multi_files and st.button("提取并生成综述"):
        if client:
            combined_text = "".join([f"\n--- {f.name} ---\n" + "".join(
                [p.get_text() for p in fitz.open(stream=f.read(), filetype="pdf")])[:4000] for f in multi_files])
            res = client.chat.completions.create(model="deepseek-chat", messages=[
                {"role": "user", "content": f"对比生成综述：{combined_text}"}])
            st.info(res.choices[0].message.content)

# ----------------- Tab 3: 数据图表 -----------------
with tab3:
    st.subheader("📊 CSV 实验数据自动可视化")
    csv_file = st.file_uploader("请上传 CSV", type=["csv"], key="csv_file")
    if csv_file and st.button("生成图表与图注"):
        if client:
            df = pd.read_csv(csv_file)
            # 🚀 防崩溃保护：检查是否有足够的列
            if df.shape[1] < 2:
                st.error("⚠️ CSV 文件格式错误：请确保文件至少包含两列数据（如：X轴, Y轴）。")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', color='b')
                st.pyplot(fig)
                res = client.chat.completions.create(model="deepseek-chat", messages=[
                    {"role": "user", "content": f"英文图注：{df.describe().to_string()}"}])
                st.success(f"**图注：**\n{res.choices[0].message.content}")

# ----------------- Tab 4: arXiv 检索 -----------------
with tab4:
    st.subheader("🌐 arXiv 实时联网检索")
    query = st.text_input("关键词:", key="arxiv_query")
    if st.button("搜索") and query:
        st.info(tool_search_arxiv(query))

# ----------------- Tab 5: 引用核验 -----------------
with tab5:
    st.subheader("🔍 参考文献防幻觉核验")
    ref = st.text_input("输入要核验的论文标题:", key="ref_query")
    if st.button("开始核验") and ref:
        url = f'http://export.arxiv.org/api/query?search_query=ti:{ref}&start=0&max_results=1'
        try:
            root = ET.fromstring(requests.get(url, timeout=10).content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            if entries and "withdrawn" not in entries[0].find('{http://www.w3.org/2005/Atom}title').text.lower():
                st.success(f"✅ 核验通过！真实标题：{entries[0].find('{http://www.w3.org/2005/Atom}title').text.strip()}")
            else:
                st.error("❌ 核验失败！该引用极有可能是大模型产生的幻觉！")
        except:
            st.warning("⚠️ 网络超时。")

# ----------------- Tab 6: 论文框架 -----------------
with tab6:
    st.subheader("📝 论文框架与架构图生成")
    idea = st.text_area("输入 Idea:", key="idea_input")
    if st.button("生成") and idea:
        if client:
            res = client.chat.completions.create(model="deepseek-chat",
                                                 messages=[{"role": "user", "content": f"生成论文框架：{idea}"}])
            st.info(res.choices[0].message.content)

# ----------------- Tab 7: 学术翻译 -----------------
with tab7:
    st.subheader("🌍 学术级中英互译")
    text_to_translate = st.text_area("请输入需要翻译的文本:", height=200)
    direction = st.radio("翻译方向", ["中文 -> 英文", "英文 -> 中文"], horizontal=True)
    if st.button("🔠 开始翻译") and text_to_translate:
        if client:
            # 🚀 Bug 修复：精确判断方向，绝不混淆
            prompt_dir = "中译英" if direction == "中文 -> 英文" else "英译中"
            prompt = f"学术翻译（{prompt_dir}），保留公式：{text_to_translate}"
            res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
            st.success(res.choices[0].message.content)