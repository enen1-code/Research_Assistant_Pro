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

# ================= 1. 页面基础配置与自定义美化 CSS =================
st.set_page_config(page_title="科研小助手 Pro", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1E90FF, #00CED1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .sub-title {font-size: 1.2rem; color: #666; margin-bottom: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {background-color: #1E90FF; color: white;}
    .stButton>button {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤖 科研小助手 Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">深度融合大模型推理与 Plan-and-Solve 架构的全流程科研辅助工具</p>',
            unsafe_allow_html=True)

# ================= 2. 侧边栏与全局状态管理 (本地/云端双兼容) =================
with st.sidebar:
    st.markdown("### 🔑 鉴权配置")

    # 1. 安全地尝试从 Streamlit Cloud 的 Secrets 中读取 Key
    api_key_from_secrets = None
    try:
        # 如果本地没有 secrets.toml，这里会报错，被 except 捕获
        if "DEEPSEEK_API_KEY" in st.secrets:
            api_key_from_secrets = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass

    # 2. 判断是否有云端 Key，如果没有，则显示输入框
    if api_key_from_secrets:
        API_KEY = api_key_from_secrets
        st.success("✅ 已成功读取云端密钥")
    else:
        input_api_key = st.text_input("请输入 DeepSeek API Key", type="password", help="格式以 sk- 开头")
        DEFAULT_KEY = ""  # ⚠️ 千万不要把真实 Key 写死在这里，传 GitHub 前必须留空！
        API_KEY = input_api_key if input_api_key else DEFAULT_KEY

    # 初始化大模型客户端 (加入空值保护)
    if not API_KEY:
        st.warning("⚠️ 请先在左侧输入您的 API Key 才能使用大模型功能。")
        client = None
    else:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    st.divider()
    st.markdown("### 💡 系统架构")
    st.info("- **前端**: Streamlit\n- **大模型**: DeepSeek-V3\n- **核心**: Plan-and-Solve Agent / 纯手写 RAG")

    st.markdown("### ⚙️ 状态管理")
    if st.button("🗑️ 清空对话与缓存"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("缓存已清空！")
        time.sleep(1)
        st.rerun()

    # ================= 3. 定义 Agent 工具库 =================


def tool_search_arxiv(query):
    """工具1：联网搜索 arXiv"""
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=2'
    try:
        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.content)
        results = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            results.append(f"标题: {entry.find('{http://www.w3.org/2005/Atom}title').text.strip()}")
        return "\n".join(results) if results else "未找到相关论文。"
    except Exception as e:
        return f"网络超时或无法访问 arXiv: {str(e)}"


def tool_query_pdf(question):
    """工具2：查询本地上传的论文"""
    if "chunks" not in st.session_state:
        return "用户尚未上传本地论文，无法回答。请提示用户先上传。"

    keywords = [w for w in re.findall(r'\w+', question) if len(w) > 2]
    scored = [(sum(1 for kw in keywords if kw.lower() in c.lower()), c) for c in st.session_state.chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    context = "\n\n".join([c[1] for c in scored[:3]])

    msg = [{"role": "system", "content": f"根据上下文回答：{context}"}, {"role": "user", "content": question}]
    try:
        res = client.chat.completions.create(model="deepseek-chat", messages=msg, timeout=60)
        return res.choices[0].message.content
    except Exception as e:
        return f"本地检索失败: {str(e)}"


# ================= 4. 标签页定义 =================
tab_agent, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧠 深度规划 Agent", "📄 单文精读", "📚 多文综述", "📊 数据图表", "🌐 arXiv 检索", "🔍 引用核验", "📝 论文框架"
])

# ----------------- Tab Agent: Plan-and-Solve 自主规划中枢 -----------------
with tab_agent:
    st.subheader("🧠 Plan-and-Solve 深度规划 Agent")
    st.info("💡 提出复杂问题，Agent 会先制定**全局任务计划**，然后逐步执行并自动汇总报告。")

    agent_file = st.file_uploader("👉 如需让 Agent 读论文，请先在此上传 PDF", type=["pdf"], key="agent_pdf")
    if agent_file and st.button("加载到 Agent 大脑"):
        if not client:
            st.error("请先配置 API Key！")
        else:
            with st.spinner("解析中..."):
                doc = fitz.open(stream=agent_file.read(), filetype="pdf")
                text = "".join([page.get_text() for page in doc])
                st.session_state.chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(
                    text)
                st.success("✅ Agent 已记住这篇论文！现在可以直接向它提问了。")

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # 展示历史记录
    for msg in st.session_state.agent_messages:
        if msg.get("role") != "tool" and msg.get("content"):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input(
            "输入复杂科研需求（例如：总结本地这篇论文的核心贡献，并去arXiv找两篇最新的相关论文进行对比）..."):
        if not client:
            st.error("⚠️ 请先在左侧侧边栏输入您的 DeepSeek API Key！")
        else:
            st.session_state.agent_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):

                # ================= 阶段一：Planner（规划者） =================
                with st.status("🧠 阶段一：Agent 正在全局规划任务...", expanded=True) as status_plan:
                    planner_prompt = """你是一个顶级科研 Agent。请将用户的复杂需求拆解为有序的任务列表。
你必须输出严格的 JSON 格式，不要包含任何其他废话。
JSON 格式要求：
{
  "steps": [
    {"id": 1, "task": "任务描述", "tool": "工具名", "query": "传给工具的参数"},
    {"id": 2, "task": "任务描述", "tool": "工具名", "query": "传给工具的参数"}
  ]
}
可用工具：
- tool_query_pdf: 查询本地已上传的PDF论文。query填用户的原始问题。
- tool_search_arxiv: 联网搜索arXiv。query填英文检索词。
- None: 如果不需要外部工具，直接由大模型自己思考，query填要思考的问题。"""

                    try:
                        time.sleep(0.5)  # 增加视觉停顿感
                        plan_response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "system", "content": planner_prompt},
                                      {"role": "user", "content": prompt}],
                            timeout=60
                        )
                        plan_text = plan_response.choices[0].message.content
                        json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
                        if json_match:
                            plan_json = json.loads(json_match.group(0))
                        else:
                            plan_json = json.loads(plan_text)

                        steps = plan_json.get("steps", [])
                        st.write(f"📋 **已生成包含 {len(steps)} 个步骤的执行计划**")
                        for step in steps:
                            st.write(f"🔹 **步骤 {step['id']}**: {step['task']} (调用: `{step['tool']}`)")
                        status_plan.update(label="✅ 任务规划完成！开始执行...", state="complete", expanded=False)
                    except Exception as e:
                        st.error(f"❌ 计划生成失败，JSON解析错误: {e}")
                        steps = []

                # ================= 阶段二：Executor（执行者） =================
                if steps:
                    with st.status("🛠️ 阶段二：正在逐步执行任务...", expanded=True) as status_exec:
                        execution_results = []

                        for step in steps:
                            task_name = step.get("task", "")
                            tool_name = step.get("tool", "None")
                            query_str = step.get("query", "")

                            st.write(f"▶️ **正在执行步骤 {step['id']}**: {task_name}")
                            time.sleep(0.5)

                            result = ""
                            if tool_name == "tool_query_pdf":
                                result = tool_query_pdf(query_str)
                                st.write(f"   └─ 📄 本地检索结果获取成功。")
                            elif tool_name == "tool_search_arxiv":
                                result = tool_search_arxiv(query_str)
                                st.write(f"   └─ 🌐 arXiv 联网搜索完成。")
                            else:
                                think_response = client.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=[{"role": "user", "content": query_str}],
                                    timeout=60
                                )
                                result = think_response.choices[0].message.content
                                st.write(f"   └─ 💡 模型逻辑推理完成。")

                            execution_results.append(f"【步骤 {step['id']} 结果】\n{result}")

                        status_exec.update(label="✅ 所有步骤执行完毕！", state="complete", expanded=False)

                    # ================= 阶段三：Synthesizer（综合者） =================
                    with st.spinner("📝 阶段三：Agent 正在汇总所有结果，生成最终报告..."):
                        synthesis_prompt = f"""用户原始问题：{prompt}
请你作为顶级的科研助手，根据以下分布执行获得的信息，给用户生成一份逻辑严密、格式精美的最终回复。不要提及“步骤”、“工具”等系统内部字眼，直接给结果。

【执行过程中收集到的信息】：
{chr(10).join(execution_results)}
"""
                        try:
                            time.sleep(0.5)
                            final_response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[{"role": "user", "content": synthesis_prompt}],
                                timeout=60
                            )
                            final_answer = final_response.choices[0].message.content

                            st.toast("✅ 报告生成完毕！", icon="🚀")
                            st.markdown("---")
                            st.markdown("### 🎯 最终综合报告")
                            st.markdown(final_answer)

                            # 一键导出功能
                            st.download_button(
                                label="📥 将报告保存为 Markdown 文件",
                                data=final_answer,
                                file_name="Research_Report.md",
                                mime="text/markdown",
                                help="将生成的学术报告保存到本地"
                            )

                            st.session_state.agent_messages.append({"role": "assistant", "content": final_answer})
                        except Exception as e:
                            st.error(f"❌ 综合报告生成失败: {e}")

# ----------------- Tab 1: 单文精读 -----------------
with tab1:
    st.subheader("📄 上传单篇论文，进行深度问答")
    uploaded_file = st.file_uploader("请上传 PDF 论文", type=["pdf"], key="single_file")
    if uploaded_file and st.button("🚀 解析并加载论文"):
        with st.spinner("正在解析..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in doc])
            st.session_state.chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(text)
            st.success(f"解析成功！已构建索引。")
    if prompt := st.chat_input("请输入针对论文的问题..."):
        if not client:
            st.error("请配置 API Key！")
        elif "chunks" not in st.session_state:
            st.error("请先上传并解析一篇 PDF 论文！")
        else:
            st.chat_message("user").markdown(prompt)
            with st.spinner("正在检索..."):
                keywords = [w for w in re.findall(r'\w+', prompt) if len(w) > 2]
                scored = [(sum(1 for kw in keywords if kw.lower() in c.lower()), c) for c in st.session_state.chunks]
                scored.sort(key=lambda x: x[0], reverse=True)
                context = "\n\n".join([c[1] for c in scored[:3]])
                with st.chat_message("assistant"):
                    try:
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "system", "content": f"根据上下文回答：{context}"},
                                      {"role": "user", "content": prompt}],
                            stream=True
                        )
                        st.write_stream(response)
                    except Exception as e:
                        st.error(f"请求超时: {e}")

# ----------------- Tab 2: 多文综述 -----------------
with tab2:
    st.subheader("📚 多篇论文联合对比与综述生成")
    multi_files = st.file_uploader("请一次性上传多篇 PDF 论文", type=["pdf"], accept_multiple_files=True,
                                   key="multi_files")
    if multi_files and st.button("🧠 提取多文核心并生成综述"):
        if not client:
            st.error("请配置 API Key！")
        else:
            with st.spinner("提取中..."):
                combined_text = ""
                for file in multi_files:
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    combined_text += f"\n\n--- {file.name} ---\n" + "".join([p.get_text() for p in doc])[:5000]
                st.session_state.multi_text = combined_text
                st.success("已加载！")
    if "multi_text" in st.session_state and st.button("✨ 生成领域演进对比综述"):
        with st.spinner("生成综述中..."):
            try:
                res = client.chat.completions.create(model="deepseek-chat", messages=[
                    {"role": "user", "content": f"对比这些论文并生成综述：{st.session_state.multi_text}"}])
                st.info(res.choices[0].message.content)
            except Exception as e:
                st.error(f"生成失败: {e}")

# ----------------- Tab 3: CSV 数据图表 -----------------
with tab3:
    st.subheader("📊 CSV 实验数据自动可视化")
    csv_file = st.file_uploader("请上传 CSV", type=["csv"], key="csv_file")
    if csv_file and st.button("📈 生成图表与图注"):
        if not client:
            st.error("请配置 API Key！")
        else:
            df = pd.read_csv(csv_file)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', color='b')
            st.pyplot(fig)
            try:
                res = client.chat.completions.create(model="deepseek-chat", messages=[
                    {"role": "user", "content": f"根据数据写一句英文图注：{df.describe().to_string()}"}])
                st.success(f"**图注：**\n{res.choices[0].message.content}")
            except Exception as e:
                st.error(f"图注生成失败: {e}")

# ----------------- Tab 4: arXiv 检索 -----------------
with tab4:
    st.subheader("🌐 arXiv 实时联网检索")
    query = st.text_input("关键词:", key="arxiv_query")
    if st.button("搜索") and query:
        with st.spinner("正在请求..."):
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
            st.warning("⚠️ 网络超时，无法连接核验服务器。")

# ----------------- Tab 6: 论文框架 -----------------
with tab6:
    st.subheader("📝 论文框架与架构图生成")
    idea = st.text_area("输入 Idea:", key="idea_input")
    if st.button("生成") and idea:
        if not client:
            st.error("请配置 API Key！")
        else:
            try:
                res = client.chat.completions.create(model="deepseek-chat",
                                                     messages=[{"role": "user", "content": f"生成论文框架：{idea}"}])
                st.info(res.choices[0].message.content)
            except Exception as e:
                st.error(f"生成失败: {e}")