import streamlit as st
import os
import json
import tempfile
from datetime import datetime
from excel_to_jsonl import convert_excel_to_jsonl
from logger import setup_logger
from gpt_fine_tuner import GPTFineTuner

logger = setup_logger(__name__)

LOCAL_DATA_PATH = "local_data.json"

# Default structure for local_data.json
DEFAULT_LOCAL_DATA = {
    "api_key": "",
    "train_file_id": "",
    "validation_file_id": "",
    "available_models": [],
    "job_id": "",
    "model_id": "",
    "system_prompt": "",
    "user_test_message": "",
    "temperature": "",
    "model_response": ""
}

# Global variable for local_data
local_data = None

def load_local_data():
    global local_data
    if local_data is not None:
        return local_data
    if not os.path.exists(LOCAL_DATA_PATH):
        with open(LOCAL_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_LOCAL_DATA, f, ensure_ascii=False, indent=2)
        logger.info("local_data.json created with default structure.")
        local_data = DEFAULT_LOCAL_DATA.copy()
        return local_data
    with open(LOCAL_DATA_PATH, "r", encoding="utf-8") as f:
        logger.info("local_data.json loaded.")
        local_data = json.load(f)
        return local_data

def save_local_data(data):
    global local_data
    with open(LOCAL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    local_data = data
    logger.info("local_data.json updated.")

def fetch_and_cache_available_models(api_key):
    """
    Fetch eligible models from OpenAI and cache them in local_data.json under 'available_models'.
    """
    from gpt_fine_tuner import GPTFineTuner
    global local_data
    if local_data is None:
        local_data = load_local_data()
    fine_tuner = GPTFineTuner(api_key=api_key)
    model_list = fine_tuner.list_available_models()
    def friendly_name(model_id):
        if "o4-mini" in model_id:
            return "o4 Mini"
        elif "o3-pro" in model_id or "gpt-4o-3-pro" in model_id:
            return "o3 Pro"
        elif "o3-mini" in model_id or "gpt-4o-3-mini" in model_id:
            return "o3 Mini"
        elif "o3" in model_id and "pro" not in model_id and "mini" not in model_id:
            return "o3"
        elif "gpt-4.1-nano" in model_id:
            return "GPT 4.1 Nano"
        elif "gpt-4.1-mini" in model_id:
            return "GPT 4.1 Mini"
        elif "gpt-4.1" in model_id or "gpt-4-1106" in model_id:
            return "GPT 4.1"
        elif "gpt-4o" in model_id and "mini" not in model_id:
            return "GPT 4o"
        elif "gpt-4o-mini" in model_id:
            return "GPT 4o Mini"
        elif "chatgpt-4o" in model_id:
            return "ChatGPT 4o"
        else:
            return model_id
    if model_list:
        available_models = [
            {"label": f"{friendly_name(m['id'])} ({m['id']})", "value": m['id']} for m in model_list
        ]
        logger.info(f"缓存模型列表: {available_models}")
    else:
        fallback_ids = [
            "gpt-4.1-mini-2025-04-14",
            "o4-mini-2025-04-16",
            "o3-2025-04-16",
            "o3-pro-2025-06-10",
            "o3-mini-2025-01-31",
            "gpt-4.1-2025-04-14",
            "gpt-4o-2024-11-20",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4o-mini-2024-07-18",
        ]
        available_models = [
            {"label": f"{friendly_name(mid)} ({mid})", "value": mid} for mid in fallback_ids
        ]
        logger.warning("未能从OpenAI获取模型列表，使用静态列表。")
    local_data["available_models"] = available_models
    save_local_data(local_data)
    return available_models

def home_page():
    st.title("Model FineTuner - Home")
    st.markdown("""
    ## Welcome to the Prompt FineTuner
    This tool helps you:
    1. Convert Excel data to the appropriate JSONL for fine-tuning
    2. Upload training/validation files to OpenAI
    3. Create and monitor fine-tuning model jobs
    4. Test your fine-tuned or any OpenAI model
    """)
    
    st.markdown("---")
    st.subheader("Instructions / Steps")
    st.markdown("""
    1. Set your OpenAI API Key down below.
    2. Go to **Convert Excel to JSONL** to prepare training/validation data.
    3. Go to **Upload Files to OpeanAI** and upload training (and optional validation) files.
    4. Create a fine-tuning job and monitor its progress.
    5. Test your model with custom prompts.
    """)
    st.markdown("---")

    # API Key input
    st.subheader("Set your OpenAI API Key")
    global local_data
    if local_data is None:
        local_data = load_local_data()
    api_key = st.text_input("OpenAI API Key", value=local_data.get("api_key", ""), type="password")
    if st.button("Save API Key"):
        local_data["api_key"] = api_key
        save_local_data(local_data)
        st.success("API Key saved to local_data.json!")
        logger.info("User saved OpenAI API Key.")
    # On first load, if available_models is missing or empty, fetch and cache
    if api_key:
        if not local_data.get("available_models"):
            st.info("正在获取可用模型列表...")
            fetch_and_cache_available_models(api_key)
    st.markdown("---")
    # Button to refresh model list
    st.subheader("Refresh Model List")
    if st.button("刷新可用模型列表"):
        if api_key:
            fetch_and_cache_available_models(api_key)
            st.success("模型列表已刷新！")
        else:
            st.warning("请先设置API Key。")

def excel_to_jsonl_page():
    st.title("Convert Excel to JSONL")
    st.markdown("""
    ### Step 1: Upload your Excel file
    - The file should contain columns for System Prompt, Input, and Output.
    - Supported formats: .xlsx, .xls
    """)
    uploaded_file = st.file_uploader("选择Excel文件", type=["xlsx", "xls"])
    st.markdown("---")
    st.markdown("### Step 2: Set output file name")
    today_str = datetime.now().strftime("-%Y-%m-%d")
    base_default = os.path.splitext(uploaded_file.name)[0] if uploaded_file else "output"
    user_base_name = st.text_input("输出JSONL文件名（不含前缀和日期后缀）", value="")
    if user_base_name.strip():
        output_filename = f"file-{user_base_name.strip()}{today_str}.jsonl"
    else:
        output_filename = f"file-{base_default}{today_str}.jsonl"
    st.markdown(f"**最终文件名:** `{output_filename}`")
    st.markdown("---")
    st.markdown("### Step 3: Convert and Save")
    save_dir = st.text_input("保存文件夹路径", value=os.getcwd())
    if st.button("开始转换并保存"):
        if not uploaded_file:
            st.error("请先上传一个Excel文件！")
            logger.error("用户未上传Excel文件，点击了开始转换并保存。")
        else:
            logger.info(f"用户上传了文件: {uploaded_file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp_excel:
                tmp_excel.write(uploaded_file.read())
                tmp_excel_path = tmp_excel.name
            logger.info(f"临时Excel文件路径: {tmp_excel_path}")
            output_path = os.path.join(save_dir, output_filename)
            logger.info(f"输出JSONL文件路径: {output_path}")
            try:
                logger.info("开始转换Excel文件到JSONL格式。")
                convert_excel_to_jsonl(tmp_excel_path, output_path)
                st.success(f"转换完成！文件已保存到: {output_path}")
                logger.info("转换成功，准备提供下载。")
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="额外下载JSONL文件",
                        data=f,
                        file_name=output_filename,
                        mime="application/jsonl"
                    )
            except Exception as e:
                st.error(f"转换失败: {e}")
                logger.error(f"转换失败: {e}")

def upload_files_page():
    st.title("Upload Files to OpenAI")
    st.markdown("""
    ### Step 1: Upload your training and (optional) validation JSONL files
    - These should be in the correct OpenAI format.
    """)
    global local_data
    if local_data is None:
        local_data = load_local_data()
    api_key = local_data.get("api_key", "")
    if not api_key:
        st.warning("Please set your OpenAI API Key on the Home page first.")
        return
    fine_tuner = GPTFineTuner(api_key=api_key)

    train_file = st.file_uploader("上传训练文件 (JSONL)", type=["jsonl"], key="train_file")
    val_file = st.file_uploader("上传验证文件 (可选, JSONL)", type=["jsonl"], key="val_file")

    if st.button("上传训练文件到OpenAI"):
        if not train_file:
            st.error("请上传训练文件！")
            logger.error("用户未上传训练文件，点击了上传训练文件按钮。")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp_train:
                tmp_train.write(train_file.read())
                tmp_train_path = tmp_train.name
            logger.info(f"准备上传训练文件: {tmp_train_path}")
            file_id = fine_tuner.upload_file(file_path=tmp_train_path)
            if file_id:
                st.success(f"训练文件已上传，file_id: {file_id}")
                local_data["train_file_id"] = file_id
                save_local_data(local_data)
                logger.info(f"训练文件上传成功，file_id: {file_id}")
            else:
                st.error("训练文件上传失败，请检查日志。")
                logger.error("训练文件上传失败。")

    if st.button("上传验证文件到OpenAI"):
        if not val_file:
            st.error("请上传验证文件！")
            logger.error("用户未上传验证文件，点击了上传验证文件按钮。")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp_val:
                tmp_val.write(val_file.read())
                tmp_val_path = tmp_val.name
            logger.info(f"准备上传验证文件: {tmp_val_path}")
            file_id = fine_tuner.upload_file(file_path=tmp_val_path)
            if file_id:
                st.success(f"验证文件已上传，file_id: {file_id}")
                local_data["validation_file_id"] = file_id
                save_local_data(local_data)
                logger.info(f"验证文件上传成功，file_id: {file_id}")
            else:
                st.error("验证文件上传失败，请检查日志。")
                logger.error("验证文件上传失败。")

    # Show current file IDs
    st.markdown("---")
    st.markdown("#### 当前已保存的 file_id:")
    st.markdown(f"- 训练文件: `{local_data.get('train_file_id', '')}`")
    st.markdown(f"- 验证文件: `{local_data.get('validation_file_id', '')}`")

def create_fine_tune_job_page():
    st.title("Create Fine-tuning Job")
    st.markdown("""
    ### Step 1: Select Model and Files
    - Choose a base model for fine-tuning (default: latest gpt-4.1 mini)
    - Edit file IDs if needed (auto-filled from uploads)
    """)
    global local_data
    if local_data is None:
        local_data = load_local_data()
    api_key = local_data.get("api_key", "")
    if not api_key:
        st.warning("Please set your OpenAI API Key on the Home page first.")
        return
    # Use cached models from local_data.json
    available_models = local_data.get("available_models", [])
    def friendly_name(model_id):
        if "o4-mini" in model_id:
            return "o4 Mini"
        elif "o3-pro" in model_id or "gpt-4o-3-pro" in model_id:
            return "o3 Pro"
        elif "o3-mini" in model_id or "gpt-4o-3-mini" in model_id:
            return "o3 Mini"
        elif "o3" in model_id and "pro" not in model_id and "mini" not in model_id:
            return "o3"
        elif "gpt-4.1-nano" in model_id:
            return "GPT 4.1 Nano"
        elif "gpt-4.1-mini" in model_id:
            return "GPT 4.1 Mini"
        elif "gpt-4.1" in model_id or "gpt-4-1106" in model_id:
            return "GPT 4.1"
        elif "gpt-4o" in model_id and "mini" not in model_id:
            return "GPT 4o"
        elif "gpt-4o-mini" in model_id:
            return "GPT 4o Mini"
        elif "chatgpt-4o" in model_id:
            return "ChatGPT 4o"
        else:
            return model_id
    if not available_models:
        # fallback static list
        fallback_ids = [
            "gpt-4.1-mini-2025-04-14",
            "o4-mini-2025-04-16",
            "o3-2025-04-16",
            "o3-pro-2025-06-10",
            "o3-mini-2025-01-31",
            "gpt-4.1-2025-04-14",
            "gpt-4o-2024-11-20",
            "gpt-4.1-nano-2025-04-14",
            "gpt-4o-mini-2024-07-18",
        ]
        available_models = [
            {"label": f"{friendly_name(mid)} ({mid})", "value": mid} for mid in fallback_ids
        ]
        logger.warning("未能从缓存或OpenAI获取模型列表，使用静态列表。")
    # Default to latest GPT 4.1 mini if available
    default_model_idx = 0
    for i, m in enumerate(available_models):
        if "gpt-4.1-mini" in m["value"]:
            default_model_idx = i
            break
    model_labels = [m["label"] for m in available_models]
    model_label = st.selectbox("选择基础模型 (Base Model)", model_labels, index=default_model_idx)
    model_value = next((m["value"] for m in available_models if m["label"] == model_label), available_models[default_model_idx]["value"])
    logger.info(f"用户选择模型: {model_label} ({model_value})")

    st.markdown("---")
    st.markdown("### Step 2: Set File IDs")
    train_file_id = st.text_input("训练文件 file_id", value=local_data.get("train_file_id", "") or "")
    val_file_id = st.text_input("验证文件 file_id (optional but recommended)", value=local_data.get("validation_file_id", "") or "")

    st.markdown("---")
    st.markdown("### Step 3: Method & Hyperparameters")
    method = st.selectbox("方法 (Method)", ["supervised"], index=0)
    st.caption("目前仅支持 supervised fine-tuning")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_epochs = st.text_input("n_epochs", value="auto", help="训练轮数 (auto = let OpenAI decide)")
    with col2:
        batch_size = st.text_input("batch_size", value="auto", help="每批次样本数 (auto = let OpenAI decide)")
    with col3:
        learning_rate_multiplier = st.text_input("learning_rate_multiplier", value="auto", help="学习率乘数 (auto = let OpenAI decide)")

    st.markdown("---")
    st.markdown("### Step 4: Create Fine-tuning Job")
    if st.button("创建微调任务 (Create Fine-tune Job)"):
        # Prepare hyperparameters
        def parse_int_auto(val):
            return None if val.strip().lower() == "auto" else int(val)
        def parse_float_auto(val):
            return None if val.strip().lower() == "auto" else float(val)
        hp_n_epochs = parse_int_auto(n_epochs)
        hp_batch_size = parse_int_auto(batch_size)
        hp_lr_mult = parse_float_auto(learning_rate_multiplier)
        logger.info(f"用户请求创建微调任务: model={model_value}, train_file_id={train_file_id}, val_file_id={val_file_id}, n_epochs={hp_n_epochs}, batch_size={hp_batch_size}, lr_mult={hp_lr_mult}")
        fine_tuner = GPTFineTuner(api_key=api_key) # Only instantiate fine_tuner when actually creating a job
        job_id = fine_tuner.create_fine_tune_job(
            training_file_id=train_file_id or "",
            validation_file_id=val_file_id if val_file_id else None,
            model=model_value,
            n_epochs=hp_n_epochs,
            batch_size=hp_batch_size,
            learning_rate_multiplier=hp_lr_mult,
        )
        if job_id:
            st.success(f"微调任务已创建，job_id: {job_id}")
            local_data["job_id"] = job_id
            save_local_data(local_data)
            logger.info(f"微调任务创建成功，job_id: {job_id}")
            # Optionally, monitor and fetch model_id
            with st.spinner("等待训练完成... (可在日志中查看进度)"):
                model_id = fine_tuner.monitor_training(job_id)
            if model_id:
                st.success(f"训练完成！模型ID: {model_id}")
                local_data["model_id"] = model_id
                save_local_data(local_data)
                logger.info(f"训练完成，模型ID: {model_id}")
            else:
                st.warning("训练未完成或失败，请检查日志。")
        else:
            st.error("微调任务创建失败，请检查日志。")
            logger.error("微调任务创建失败。")

def test_model_page():
    st.title("Test Model")
    st.markdown("""
    ### Test any model (manual, one prompt/input, one output)
    - Enter the model id, system prompt, user input, and temperature.
    - Click the button to get the model's response.
    """)
    global local_data
    if local_data is None:
        local_data = load_local_data()
    api_key = local_data.get("api_key", "")
    if not api_key:
        st.warning("Please set your OpenAI API Key on the Home page first.")
        return
    # Inputs
    model_id = st.text_input("Model ID", value=local_data.get("model_id", "") or "")
    system_prompt = st.text_area("System Prompt", value=local_data.get("system_prompt", "") or "", height=100)
    user_test_message = st.text_area("User Test Message/Input", value=local_data.get("user_test_message", "") or "", height=100)
    temperature = st.text_input("Temperature", value=str(local_data.get("temperature", "0.0")))
    if st.button("Run Test (Ask Model)"):
        try:
            temp_val = float(temperature)
        except Exception:
            temp_val = 0.0
        from gpt_fine_tuner import GPTFineTuner
        fine_tuner = GPTFineTuner(api_key=api_key)
        logger.info(f"测试模型: model_id={model_id}, temperature={temp_val}")
        import time
        import threading
        import queue
        progress = st.progress(0, text="Waiting for model response...")
        result_queue = queue.Queue()
        def call_model():
            res = fine_tuner.ask_model(
                model_id=model_id or "",
                system_prompt=system_prompt or "",
                test_message=user_test_message or "",
                temperature=temp_val
            )
            result_queue.put(res)
        thread = threading.Thread(target=call_model)
        thread.start()
        progress_val = 0
        start_time = time.time()
        max_wait = 60  # seconds
        while thread.is_alive() and (time.time() - start_time < max_wait):
            progress_val = (progress_val + 3) % 100
            progress.progress(progress_val, text="Waiting for model response...")
            time.sleep(0.1)
        thread.join(timeout=1)
        progress.progress(100, text="Model response received!")
        time.sleep(0.2)
        progress.empty()
        result = None
        if not result_queue.empty():
            result = result_queue.get()
        if result is not None:
            # Show as styled markdown with background and color
            import html
            st.markdown(
                f'<div style="background: #eaf4fb; color: #222; border-radius: 8px; padding: 1em; margin-bottom: 1em; font-size: 1.1em; white-space: pre-wrap; word-break: break-word;">{html.escape(result)}</div>',
                unsafe_allow_html=True
            )
            local_data["model_id"] = model_id
            local_data["system_prompt"] = system_prompt
            local_data["user_test_message"] = user_test_message
            local_data["temperature"] = temperature
            local_data["model_response"] = result
            save_local_data(local_data)
            logger.info(f"模型测试结果已保存。")
        else:
            st.error("模型未返回结果，请检查日志和参数。")
            logger.error("模型未返回结果。")

# --- PAGE ROUTING ---
def main():
    st.markdown(
        """
        <style>
        .sidebar-buttons button {
            width: 100%;
            margin-bottom: 8px;
            border: none;
            background: #f0f2f6;
            color: #333;
            padding: 0.75em 1em;
            border-radius: 0.5em;
            font-size: 1.1em;
            transition: background 0.2s, color 0.2s;
            cursor: pointer;
            text-align: left;
        }
        .sidebar-buttons button.selected, .sidebar-buttons button:hover {
            background: #4f8bf9;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Home",
        "Convert Excel to JSONL",
        "Upload Files to OpenAI",
        "Create Fine-Tuning Job",
        "Test Model"
    ]
    if "page" not in st.session_state:
        st.session_state.page = pages[0]

    with st.sidebar:
        st.markdown("""
        <div style='font-size:2em; font-weight:bold; margin-bottom:0.5em;'>Navigation</div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='sidebar-buttons'>", unsafe_allow_html=True)
        for p in pages:
            if st.button(p, key=p, help=p, use_container_width=True):
                st.session_state.page = p
                logger.info(f"用户切换到页面: {p}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Highlight the selected page button
    st.markdown(
        f"""
        <style>
        .element-container button[title=\"{st.session_state.page}\"] {{
            background: #4f8bf9 !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render the selected page
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Convert Excel to JSONL":
        excel_to_jsonl_page()
    elif st.session_state.page == "Upload Files to OpenAI":
        upload_files_page()
    elif st.session_state.page == "Create Fine-Tuning Job":
        create_fine_tune_job_page()
    elif st.session_state.page == "Test Model":
        test_model_page()
    else:
        st.info("This page is coming soon!")
        logger.info(f"用户访问了未实现页面: {st.session_state.page}")

if __name__ == "__main__":
    main()
