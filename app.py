"""
app.py
-------
Streamlit Web Interface for:
"Chatbot using LLM with Prompt Engineering"

Academic practical project demonstrating:
- Multi-task NLP: Question Answering, Text Summarization, Text Classification.
- Prompt Engineering Strategies: Zero-shot, Few-shot, Role prompting.
- Direct Groq Chat Completions API integration via standard HTTP requests.
- Response Quality Score (RQS) mathematical evaluation.
- Repeatability / Consistency testing.
- Test dataset benchmark validation.
- Evaluation history persistence via Pandas CSV.
"""

import os
import streamlit as st
import pandas as pd

# Load internal modules
from prompts.prompt_templates import (
    build_qa_prompt,
    build_summarization_prompt,
    build_classification_prompt,
    format_messages_for_display
)
from tasks.chatbot_tasks import (
    get_api_credentials,
    run_question_answering,
    run_text_summarization,
    run_text_classification,
    call_groq_api
)
from evaluation.evaluate import (
    calculate_rqs,
    save_evaluation_result,
    load_evaluation_history,
    clear_evaluation_history,
    extract_sentiment_label,
    calculate_classification_accuracy
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chatbot using LLM with Prompt Engineering",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Sidebar: Configuration & Academic Info
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ System Configuration")

api_key, model_name = get_api_credentials()

# Allow user to supply or override API key directly in session if not in .env
if not api_key:
    st.sidebar.warning("⚠️ No API Key found in `.env`")
    user_key_input = st.sidebar.text_input(
        "Enter Groq API Key:",
        type="password",
        help="Get a free key at https://console.groq.com/keys"
    )
    if user_key_input:
        os.environ["GROQ_API_KEY"] = user_key_input.strip()
        api_key = user_key_input.strip()
        st.sidebar.success("✅ API Key active for this session!")
else:
    st.sidebar.success("✅ Groq API Key detected")

st.sidebar.markdown(f"**Model:** `{model_name}`")
st.sidebar.markdown("**Backend:** Direct HTTP (`requests`)")

st.sidebar.divider()
st.sidebar.markdown("### 🎓 Academic Practical Details")
st.sidebar.info(
    "**Project:** Chatbot using LLM with Prompt Engineering\n\n"
    "**Core NLP Tasks:**\n"
    "- Question Answering\n"
    "- Text Summarization\n"
    "- Text Classification\n\n"
    "**Prompt Strategies:**\n"
    "- Zero-shot\n"
    "- Few-shot\n"
    "- Role Prompting\n\n"
    "**Evaluation:**\n"
    "- RQS = (Acc + Rel + Cons + Flu) / 4"
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🤖 Chatbot using LLM with Prompt Engineering")
st.caption(
    "An academic NLP practical application demonstrating prompt engineering strategies, "
    "LLM API integration via Groq, and mathematical Response Quality Score (RQS) evaluation."
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_consistency, tab_benchmark, tab_history = st.tabs([
    "💬 Interactive Chatbot",
    "🔄 Consistency Testing",
    "🧪 Dataset Benchmark",
    "📊 Evaluation History"
])


# =============================================================================
# TAB 1: INTERACTIVE CHATBOT
# =============================================================================
with tab_chat:
    st.subheader("1. Task & Prompt Strategy Selection")
    
    col_task, col_strat = st.columns(2)
    with col_task:
        selected_task = st.selectbox(
            "Select NLP Task:",
            ["Question Answering", "Text Summarization", "Text Classification"],
            help="Choose the NLP task you want the LLM to perform."
        )

    with col_strat:
        selected_strategy = st.selectbox(
            "Select Prompt Engineering Strategy:",
            ["Zero-shot Prompting", "Few-shot Prompting", "Role Prompting"],
            help="Choose the prompt construction technique."
        )

    # Dynamic placeholder and sample prompts based on selected task
    if selected_task == "Question Answering":
        placeholder_text = "e.g., What is the difference between Supervised and Unsupervised Learning?"
        sample_options = [
            "What is Artificial Intelligence?",
            "How does Natural Language Processing work?",
            "What is the role of activation functions in neural networks?"
        ]
    elif selected_task == "Text Summarization":
        placeholder_text = "e.g., Paste a paragraph or article here to generate a concise summary..."
        sample_options = [
            "Artificial intelligence is transforming healthcare by improving diagnostic accuracy, accelerating drug discovery, and optimizing patient care. Machine learning models analyze medical scans to detect diseases early, while robotic assistants aid surgeons during complex procedures.",
            "Renewable energy sources like solar and wind power are critical for curbing greenhouse gas emissions. As technological advancements make solar panels and wind turbines more cost-effective, adoption has accelerated worldwide, reducing dependence on fossil fuels."
        ]
    else:  # Text Classification
        placeholder_text = "e.g., Enter a customer review or sentence to classify its sentiment..."
        sample_options = [
            "The customer service was exceptional and my issue was resolved within five minutes! Absolutely love this company.",
            "The battery life is terrible and the screen stopped working on the second day. Total waste of money.",
            "The order was placed on Thursday and delivered on Saturday in standard cardboard packaging."
        ]

    st.subheader("2. Input Text")
    
    # Quick sample loader
    selected_sample = st.selectbox("💡 Or load a quick sample input:", ["(Custom input)"] + sample_options)
    initial_value = "" if selected_sample == "(Custom input)" else selected_sample

    user_input = st.text_area(
        "Enter your input below:",
        value=initial_value,
        placeholder=placeholder_text,
        height=130
    )

    # Generate Response Button
    if st.button("🚀 Generate Response", type="primary", width="stretch"):
        if not user_input.strip():
            st.warning("⚠️ Please provide input text before generating a response.")
        else:
            with st.spinner("Calling Groq LLM API..."):
                try:
                    if selected_task == "Question Answering":
                        response_text = run_question_answering(user_input, selected_strategy)
                        prompt_messages = build_qa_prompt(user_input, selected_strategy)
                    elif selected_task == "Text Summarization":
                        response_text = run_text_summarization(user_input, selected_strategy)
                        prompt_messages = build_summarization_prompt(user_input, selected_strategy)
                    else:
                        response_text = run_text_classification(user_input, selected_strategy)
                        prompt_messages = build_classification_prompt(user_input, selected_strategy)

                    # Store in session state for evaluation scoring
                    st.session_state["last_response"] = response_text
                    st.session_state["last_input"] = user_input
                    st.session_state["last_task"] = selected_task
                    st.session_state["last_strategy"] = selected_strategy
                    st.session_state["last_prompt_messages"] = prompt_messages

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Display Generated Response
    if "last_response" in st.session_state and st.session_state["last_response"]:
        st.subheader("3. Generated LLM Response")
        st.success(f"**Task:** {st.session_state['last_task']} | **Strategy:** {st.session_state['last_strategy']}")
        st.markdown(st.session_state["last_response"])

        # Collapsible prompt inspector for academic viva demonstration
        with st.expander("🔍 Inspect Exact Formatted Prompt Sent to Groq (Academic Demo)"):
            st.code(format_messages_for_display(st.session_state["last_prompt_messages"]), language="text")

        st.divider()

        # ---------------------------------------------------------------------
        # Evaluation Section
        # ---------------------------------------------------------------------
        st.subheader("4. Response Quality Score (RQS) Manual Evaluation")
        st.caption(
            "Rate each parameter from 1 (Poor) to 5 (Excellent). "
            "Note: These criteria are rated by the human evaluator according to the practical examination rubric."
        )

        col1, col2 = st.columns(2)
        with col1:
            eval_acc = st.slider("Accuracy (Is the information factual and correct?)", 1, 5, 5)
            eval_rel = st.slider("Relevance (Does it answer the user's input directly?)", 1, 5, 5)
        with col2:
            eval_cons = st.slider("Consistency (Is it stable with expected domain logic?)", 1, 5, 4)
            eval_flue = st.slider("Fluency (Is the grammar natural and readable?)", 1, 5, 5)

        # Compute Mathematical RQS
        rqs, rqs_pct = calculate_rqs(eval_acc, eval_rel, eval_cons, eval_flue)

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Response Quality Score (RQS)", f"{rqs} / 5.0")
        metric_col2.metric("RQS Percentage", f"{rqs_pct}%")

        st.latex(r"\text{RQS} = \frac{\text{Accuracy} + \text{Relevance} + \text{Consistency} + \text{Fluency}}{4}")

        if st.button("💾 Save Evaluation to Results CSV", width="stretch"):
            save_evaluation_result(
                task=st.session_state["last_task"],
                prompt_strategy=st.session_state["last_strategy"],
                user_input=st.session_state["last_input"],
                response_text=st.session_state["last_response"],
                accuracy=eval_acc,
                relevance=eval_rel,
                consistency=eval_cons,
                fluency=eval_flue
            )
            st.success("✅ Evaluation successfully saved to `evaluation/results.csv`!")


# =============================================================================
# TAB 2: CONSISTENCY TESTING
# =============================================================================
with tab_consistency:
    st.subheader("🔄 Repeatability & Consistency Evaluation")
    st.write(
        "Evaluate whether the LLM delivers consistent, stable responses when the same input "
        "and prompt strategy are submitted repeatedly."
    )

    c_task = st.selectbox("Consistency Task:", ["Question Answering", "Text Summarization", "Text Classification"], key="c_task")
    c_strat = st.selectbox("Consistency Strategy:", ["Zero-shot Prompting", "Few-shot Prompting", "Role Prompting"], key="c_strat")

    default_test_input = (
        "The food arrived cold and forty minutes late. Very disappointed."
        if c_task == "Text Classification"
        else "What is Machine Learning?"
    )
    c_input = st.text_area("Input for Consistency Test:", value=default_test_input, height=100)
    iterations = st.slider("Number of Test Iterations:", min_value=2, max_value=4, value=2)

    if st.button("▶️ Run Consistency Test", type="primary"):
        if not c_input.strip():
            st.warning("Please provide input text.")
        else:
            responses = []
            with st.spinner(f"Generating {iterations} independent responses via Groq API..."):
                try:
                    for i in range(iterations):
                        if c_task == "Question Answering":
                            res = run_question_answering(c_input, c_strat)
                        elif c_task == "Text Summarization":
                            res = run_text_summarization(c_input, c_strat)
                        else:
                            res = run_text_classification(c_input, c_strat)
                        responses.append(res)
                    
                    st.success(f"Generated {iterations} responses successfully!")

                    # Display responses side-by-side or stacked
                    cols = st.columns(iterations)
                    labels = []
                    for idx, col in enumerate(cols):
                        with col:
                            st.markdown(f"**Iteration #{idx + 1}:**")
                            st.info(responses[idx])
                            if c_task == "Text Classification":
                                label = extract_sentiment_label(responses[idx])
                                labels.append(label)
                                st.caption(f"Detected Label: **{label}**")

                    # Consistency analysis summary
                    st.divider()
                    st.markdown("### 📋 Consistency Analysis Summary")
                    if c_task == "Text Classification":
                        all_identical = len(set(labels)) == 1
                        if all_identical:
                            st.success(f"✅ **Consistent Output**: All {iterations} iterations predicted the label `{labels[0]}`.")
                        else:
                            st.warning(f"⚠️ **Inconsistent Labels**: Predictions varied across runs: {labels}")
                    else:
                        st.info(
                            "ℹ️ **Evaluation Note**: For generative tasks (QA & Summarization), LLMs exhibit natural "
                            "phrasing variations while retaining core semantic content. Compare the summaries/answers "
                            "above to verify semantic consistency."
                        )

                except Exception as e:
                    st.error(f"❌ Consistency test failed: {str(e)}")


# =============================================================================
# TAB 3: DATASET BENCHMARKING
# =============================================================================
with tab_benchmark:
    st.subheader("🧪 Test Dataset & Academic Benchmarking")
    st.write(
        "Evaluate the chatbot against the standard practical dataset stored in `data/test_dataset.csv`. "
        "For classification, automated Accuracy % is computed. For QA and Summarization, outputs are compared side-by-side."
    )

    dataset_path = os.path.join(os.path.dirname(__file__), "data", "test_dataset.csv")

    if not os.path.exists(dataset_path):
        st.error(f"Dataset file not found at `{dataset_path}`.")
    else:
        df_test = pd.read_csv(dataset_path)
        st.markdown("**Test Dataset Preview:**")
        st.dataframe(df_test, width="stretch")

        benchmark_task = st.selectbox(
            "Filter benchmark by Task:",
            ["All Tasks", "Text Classification", "Question Answering", "Text Summarization"]
        )
        bench_strat = st.selectbox(
            "Benchmark Strategy:",
            ["Few-shot Prompting", "Zero-shot Prompting", "Role Prompting"],
            key="bench_strat"
        )

        if st.button("🚀 Run Dataset Benchmark Test", type="primary"):
            if benchmark_task == "All Tasks":
                filtered_df = df_test
            else:
                filtered_df = df_test[df_test["task"] == benchmark_task]

            results = []
            progress_bar = st.progress(0)
            total_items = len(filtered_df)

            with st.spinner("Running test cases through Groq API..."):
                try:
                    for idx, (_, row) in enumerate(filtered_df.iterrows()):
                        t_task = row["task"]
                        t_input = row["input_text"]
                        t_expected = row["expected_output"]

                        if t_task == "Question Answering":
                            actual = run_question_answering(t_input, bench_strat)
                            test_status = "Manual Review"
                        elif t_task == "Text Summarization":
                            actual = run_text_summarization(t_input, bench_strat)
                            test_status = "Manual Review"
                        else:  # Text Classification
                            actual = run_text_classification(t_input, bench_strat)
                            predicted_label = extract_sentiment_label(actual)
                            is_pass = predicted_label.lower() == str(t_expected).strip().lower()
                            test_status = "PASS" if is_pass else "FAIL"

                        results.append({
                            "Task": t_task,
                            "Input": t_input,
                            "Expected Output": t_expected,
                            "Actual Output": actual,
                            "Result": test_status
                        })
                        progress_bar.progress((idx + 1) / total_items)

                    results_df = pd.DataFrame(results)
                    st.divider()
                    st.subheader("📈 Benchmark Results")
                    st.dataframe(results_df, width="stretch")

                    # Calculate quantitative accuracy for classification items
                    class_items = [r for r in results if r["Task"] == "Text Classification"]
                    if class_items:
                        preds = [extract_sentiment_label(r["Actual Output"]) for r in class_items]
                        truths = [r["Expected Output"] for r in class_items]
                        acc_pct, correct, total = calculate_classification_accuracy(preds, truths)

                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Classification Accuracy", f"{acc_pct}%")
                        col_m2.metric("Passed Test Cases", f"{correct} / {total}")
                        col_m3.metric("Failed Cases", f"{total - correct}")

                        st.latex(r"\text{Accuracy} = \left(\frac{\text{Correct Predictions}}{\text{Total Predictions}}\right) \times 100")

                except Exception as e:
                    st.error(f"❌ Benchmark failed: {str(e)}")


# =============================================================================
# TAB 4: EVALUATION HISTORY
# =============================================================================
with tab_history:
    st.subheader("📊 Saved Evaluation History (CSV)")
    st.write("Browse, inspect, or download historical evaluation records stored in `evaluation/results.csv`.")

    df_history = load_evaluation_history()

    if df_history.empty:
        st.info("No evaluation records saved yet. Perform a query in the Chatbot tab and click 'Save Evaluation'.")
    else:
        st.dataframe(df_history, width="stretch")

        # Summary statistics
        if "RQS" in df_history.columns:
            try:
                numeric_rqs = pd.to_numeric(df_history["RQS"], errors="coerce")
                avg_rqs = numeric_rqs.mean()
                total_records = len(df_history)

                stat_col1, stat_col2 = st.columns(2)
                stat_col1.metric("Total Evaluations Saved", total_records)
                stat_col2.metric("Average RQS Score", f"{round(avg_rqs, 2)} / 5.0")
            except Exception:
                pass

        col_dl, col_clr = st.columns([3, 1])
        with col_dl:
            csv_data = df_history.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="evaluation_results.csv",
                mime="text/csv",
                width="stretch"
            )

        with col_clr:
            if st.button("🗑️ Clear History", width="stretch"):
                clear_evaluation_history()
                st.warning("Evaluation history cleared.")
                st.rerun()
