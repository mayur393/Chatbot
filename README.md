# Chatbot using LLM with Prompt Engineering

An academic Python practical project demonstrating modern Natural Language Processing (NLP) workflows using Large Language Models (LLMs), prompt engineering strategies, direct HTTP-based API interaction, and quantitative response quality evaluation.

---

## 1. Project Aim

To develop an AI chatbot utilizing a Large Language Model (LLM) through direct API communication with Prompt Engineering techniques to perform:
1. **Question Answering**
2. **Text Summarization**
3. **Text Classification** (Sentiment Analysis)

The application evaluates generated responses across quality, relevance, accuracy, consistency, and fluency using an empirical mathematical model (Response Quality Score).

---

## 2. Project Objectives

- Implement direct, lightweight HTTP requests to the **Groq Chat Completions API** without complex or opaque frameworks (such as LangChain or LangGraph).
- Implement three core prompt engineering strategies: **Zero-shot Prompting**, **Few-shot Prompting**, and **Role Prompting**.
- Design an intuitive, beginner-friendly **Streamlit** user interface suitable for practical demonstrations and viva examinations.
- Formulate and compute the **Response Quality Score (RQS)** and its percentage.
- Implement repeatability/consistency testing and test dataset benchmarking (`data/test_dataset.csv`).
- Maintain persistent evaluation history using **Pandas** in CSV format (`evaluation/results.csv`).

---

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web-based user interface and interactive widgets |
| **Groq API** | High-speed cloud LLM inference (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) |
| **Requests** | Direct HTTP communication with Groq's REST API endpoint |
| **python-dotenv** | Secure loading of environment variables from `.env` |
| **Pandas** | Tabular data management, benchmarking, and CSV evaluation history |

> [!NOTE]
> This project intentionally avoids heavy frameworks like LangChain, LangGraph, or local vector databases to keep the codebase clear, transparent, and easy for beginners to explain during examinations.

---

## 4. Project Folder Structure

```
Chatbot/
│
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Minimal project dependencies
├── .env.example               # Example environment variable template
├── .gitignore                 # Git ignore rules (protects API keys & venv)
├── README.md                  # Complete documentation and viva guide
│
├── prompts/
│   ├── __init__.py            # Python package marker
│   └── prompt_templates.py    # Zero-shot, Few-shot, and Role prompt constructors
│
├── tasks/
│   ├── __init__.py            # Python package marker
│   └── chatbot_tasks.py       # Direct Groq API client and NLP task handlers
│
├── evaluation/
│   ├── __init__.py            # Python package marker
│   ├── evaluate.py            # RQS mathematical formula and CSV logger
│   └── results.csv            # Auto-generated evaluation history
│
└── data/
    └── test_dataset.csv       # Benchmark dataset with 9 sample practical test cases
```

---

## 5. Mathematical Evaluation Model

### Response Quality Score (RQS)

The evaluation system uses the practical mathematical model:

$$\text{RQS} = \frac{\text{Accuracy} + \text{Relevance} + \text{Consistency} + \text{Fluency}}{4}$$

$$\text{RQS Percentage} = \left(\frac{\text{RQS}}{5}\right) \times 100$$

Where each criterion is rated on a 1 to 5 scale by the human evaluator:
- **Accuracy (1–5):** Factual correctness and absence of hallucinations.
- **Relevance (1–5):** Direct alignment with the user's question or source text.
- **Consistency (1–5):** Stability of output logic across multiple submissions.
- **Fluency (1–5):** Readability, grammatical correctness, and natural expression.

### Classification Accuracy Formula

$$\text{Classification Accuracy} = \left(\frac{\text{Correct Predictions}}{\text{Total Predictions}}\right) \times 100$$

---

## 6. Prompt Engineering Techniques

### 1. Zero-shot Prompting
- **Concept:** The model is presented with task instructions and the raw input without prior examples.
- **Use Case:** Tests the foundational zero-shot generalization capabilities of the LLM.

### 2. Few-shot Prompting
- **Concept:** The prompt includes 1 to 3 curated input-output demonstrations before presenting the user's input.
- **Use Case:** Enforces strict output structure, formatting, and calibration (e.g., standardizing sentiment labels to `Positive`, `Negative`, `Neutral`).

### 3. Role Prompting
- **Concept:** The model is instructed to adopt an explicit persona or professional identity (e.g., "College Professor", "Executive Editor", "Customer Insights Specialist").
- **Use Case:** Tunes tone, vocabulary, depth, and explanatory clarity to fit specific audiences.

---

## 7. Installation & Setup Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- An internet connection.
- A free Groq API key (generate one at [console.groq.com/keys](https://console.groq.com/keys)).

---

### Step 1: Clone or Navigate to the Project Folder

```bash
# On Windows PowerShell
cd c:\Users\mayur303\Desktop\Chatbot
```

### Step 2: Create a Virtual Environment

**On Windows:**
```powershell
py -m venv venv
venv\Scripts\activate
```

**On Linux (Ubuntu / Debian):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Linux (Fedora / RHEL):**
```bash
# Fedora includes python3-virtualenv or venv
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables

1. Copy `.env.example` to create a `.env` file:
   - **Windows:** `copy .env.example .env`
   - **Linux:** `cp .env.example .env`
2. Open `.env` in a text editor and add your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

*(Alternatively, you can also paste your API key directly into the Streamlit sidebar during runtime!)*

---

### Step 5: Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will automatically open in your default browser at: `http://localhost:8501`.

---

## 8. Application Features & Demonstration

### Tab 1: 💬 Interactive Chatbot
- Select an NLP Task (**Question Answering**, **Text Summarization**, or **Text Classification**).
- Select a Prompt Strategy (**Zero-shot**, **Few-shot**, or **Role Prompting**).
- Enter custom text or select a ready-made academic sample input.
- Click **Generate Response** to send the request directly to Groq.
- Expand the **"Inspect Exact Formatted Prompt"** viewer to see the formatted prompt sent to the LLM.
- Rate the response using the interactive sliders (1–5) and view the computed RQS and percentage.
- Click **Save Evaluation to Results CSV** to log the record.

### Tab 2: 🔄 Consistency Testing
- Run 2 to 4 repeated inferences on the exact same input.
- Compare outputs side-by-side.
- For sentiment classification, verifies whether predicted labels are identical across runs.

### Tab 3: 🧪 Dataset Benchmark
- Loads `data/test_dataset.csv` containing standard practical test cases.
- Runs batch evaluation across selected prompt strategies.
- For classification, automatically calculates **Accuracy %** with **PASS/FAIL** status.
- For QA and summarization, presents side-by-side comparisons for manual inspection.

### Tab 4: 📊 Evaluation History
- Displays the persistent history table from `evaluation/results.csv`.
- Displays total saved evaluations and average RQS score.
- Allows downloading the updated CSV file or resetting the log.

---

## 9. Viva Voce Examination Questions & Answers

### Q1: What is a Large Language Model (LLM)?
> **Answer:** An LLM is a deep neural network (typically based on the Transformer architecture) trained on massive amounts of textual data to predict subsequent tokens. It learns statistical patterns, grammar, reasoning heuristics, and factual associations, allowing it to perform diverse tasks such as translation, summarization, classification, and question answering.

### Q2: What is Prompt Engineering?
> **Answer:** Prompt engineering is the practice of structuring, refining, and designing the input instructions and context provided to an LLM to elicit accurate, relevant, consistent, and safe outputs without retraining or fine-tuning the model's internal weights.

### Q3: What is the difference between Zero-shot and Few-shot prompting?
> **Answer:** 
> - In **Zero-shot prompting**, the model is given only the task instruction and the input text with zero examples. It relies entirely on its pre-trained knowledge.
> - In **Few-shot prompting**, the prompt includes a few demonstration pairs (input $\rightarrow$ output) before the target input. This calibrates the LLM to follow the desired output format and style precisely.

### Q4: What is Role Prompting?
> **Answer:** Role prompting instructs the LLM to assume a specific character, persona, or professional domain (e.g., "Educational Assistant", "Senior Research Scientist"). This steers the vocabulary, tone, perspective, and explanatory depth of the generated response.

### Q5: How does this project communicate with the Groq API?
> **Answer:** The project uses Python's standard `requests` library to send an HTTP POST request to Groq's OpenAI-compatible endpoint (`https://api.groq.com/openai/v1/chat/completions`). The payload includes the model name (`llama-3.3-70b-versatile`), a list of messages (system and user), and a sampling temperature. Groq processes the inference and returns a JSON response containing the generated text in `choices[0].message.content`.

### Q6: Why use direct requests instead of LangChain or LangGraph?
> **Answer:** Direct HTTP requests demonstrate how LLM APIs fundamentally operate at the networking and protocol level. Frameworks like LangChain add abstractions that can obscure the underlying mechanics, introduce unnecessary dependencies, and make it difficult for beginners to understand and debug.

### Q7: How does Question Answering work in this system?
> **Answer:** The system wraps the user's question with instructions specifying clarity, factual correctness, and conciseness. When sent to the LLM, the model uses its pre-trained knowledge base to synthesize an accurate, direct explanation.

### Q8: How does Text Summarization work?
> **Answer:** The summarization prompt instructs the model to extract the primary arguments, condense verbose sentences, and avoid hallucinating external details, generating a concise summary of the provided text.

### Q9: How is Text Classification performed without keyword matching?
> **Answer:** Rather than hardcoding keyword dictionaries (e.g., checking if "good" or "bad" exists), the raw text is sent to the LLM with a classification instruction and bounded output categories (`Positive`, `Negative`, `Neutral`). The LLM performs semantic analysis of tone, context, and emotion to predict the label and justify its decision.

### Q10: What is the Response Quality Score (RQS)?
> **Answer:** RQS is an empirical scoring metric calculated as the arithmetic mean of four key dimensions:
> $$\text{RQS} = \frac{\text{Accuracy} + \text{Relevance} + \text{Consistency} + \text{Fluency}}{4}$$
> Each criterion is scored on a scale of 1 to 5, resulting in an RQS out of 5.0 and an equivalent percentage.

### Q11: How is consistency evaluated?
> **Answer:** In Tab 2, the application submits the identical prompt multiple times to the Groq API under controlled parameters. For classification, predicted labels across runs are compared for identity. For generative tasks (QA and Summarization), responses are compared to confirm semantic coherence across natural linguistic variations.

---

## 10. Conclusion

This project successfully demonstrates an end-to-end, academic-grade AI Chatbot system. By combining direct API communication, diverse prompt engineering strategies, multi-task NLP execution, and empirical evaluation, it provides a solid foundation for understanding LLM applications without unnecessary architectural complexity.
