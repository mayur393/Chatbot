"""
evaluation/evaluate.py
-----------------------
This module implements the mathematical evaluation framework and history
tracking for the chatbot project.

Mathematical Model:
- Response Quality Score (RQS):
      RQS = (Accuracy + Relevance + Consistency + Fluency) / 4
      RQS Percentage = (RQS / 5) * 100
      
Criteria (Rated 1 to 5):
- Accuracy: Does the response contain correct, truthful information?
- Relevance: Does the response directly address the question or input?
- Consistency: Does the response remain stable when re-submitted?
- Fluency: Is the response grammatically correct, natural, and readable?

Evaluation History:
- Stored in CSV format at 'evaluation/results.csv' using Pandas.
- Automatically initialized if missing.
"""

import os
from datetime import datetime
import pandas as pd

# Filepath for saving evaluation history
EVALUATION_FILE_PATH = os.path.join(os.path.dirname(__file__), "results.csv")

# Standard CSV columns
COLUMNS = [
    "Timestamp",
    "Task",
    "Prompt Strategy",
    "Input",
    "Generated Response",
    "Accuracy",
    "Relevance",
    "Consistency",
    "Fluency",
    "RQS",
    "RQS Percentage"
]


def calculate_rqs(accuracy: float, relevance: float, consistency: float, fluency: float) -> tuple:
    """
    Calculates the Response Quality Score (RQS) and its percentage.
    
    Formula:
        RQS = (Accuracy + Relevance + Consistency + Fluency) / 4
        RQS Percentage = (RQS / 5) * 100
        
    Parameters:
        accuracy (float): Score from 1 to 5.
        relevance (float): Score from 1 to 5.
        consistency (float): Score from 1 to 5.
        fluency (float): Score from 1 to 5.
        
    Returns:
        tuple: (rqs_score, rqs_percentage) rounded to 2 decimal places.
    """
    rqs = (accuracy + relevance + consistency + fluency) / 4.0
    rqs_percentage = (rqs / 5.0) * 100.0
    return round(rqs, 2), round(rqs_percentage, 2)


def save_evaluation_result(
    task: str,
    prompt_strategy: str,
    user_input: str,
    response_text: str,
    accuracy: int,
    relevance: int,
    consistency: int,
    fluency: int
) -> tuple:
    """
    Calculates RQS and saves the evaluation record to 'evaluation/results.csv'.
    
    Returns:
        tuple: (rqs, rqs_pct)
    """
    rqs, rqs_pct = calculate_rqs(accuracy, relevance, consistency, fluency)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = {
        "Timestamp": timestamp,
        "Task": task,
        "Prompt Strategy": prompt_strategy,
        "Input": user_input.replace("\n", " ").strip(),
        "Generated Response": response_text.replace("\n", " ").strip(),
        "Accuracy": accuracy,
        "Relevance": relevance,
        "Consistency": consistency,
        "Fluency": fluency,
        "RQS": rqs,
        "RQS Percentage": f"{rqs_pct}%"
    }

    # Load existing or create new DataFrame
    if os.path.exists(EVALUATION_FILE_PATH):
        try:
            df = pd.read_csv(EVALUATION_FILE_PATH)
            # Ensure correct columns exist
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = None
        except Exception:
            df = pd.DataFrame(columns=COLUMNS)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    # Append new row
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)

    # Save to CSV
    df.to_csv(EVALUATION_FILE_PATH, index=False)
    return rqs, rqs_pct


def load_evaluation_history() -> pd.DataFrame:
    """
    Loads saved evaluation records from 'evaluation/results.csv'.
    
    Returns:
        pd.DataFrame: DataFrame containing historical evaluations.
    """
    if not os.path.exists(EVALUATION_FILE_PATH):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(EVALUATION_FILE_PATH)
        return df
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


def clear_evaluation_history() -> bool:
    """
    Resets the evaluation history file.
    
    Returns:
        bool: True if cleared successfully.
    """
    if os.path.exists(EVALUATION_FILE_PATH):
        try:
            os.remove(EVALUATION_FILE_PATH)
            return True
        except Exception:
            return False
    return True


# ========================================================
# Benchmark & Accuracy Evaluation Utilities
# ========================================================

def extract_sentiment_label(response_text: str) -> str:
    """
    Extracts the predicted sentiment label (Positive, Negative, or Neutral)
    from the LLM's response.
    
    Returns:
        str: 'Positive', 'Negative', 'Neutral', or 'Unknown'.
    """
    text_lower = response_text.lower()
    
    # Check for direct label patterns like "Label: Positive"
    if "label: positive" in text_lower or "positive" in text_lower.split():
        return "Positive"
    elif "label: negative" in text_lower or "negative" in text_lower.split():
        return "Negative"
    elif "label: neutral" in text_lower or "neutral" in text_lower.split():
        return "Neutral"
    
    # Fallback search
    if "positive" in text_lower:
        return "Positive"
    elif "negative" in text_lower:
        return "Negative"
    elif "neutral" in text_lower:
        return "Neutral"
    
    return "Unknown"


def calculate_classification_accuracy(predictions: list, ground_truths: list) -> tuple:
    """
    Calculates classification accuracy percentage.
    
    Formula:
        Accuracy = (Correct Predictions / Total Predictions) * 100
        
    Returns:
        tuple: (accuracy_percentage, correct_count, total_count)
    """
    total = len(ground_truths)
    if total == 0:
        return 0.0, 0, 0

    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        if pred.strip().lower() == truth.strip().lower():
            correct += 1

    accuracy_pct = (correct / total) * 100.0
    return round(accuracy_pct, 2), correct, total
