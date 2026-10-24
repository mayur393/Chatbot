"""
tasks/chatbot_tasks.py
-----------------------
This module handles communication with the Groq LLM API and coordinates
the three core NLP tasks:
1. Question Answering
2. Text Summarization
3. Text Classification

Key Features:
- Direct HTTP requests to Groq using the 'requests' library (No LangChain/LangGraph).
- Loads API credentials from .env via 'python-dotenv'.
- Clear error handling for missing keys, network timeouts, and API errors.
- Simple, beginner-friendly functions.
"""

import os
import requests
from dotenv import load_dotenv

# Import the prompt template generators
from prompts.prompt_templates import (
    build_qa_prompt,
    build_summarization_prompt,
    build_classification_prompt
)

# Load environment variables from .env file in the project root
load_dotenv()

# Constants
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 30


def get_api_credentials():
    """
    Retrieves the Groq API Key and Model name from environment variables.
    
    Returns:
        tuple: (api_key, model_name)
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL
    return api_key, model


def call_groq_api(messages: list, temperature: float = 0.7) -> str:
    """
    Sends a direct HTTP POST request to the Groq Chat Completions API.
    
    Parameters:
        messages (list): List of message dicts (role and content).
        temperature (float): Sampling temperature (0.0 = deterministic, 1.0 = creative).
        
    Returns:
        str: The generated assistant response text.
        
    Raises:
        ValueError: If the API key is missing or invalid.
        ConnectionError: If network connection fails.
        RuntimeError: If Groq API responds with an error code.
    """
    api_key, model = get_api_credentials()

    # 1. Validate API Key existence
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "Groq API Key is missing or not configured!\n\n"
            "Please create a '.env' file in the project folder with your API key:\n"
            "GROQ_API_KEY=gsk_your_actual_key_here\n\n"
            "You can get a free key from: https://console.groq.com/keys"
        )

    # 2. Prepare HTTP headers and payload
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024
    }

    # 3. Send the HTTP POST request
    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout:
        raise ConnectionError(
            f"The request to Groq API timed out after {REQUEST_TIMEOUT_SECONDS} seconds. "
            "Please check your internet connection and try again."
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Could not connect to the Groq API server. "
            "Please verify that your internet connection is active."
        )
    except Exception as e:
        raise ConnectionError(f"Unexpected connection error occurred: {str(e)}")

    # 4. Check HTTP status code
    if response.status_code == 401:
        raise ValueError(
            "Authentication failed (HTTP 401): Your Groq API Key appears to be invalid. "
            "Please check the GROQ_API_KEY in your .env file."
        )
    elif response.status_code == 429:
        raise RuntimeError(
            "Rate limit reached (HTTP 429): Too many requests sent to Groq. "
            "Please wait a moment and try again."
        )
    elif response.status_code != 200:
        # Extract error message if provided by Groq
        try:
            error_data = response.json()
            err_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            err_msg = response.text
        raise RuntimeError(f"Groq API Error (HTTP {response.status_code}): {err_msg}")

    # 5. Parse JSON response and extract generated text
    try:
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Groq API returned an empty choice list.")
        generated_text = choices[0].get("message", {}).get("content", "").strip()
        return generated_text
    except Exception as e:
        raise RuntimeError(f"Failed to parse Groq response JSON: {str(e)}")


# ========================================================
# High-Level NLP Task Handlers
# ========================================================

def run_question_answering(question: str, strategy: str) -> str:
    """
    Executes the Question Answering task using the selected prompt strategy.
    
    Parameters:
        question (str): User question.
        strategy (str): 'Zero-shot Prompting', 'Few-shot Prompting', or 'Role Prompting'.
        
    Returns:
        str: The LLM generated answer.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty. Please enter a valid question.")

    messages = build_qa_prompt(question, strategy)
    return call_groq_api(messages, temperature=0.5)


def run_text_summarization(text: str, strategy: str) -> str:
    """
    Executes the Text Summarization task using the selected prompt strategy.
    
    Parameters:
        text (str): Input paragraph or article.
        strategy (str): 'Zero-shot Prompting', 'Few-shot Prompting', or 'Role Prompting'.
        
    Returns:
        str: The LLM generated summary.
    """
    if not text.strip():
        raise ValueError("Text input cannot be empty. Please enter a paragraph to summarize.")

    messages = build_summarization_prompt(text, strategy)
    return call_groq_api(messages, temperature=0.3)


def run_text_classification(text: str, strategy: str) -> str:
    """
    Executes the Text Classification (Sentiment Analysis) task using the selected prompt strategy.
    
    Parameters:
        text (str): Input text or customer review.
        strategy (str): 'Zero-shot Prompting', 'Few-shot Prompting', or 'Role Prompting'.
        
    Returns:
        str: The LLM generated classification label and explanation.
    """
    if not text.strip():
        raise ValueError("Text input cannot be empty. Please enter text to classify.")

    messages = build_classification_prompt(text, strategy)
    # Low temperature for classification ensures consistent, deterministic labeling
    return call_groq_api(messages, temperature=0.1)
