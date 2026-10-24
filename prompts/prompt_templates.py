"""
prompts/prompt_templates.py
----------------------------
This module defines prompt templates for the three NLP tasks:
1. Question Answering
2. Text Summarization
3. Text Classification (Sentiment Analysis)

It implements three Prompt Engineering strategies:
- Zero-shot Prompting: Providing task instructions and input with no examples.
- Few-shot Prompting: Providing clear input-output demonstrations before the target input.
- Role Prompting: Assigning a specific expert persona/role to guide the LLM's response style.

Each function constructs an OpenAI/Groq-compatible list of message dictionaries:
    [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
"""


def build_qa_prompt(question: str, strategy: str) -> list:
    """
    Constructs a prompt for Question Answering based on the selected strategy.
    
    Parameters:
        question (str): The question asked by the user.
        strategy (str): 'Zero-shot Prompting', 'Few-shot Prompting', or 'Role Prompting'.
        
    Returns:
        list: A list of message dictionaries for the Groq Chat API.
    """
    question = question.strip()

    if strategy == "Zero-shot Prompting":
        system_instruction = (
            "You are an AI assistant. Answer the user's question clearly, accurately, "
            "and concisely using simple language. Avoid unnecessary filler or speculation. "
            "If you do not know the answer with certainty, honestly state that you do not know."
        )
        user_content = f"Question: {question}\n\nAnswer:"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    elif strategy == "Few-shot Prompting":
        system_instruction = (
            "You are an AI assistant. Answer the user's question clearly and concisely in 2-3 sentences. "
            "Follow the pattern shown in the examples below."
        )
        # Demonstrating expected answer style with 2 short examples
        examples = (
            "Example 1:\n"
            "Question: What is Cloud Computing?\n"
            "Answer: Cloud computing is the on-demand delivery of computing services—including servers, "
            "storage, databases, and software—over the internet, allowing flexible resources without local hardware.\n\n"
            "Example 2:\n"
            "Question: What is an Algorithm?\n"
            "Answer: An algorithm is a step-by-step set of instructions or rules designed to solve a specific "
            "problem or perform a computational task efficiently.\n\n"
        )
        user_content = f"{examples}Now answer this question:\nQuestion: {question}\nAnswer:"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    elif strategy == "Role Prompting":
        system_instruction = (
            "You are a helpful educational assistant and experienced college professor. "
            "Explain complex concepts in a warm, beginner-friendly, and educational manner. "
            "Use clear analogies where helpful, keep explanations direct and structured, and avoid jargon."
        )
        user_content = (
            f"Dear Professor, please explain the following question to a beginner student:\n\n"
            f"Question: {question}"
        )
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    else:
        # Fallback to default zero-shot if strategy name does not match
        return [
            {"role": "system", "content": "Answer the question clearly and concisely."},
            {"role": "user", "content": f"Question: {question}"}
        ]


def build_summarization_prompt(text: str, strategy: str) -> list:
    """
    Constructs a prompt for Text Summarization based on the selected strategy.
    
    Parameters:
        text (str): The paragraph or article to summarize.
        strategy (str): 'Zero-shot Prompting', 'Few-shot Prompting', or 'Role Prompting'.
        
    Returns:
        list: A list of message dictionaries for the Groq Chat API.
    """
    text = text.strip()

    if strategy == "Zero-shot Prompting":
        system_instruction = (
            "You are an AI summarization assistant. Summarize the supplied text in 2-3 concise sentences. "
            "Preserve all core ideas, do not add external facts or opinions not present in the text, "
            "and use simple, clear language."
        )
        user_content = f"Text to summarize:\n{text}\n\nSummary:"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    elif strategy == "Few-shot Prompting":
        system_instruction = (
            "You are an AI summarization assistant. Produce a clear, factual, and compact summary "
            "following the format demonstrated in the example."
        )
        example = (
            "Example:\n"
            "Original Text: Renewable energy comes from natural sources that replenish themselves faster than they "
            "are consumed, such as sunlight and wind. Unlike fossil fuels, renewable sources generate far lower emissions, "
            "making them essential for combating global climate change and reducing air pollution.\n"
            "Summary: Renewable energy comes from naturally replenishing sources like solar and wind, producing significantly "
            "fewer emissions than fossil fuels and playing a critical role in addressing climate change.\n\n"
        )
        user_content = f"{example}Now summarize this text:\nOriginal Text: {text}\nSummary:"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    elif strategy == "Role Prompting":
        system_instruction = (
            "You are an expert text summarization assistant and executive editor. "
            "Your specialty is extracting high-value takeaways and summarizing complex documents into "
            "crystal-clear, publication-ready executive briefings without losing critical context."
        )
        user_content = (
            f"Please review and summarize the following passage for an executive briefing:\n\n"
            f"{text}\n\nExecutive Summary:"
        )
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    else:
        return [
            {"role": "system", "content": "Summarize the text concisely."},
            {"role": "user", "content": f"Text:\n{text}\n\nSummary:"}
        ]


def build_classification_prompt(text: str, strategy: str) -> list:
    """
    Constructs a prompt for Text Classification (Sentiment Analysis).
    
    Parameters:
        text (str): The text or product review to classify.
        strategy (str): 'Zero-shot Prompting', 'Few-shot Prompting', or 'Role Prompting'.
        
    Returns:
        list: A list of message dictionaries for the Groq Chat API.
    """
    text = text.strip()

    if strategy == "Zero-shot Prompting":
        system_instruction = (
            "You are a text classification assistant. Analyze the sentiment of the provided text. "
            "Classify the sentiment into exactly one of these three categories: Positive, Negative, or Neutral.\n"
            "Respond in this exact format:\n"
            "Label: [Positive / Negative / Neutral]\n"
            "Explanation: [A 1-2 sentence explanation of why this label was chosen based on the text]"
        )
        user_content = f"Text: {text}\n\nClassification:"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    elif strategy == "Few-shot Prompting":
        system_instruction = (
            "You are a sentiment classification assistant. Classify the sentiment into Positive, Negative, or Neutral. "
            "Follow the exact format of the examples below."
        )
        examples = (
            "Example 1:\n"
            "Text: The product is amazing and exceeded all my expectations.\n"
            "Label: Positive\n"
            "Explanation: The customer expresses strong enthusiasm and high satisfaction.\n\n"
            "Example 2:\n"
            "Text: The product is terrible and stopped working after two days.\n"
            "Label: Negative\n"
            "Explanation: The customer reports product failure and strong dissatisfaction.\n\n"
            "Example 3:\n"
            "Text: The product arrived yesterday afternoon.\n"
            "Label: Neutral\n"
            "Explanation: The statement describes an objective delivery event without emotional sentiment.\n\n"
        )
        user_content = f"{examples}Now classify this text:\nText: {text}\nLabel:"
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    elif strategy == "Role Prompting":
        system_instruction = (
            "You are a professional sentiment analysis specialist and customer insights analyst. "
            "Carefully evaluate customer feedback, determine the precise sentiment polarity "
            "(Positive, Negative, or Neutral), and provide a concise justification highlighting emotional cues."
        )
        user_content = (
            f"Analyze the sentiment of this customer statement:\n"
            f"\"{text}\"\n\n"
            f"Provide your verdict in this format:\n"
            f"Label: [Positive / Negative / Neutral]\n"
            f"Explanation: [Short justification]"
        )
        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    else:
        return [
            {"role": "system", "content": "Classify sentiment as Positive, Negative, or Neutral."},
            {"role": "user", "content": f"Text: {text}"}
        ]


def format_messages_for_display(messages: list) -> str:
    """
    Helper function to render the structured messages as a readable string
    so students can inspect the exact prompt sent to the LLM.
    """
    formatted = []
    for msg in messages:
        role_title = msg.get("role", "user").upper()
        content = msg.get("content", "")
        formatted.append(f"[{role_title}]\n{content}")
    return "\n\n" + ("=" * 40) + "\n\n".join(formatted)
