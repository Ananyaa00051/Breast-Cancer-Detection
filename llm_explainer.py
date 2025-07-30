# llm_explainer.py
import ollama

def explain_medical_term(term):
    prompt = f"Explain the medical term '{term}' in simple language, suitable for a non-medical person. Include relevance to breast cancer diagnosis."
    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']
