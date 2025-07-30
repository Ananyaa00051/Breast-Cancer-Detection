from ollama import Client

client = Client(host='http://localhost:11434')

def chat_with_kash(user_message, chat_history):
    history_text = "\n".join(chat_history)
    prompt = f"""
You are KASH, a helpful, sensitive AI assistant that answers breast cancer–related questions in plain English.

Chat history:
{history_text}

User: {user_message}
KASH:"""

    response = client.chat(
        model='mistral',
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()
