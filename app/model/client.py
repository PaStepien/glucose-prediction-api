import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_llama(prompt):
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_API_URL, json=payload)
    
    response.raise_for_status()
    return response.json()["response"]
        
        
    