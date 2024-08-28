curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "Meta-Llama-3-8B",
    "messages": [
      {
        "role": "system", 
        "content": "You are a helpful assistant."
      }, 
      {
        "role": "user", 
        "content": "Say Hi to me!"
      }
    ],
    "max_tokens": 128
  }'