
import requests
import json

DEEPSEEK_API_KEY = 'sk-kzuzuayliboyxyrsfvdknuvhzmmhdaygdnsooerfmfyurhkp'

def generate_sql_with_ai(prompt: str, model: str="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
    try:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        payload = {
          "model": model,
          "messages": [
              {
                  "role": "user",
                  "content": prompt
              }
          ],
         "stream": False,
         "max_tokens": 2048,
         "stop": None,
         "temperature": 0.7,
         "top_p": 0.7,
         "top_k": 50,
         "frequency_penalty": 0.5,
         "n": 1,
         "response_format": {"type": "text"},
         }
        headers = {
          "Authorization": "Bearer " + DEEPSEEK_API_KEY,
          "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        response_json = json.loads(response.text)
        ai_suggestion = response_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"发生错误: {str(e)}")
        ai_suggestion = "很抱歉，无法获取建议。请稍后再试。"
    return ai_suggestion