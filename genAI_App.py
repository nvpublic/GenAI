from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from langdetect import detect
from translate import Translator
import openai
import tiktoken
from openai.error import OpenAIError  # Correct import
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the URLs and API keys from environment variables
LLAMA_API_URL = os.getenv('LLAMA_API_URL', 'http://localhost:5000/llama')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_BASE_URL = "https://api.openai.com/v1"

openai.api_key = OPENAI_API_KEY
translator = Translator(to_lang='en')

def count_tokens(text, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def call_model_api(prompt, model_name):
    if model_name.startswith("llama"):
        url = LLAMA_API_URL
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("message", {}).get("content", "No content returned.")
        except requests.exceptions.RequestException as e:
            return f"Error communicating with the model: {e}"
        except ValueError:
            return "Invalid response format"
    elif model_name.startswith("deepseek"):
        try:
            response = requests.post(
                DEEPSEEK_BASE_URL + "/v1/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ]
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error communicating with the DeepSeek model: {e}"
    elif model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
        # Ensure prompt and completion tokens do not exceed the limit
        if count_tokens(prompt, model_name) > 12800:  # Corrected token limit
            return "Error: Prompt exceeds token limit"
        
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50
            )
            return response.choices[0].message['content'].strip()
        except OpenAIError as e:  # Use the correct error module
            if f"model `{model_name}` does not exist" in str(e):
                return f"Error: The model '{model_name}' does not exist or you do not have access to it."
            return f"Error communicating with the {model_name} model: {e}"
    elif model_name.startswith("dalle-mini"):
        # DALL-E models can be used for image generation
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            return response['data'][0]['url']
        except OpenAIError as e:  # Use the correct error module
            return f"Error communicating with the {model_name} model: {e}"
    else:
        return "Unsupported model"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    model_name = request.json.get("model", "llama")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    input_language = detect(user_input)

    model_response = call_model_api(user_input, model_name)

    if input_language != 'en':
        model_response = translator.translate(model_response, to_lang=input_language)

    return jsonify({"response": model_response})

if __name__ == "__main__":
    app.run(port=11435, debug=True)

