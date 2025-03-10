import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langdetect import detect
from translate import Translator
from openai.error import OpenAIError
from dotenv import load_dotenv
import os
import requests
import openai  # Import the openai module
import tiktoken

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the URLs and API keys from environment variables
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:5000/llama")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "{default_api_key}")
OPENAI_BASE_URL = "https://api.openai.com/v1"
# openai.api_key = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
translator = Translator(to_lang="en")


def count_tokens(text, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)


def call_model_api(prompt, model_name):
    if model_name.startswith("llama"):
        url = LLAMA_API_URL
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get("message", {}).get(
                "content", "No content returned."
            )
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
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error communicating with the DeepSeek model: {e}"
    elif model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
        print(f"DEBUG: Selected model {model_name} with prompt length: {len(prompt)}")
        # Ensure prompt and completion tokens do not exceed the limit
        if count_tokens(prompt, model_name) > 12800:  # Corrected token limit
            return "Error: Prompt exceeds token limit"
        try:
            print(f"DEBUG: Querying OpenAI for model {model_name}")
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
            )
            return response.choices[0].message["content"].strip()
        except OpenAIError as e:
            # Use the correct error module
            if f"model `{model_name}` does not exist" in str(e):
                return f"Error: The model '{model_name}' does not exist or you do not have access to it."
            return f"Error communicating with the {model_name} model: {e}"
    elif model_name.startswith("dalle-mini"):
        # DALL-E models can be used for image generation
        try:
            response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
            return response["data"][0]["url"]
        except OpenAIError as e:
            # Use the correct error module
            return f"Error communicating with the {model_name} model: {e}"
    else:
        return "Unsupported model"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    model_names = request.json.get("models", ["llama"])
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    input_language = detect(user_input)
    responses = []
    for model_name in model_names:
        start_time = time.time()
        model_response = call_model_api(user_input, model_name)
        end_time = time.time()
        time_taken = end_time - start_time
        if input_language != "en":
            model_response = translator.translate(
                model_response, to_lang=input_language
            )
        responses.append(
            {"model": model_name, "response": model_response, "time_taken": time_taken}
        )
    return jsonify({"responses": responses})


if __name__ == "__main__":
    app.run(port=11435, debug=True)
