# AI Projects

## Objective
The goal of this project is to enable users to query local models or popular models like LLaMA, DeepSeek, GPT-3.5-turbo, GPT-4, and DALL-E Mini without incurring token costs. This allows users to access and retrieve information efficiently and cost-effectively.

## Flask Components
- **Flask**: A micro web framework in Python used to build web applications and APIs.
- **request**: Provides access to incoming HTTP request data.
- **jsonify**: Converts Python objects to JSON format to send responses back to clients.
- **render_template**: Renders HTML templates to display in a web browser.
- **CORS**: Cross-Origin Resource Sharing. It allows a web application running at one domain (e.g., localhost:5000) to interact with a resource on another domain (e.g., an API server at localhost:11434). This is important for security but needs to be enabled for certain use cases.
- **requests**: A Python library used for making HTTP requests to APIs or web servers.
- **openai**: A Python library used to interact with OpenAI's GPT models.

## Summary
Flask is used to create a web server that listens for incoming requests. It renders an HTML template for the UI and provides an API (/chat) for handling chat requests. CORS is enabled to allow cross-origin requests. The `call_model_api` function sends user input to various models (LLaMA, DeepSeek, GPT-3.5-turbo, GPT-4, DALL-E Mini), retrieves responses, and handles errors. The chat route listens for POST requests, processes user input, and sends it to the selected model(s) via the `call_model_api` function. This is essentially a chatbot interface that connects the front-end (webpage) to multiple language models running on backend APIs.

## Installation Instructions

### Prerequisites
- **Python**: Ensure Python 3.7 or higher is installed on your system. Verify with:
  ```bash
  python --version
  ```
- **Python Package Manager (pip)**: Verify with:
  ```bash
  pip --version
  ```
- **Git**: Verify with:
  ```bash
  git --version
  ```
- **OpenAI API Key**: Ensure you have an OpenAI API key. You can get one from [OpenAI](https://beta.openai.com/signup/).

### Step 1: Clone the Repository or Get the Code
```bash
git clone https://github.com/<your-repo-name>.git
cd <your-repo-directory>
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
# For Windows:
python -m venv venv
# For macOS/Linux:
python3 -m venv venv
```
Activate the virtual environment:
```bash
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
Create `requirements.txt` (if not already in the repository):
```txt
blinker==1.8.2
certifi==2024.8.30
charset-normalizer==2.1.1
click==8.1.7
flask==3.0.3
Flask-Cors==3.0.10
idna==3.10
importlib-metadata==8.5.0
itsdangerous==2.2.0
jinja2==3.1.4
MarkupSafe==2.1.5
requests==2.28.1
six==1.16.0
urllib3==1.26.20
werkzeug==3.0.4
zipp==3.20.2
openai==0.27.0
tiktoken==0.3.0
langdetect==1.0.9
python-dotenv==1.0.0
```
Install the dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Configure the Flask Application
Ensure the URLs and API keys in the `genAI_App.py` file point to the correct API endpoints and keys:
```python
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:5000/llama")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = "https://api.openai.com/v1"
openai.api_key = OPENAI_API_KEY
```

### Step 5: Set Up Environment Variables
Create a `.env` file in the project directory and add your API keys:
```env
LLAMA_API_URL=http://localhost:5000/llama
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=your_deepseek_base_url
OPENAI_API_KEY=your_openai_api_key
```

### Step 6: Run the Flask Application
From your project directory, run:
```bash
# For Windows:
python genAI_App.py
# For macOS/Linux:
python3 genAI_App.py
```
This will start the Flask application on `http://localhost:11435`.

### Step 7: Test the API
Test the chat API with a POST request:
```bash
curl -X POST http://localhost:11435/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, GPT-3.5-turbo", "models": ["gpt-3.5-turbo"]}'
```

### Optional: Debugging
If you encounter errors, verify that all services are running and listening on the correct ports. Enable Flask debugging:
```python
if __name__ == "__main__":
    app.run(port=11435, debug=True)
```

## Summary of Commands
Clone the repository:
```bash
git clone https://github.com/<your-repo>.git
cd <your-repo-directory>
```
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Start the Flask application:
```bash
python genAI_App.py
```
Access the application at `http://localhost:11435/`.

Following these steps will get your Flask app running with the various models.

