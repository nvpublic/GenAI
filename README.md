```markdown
# AIprojects

## Objective
The goal of this project is to enable users to query local models or popular models like Llama 3.2 without incurring token costs. This allows users to access and retrieve information efficiently and cost-effectively.

## Flask Components
- **Flask**: A micro web framework in Python used to build web applications and APIs.
- **request**: Provides access to incoming HTTP request data.
- **jsonify**: Converts Python objects to JSON format to send responses back to clients.
- **render_template**: Renders HTML templates to display in a web browser.
- **CORS**: Cross-Origin Resource Sharing. It allows a web application running at one domain (e.g., localhost:5000) to interact with a resource on another domain (e.g., an API server at localhost:11434). This is important for security but needs to be enabled for certain use cases.
- **requests**: A Python library used for making HTTP requests to APIs or web servers.

## Summary
Flask is used to create a web server that listens for incoming requests. It renders an HTML template for the UI and provides an API (/chat) for handling chat requests. CORS is enabled to allow cross-origin requests. The `llama3` function sends user input to an API hosting the llama3.2 model, retrieves a response, and handles errors. The chat route listens for POST requests, processes user input, and sends it to the model via the `llama3` function. This is essentially a chatbot interface that connects the front-end (webpage) to a language model (like Llama) running on a backend API.

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
- **Ollama Model (Llama 3.2)**: Ensure that the Ollama model is installed and running on your machine at `http://localhost:11434`.

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
```
Install the dependencies:
```bash
pip install -r requirements.txt
```

### Step 4: Configure the Flask Application
Ensure the URL in the `llama3` function points to the correct API endpoint for the model:
```python
url = "http://localhost:11434/api/chat"
```

### Step 5: Run the Ollama Service
Ensure that the Ollama model service (Llama 3.2) is running and accessible at `http://localhost:11434`. Check if Ollama is running:
```bash
curl http://localhost:11434/status
```

### Step 6: Run the Flask Application
From your project directory, run:
```bash
# For Windows:
python app.py
# For macOS/Linux:
python3 app.py
```
This will start the Flask application on `http://localhost:5000`.

### Step 7: Test the API
Test the chat API with a POST request:
```bash
curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, Llama"}'
```

### Optional: Debugging
If you encounter errors, verify that Ollama is running and listening on the correct port. Enable Flask debugging:
```python
if __name__ == "__main__":
    app.run(debug=True)
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
python app.py
```
Access the application at `http://localhost:5000/`.

Following these steps will get your Flask app running with the Llama 3.2 model.  
```
 
