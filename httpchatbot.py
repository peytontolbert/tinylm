from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
from utils import BigramLanguageModel
import requests
import time
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 256 # what is the maximum context length for predictions?


torch.manual_seed(1337)
input = "qaquestions.txt"
with open(input, 'r', encoding='utf-8') as f:
    text = f.read() 
print("length of dataset in characters: ", len(text))
# Split the dataset into lines
print(text[:200])
#All the unique chars
# Tokenize the text into words
words = text.split()
vocab = sorted(set(words + ["[UNK]", "[END]"]))
vocab_size = len(vocab)
print("Vocab size: ", vocab_size)
model = BigramLanguageModel(vocab_size)  # Ensure this is the same architecture as before
checkpoint = torch.load('finetuned_2024_complete.pth')
model.load_state_dict(checkpoint['model_state_dict'])

stoi = { w:i for i,w in enumerate(vocab) }
itos = { i:w for i,w in enumerate(vocab) }
# Encode function now uses <UNK> for unknown words
encode = lambda s: [stoi.get(w, stoi["[UNK]"]) for w in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

m = model.to(device)
m.eval()  # Set the model to evaluation mode
app = Flask(__name__)
CORS(app, origins=['*'])


# Initialize the counter from a file
counter_file = 'build_agent_counter.txt'
if os.path.exists(counter_file):
    with open(counter_file, 'r') as f:
        build_agent_counter = int(f.read().strip())
else:
    build_agent_counter = 0


# Load your model here (as you have done above)

def chat_with_ollama(prompt: str, system_prompt: str, retries: int=5, delay: int=5):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "hermes3",
        "prompt": f"""{prompt}""",
        "format": "json",
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    for i in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Ensure a 4XX/5XX error raises an exception
            response_data = response.json()  # Parse the JSON response
            print(f"response data: {response_data['response']}")
            if 'response' in response_data:
                actual_response = response_data['response']
                if 'code' in actual_response:
                    return response_data['response']  # Return the 'response' field
                revised_prompt = (
                "The output provided does not follow the required structure. Please ensure the response follows the exact format "
                "provided below. a name and code must be included:\n\n" + system_prompt + "\n\nHere is your previous output:\n\n" + response.text + "\n\nHere is the user prompt: \n\n" + prompt
           )
                payload['prompt'] = revised_prompt
            else:
                raise KeyError("'response' key not found in the API response")
        except (requests.exceptions.RequestException, KeyError) as e:
            if i < retries - 1:  # i is zero indexed
                time.sleep(delay)  # wait before trying again
            else:
                raise e  # re-raise the last exception if all retries fail

@app.route('/build_agent', methods=['POST'])
@cross_origin()
def generate_build():
    global build_agent_counter
    data = request.json
    print(data)
    context_str = data['description']
    system_prompt = "You are to generate a new Python agent class based on the user's request. Follow the exact structure provided in the template below. Replace the placeholders with the specific details from the user's request. Respond only in JSON format with two keys: 'name' (the agent class name) and 'code' (the Python code for the agent class). If no details are provided by the user, infer reasonable logic based on the agent's name and description provided by the user.\n\n### Class Template:\n\n```python\nclass {{AgentName}}(Agent):\n    def __init__(self):\n        super().__init__(name='{{AgentName}}')\n\n    def execute(self, input_data):\n        # Implement the logic for executing the task based on the input_data\n        {{execute_method_logic}}\n\n    def generate_prompt(self, input_data):\n        # Generate the prompt based on the input_data\n        {{generate_prompt_logic}}\n```\n\nRespond with the structured agent class in JSON format using the provided template."
    results = chat_with_ollama(context_str, system_prompt)
    if isinstance(results,dict):
        return jsonify({'response': results})
    else:
        try:
            response_data = results.json() if isinstance(results,requests.Response) else results
            build_agent_counter+=1
            return jsonify({'response': response_data})
        except Exception as e:
            print(f"Failed to serialize results: {e}")
            return jsonify({'error': 'Failed to process the response'}), 500

@app.route('/generate_code', methods=['POST'])
@cross_origin()
def generate_code():
    data = request.json
    print(data)
    context_str = data['description']
    system_prompt = (
        "You are to generate Python code based on the user's request. Ensure the code is complete, functional, and follows best practices. "
        "Respond only with the code in plain text format.\n\n"
        "### Example:\n\n"
        "If the user requests a function to add two numbers, your response should be:\n\n"
        "```python\n"
        "def add_numbers(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Use this structure as a guide for your response."
    )
    results = chat_with_ollama(context_str, system_prompt)
    if isinstance(results,dict):
        return jsonify({'response': results})
    else:
        try:
            response_data = results.json() if isinstance(results,requests.Response) else results
            return jsonify({'response': response_data})
        except Exception as e:
            print(f"Failed to serialize results: {e}")
            return jsonify({'error': 'Failed to process the response'}), 500

@app.route('/send_message', methods=['POST'])
@cross_origin()
def generate_response():
    data = request.json
    context_str = data['message']
    context = "[START]Q: " + context_str + " A: "
    context_encoded = torch.tensor([encode(context)], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_ids = m.generate(context_encoded, max_new_tokens=block_size, itos=itos)
        if isinstance(generated_ids, str):
            print(generated_ids)
            return jsonify({'response': generated_ids})
        generated_str = decode(generated_ids[0].tolist())
    # Extract response
    response_start = generated_str.find('A: ') + 3  # Adding 3 to move past 'A: '
    response_end = generated_str.find('[END]')
    if response_end == -1:  # If '[END]' not found
        response_end = None
    extracted_response = generated_str[response_start:response_end]

    print(extracted_response)
    return jsonify({'response': extracted_response})

@app.route('/check_status', methods=['GET'])
@cross_origin()
def check_status():
    status = 'Online'
    return jsonify({"status": status})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
