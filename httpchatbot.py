from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
from utilss import BigramLanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 256 # what is the maximum context length for predictions?


torch.manual_seed(1337)
input = "qaquestions.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

#All the unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

model = BigramLanguageModel(vocab_size)  # Ensure this is the same architecture as before
checkpoint = torch.load('model_chatbot.pth')
model.load_state_dict(checkpoint['model_state_dict'])
stoi = checkpoint['stoi']
itos = checkpoint['itos']
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
m = model.to(device)
m.eval()  # Set the model to evaluation mode
app = Flask(__name__)
CORS(app, origins=['*'])

# Load your model here (as you have done above)

@app.route('/send_message', methods=['POST'])
@cross_origin()
def generate_response():
    data = request.json
    context_str = data['message']
    context = "[START]Q: " + context_str + " A: "
    context_encoded = torch.tensor([encode(context)], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_ids = m.generate(context_encoded, max_new_tokens=block_size)
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
