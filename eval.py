import torch
from utilss import BigramLanguageModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 64 # what is the maximum context length for predictions?


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

# Define the context/question
context_str = "[START] Q: What's your email? A:"
context_encoded = torch.tensor([encode(context_str)], dtype=torch.long).to(device)

# Generate the response
with torch.no_grad():
    generated_ids = m.generate(context_encoded, max_new_tokens=block_size)
    generated_str = decode(generated_ids[0].tolist())

print("Generated Response:")
print(generated_str)