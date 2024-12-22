import torch
from utils import BigramLanguageModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block_size = 256 # what is the maximum context length for predictions?


torch.manual_seed(1337)
input = "qaquestions.txt"
with open(input, 'r', encoding='utf-8') as f:
    text = f.read() 
print("length of dataset in characters: ", len(text))
# Split the dataset into lines
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

# Define the context/question
context_str = "Q: what can you tell me about your projects? A:"
context_encoded = torch.tensor(encode(context_str), dtype=torch.long).unsqueeze(0).to(device)

# Generate the response
with torch.no_grad():
    generated_ids = m.generate(context_encoded, max_new_tokens=block_size, itos=itos)
    if isinstance(generated_ids, str):
        print(generated_ids)
        exit()
    generated_str = decode(generated_ids[0].tolist())

print("Generated Response:")
print(generated_str)
