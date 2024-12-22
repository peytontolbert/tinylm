import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from utils import BigramLanguageModel
from moe import MoE
import math
import random
from torch.utils.data import Dataset, DataLoader

# Empties the entire PyTorch CUDA cache
torch.cuda.empty_cache()


class TextDataset(Dataset):
    def __init__(self, encoded_text, block_size):
        # Assuming encoded_text is a list of integers representing encoded characters
        self.data = encoded_text
        self.block_size = block_size

    def __len__(self):
        # The length is the number of blocks we can make
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # Get the sequence of tokens that starts at this index
        chunk = self.data[idx:idx + self.block_size + 1]
        # Input sequence (x) is the first block_size characters
        # Target sequence (y) is the last block_size characters
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
warmup_iters = 200 # how many steps to warm up for
decay_lr = True # whether to decay the learning rate
lr_decay_iters = 5000 # should be ~= max_iters
max_iters = 5000 # total number of steps to train for
eval_interval = 1
learning_rate = .01
min_lr = 1e-2 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
device = torch.device("cuda:0")
# ------------

torch.manual_seed(1337)
input = "qaquestions.txt"
with open(input, 'r', encoding='utf-8') as f:
    text = f.read() 
print("length of dataset in characters: ", len(text))
# Split the dataset into lines
lines = text.split('\n')
# Shuffle the lines
random.shuffle(lines)

# Recombine into a single string
shuffled_text = '\n'.join(lines)
text = shuffled_text
print(text[:200])
#All the unique chars
# Tokenize the text into words
words = text.split()
vocab = sorted(set(words + ["[UNK]", "[END]"]))
vocab_size = len(vocab)
print("Vocab size: ", vocab_size)
print(' '.join(vocab))

stoi = { w:i for i,w in enumerate(vocab) }
itos = { i:w for i,w in enumerate(vocab) }
encode = lambda s: [stoi.get(w, stoi["[UNK]"]) for w in s.split()] # encoder: take a string, output a list of integers
decode = lambda l: ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Let's now split up the data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Create datasets
train_dataset = TextDataset(train_data, block_size)
val_dataset = TextDataset(val_data, block_size)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, device, train_dataloader, val_dataloader):
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    out = {}
    model.eval()
    for split, dataloader in dataloaders.items():
        total_loss = 0
        num_batches = 0
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        out[split] = avg_loss
        model.train()
    return out


model = BigramLanguageModel(vocab_size)
m = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

scaler = GradScaler('cuda')
num_epochs = 10
step_count = 0
for epoch in range(num_epochs):
    lr = learning_rate
    #lr = get_lr(epoch) if decay_lr else learning_rate
    for xb, yb in train_dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            logits, total_loss = m(xb, yb)
        if total_loss is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_count += 1
            if step_count % 100 == 0:  # Log every 100 steps
                print(f"step count: {step_count}")
            if step_count % 1000 == 0:  # Save checkpoint every 1000 steps
                print(f"Saving checkpoint at epoch {epoch}, step {step_count}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'stoi': stoi,
                    'itos': itos
                }, 'new_2024_complete.pth')
                print(total_loss.item())
    losses = estimate_loss(model, device, train_dataloader, val_dataloader)
    print(f"Epoch {epoch}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos
    }, 'model_complete.pth')

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(total_loss.item())
torch.save({
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'itos': itos
}, 'model_complete.pth')
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))
