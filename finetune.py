import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utilss import BigramLanguageModel
import random
from TextDataset import TextDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
learning_rate = 4e-4
max_iters = 100000
eval_interval = 400
eval_iters = 200
n_embd = 128  # Increasing embedding size.
n_head = 6  # More attention heads.
dropout = 0.0

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
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#stoi = { ch:i for i,ch in enumerate(chars) }
#itos = { i:ch for i,ch in enumerate(chars) }
# Let's now split up the data into train and validation sets

model = BigramLanguageModel(vocab_size)  # The model must be the same as the one you trained
checkpoint = torch.load('model_complete_finetuned2.pth')
model.load_state_dict(checkpoint['model_state_dict'])
stoi = checkpoint['stoi']
itos = checkpoint['itos']
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1:block_size+1]

# Create datasets
train_dataset = TextDataset(train_data, block_size)
val_dataset = TextDataset(val_data, block_size)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
m = model.to(device)


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

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


# create a PyTorch optimizer
optimizer = optim.AdamW(m.parameters(), lr=learning_rate)

num_epochs = 10
step_count = 0
for epoch in range(num_epochs):
    lr = learning_rate
    #lr = get_lr(epoch) if decay_lr else learning_rate
    for xb, yb in train_dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, total_loss = m(xb, yb)
        if total_loss is not None:
            total_loss.backward()
            optimizer.step()
            step_count += 1
            if step_count % 500 == 0:
                print(f"Saving checkpoint at epoch {epoch}, step {step_count}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'stoi': stoi,
                    'itos': itos
                }, 'model_complete_finetuned3.pth')
                print(total_loss.item())
    losses = estimate_loss(model, device, train_dataloader, val_dataloader)
    print(f"Epoch {epoch}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos
    }, 'model_complete.pth')
    # Update learning rate using the scheduler
    # scheduler.step()



context = torch.zeros((1,1), dtype=torch.long, device=device)
print(total_loss.item())
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))
