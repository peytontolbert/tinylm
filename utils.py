import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

n_embd = 128
block_size = 256 # what is the maximum context length for predictions?
device = torch.device("cuda:0")

class MoE(nn.Module):
    def __init__(self, n_embd, num_experts, expert_layer_size, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([FeedForward(expert_layer_size) for _ in range(num_experts)])
        self.gate = nn.Linear(n_embd, num_experts)
        self.capacity_factor = capacity_factor

    def forward(self, x):
        gating_scores = F.softmax(self.gate(x), dim=-1)
        # Get the index of the expert with highest gate score per token
        # The shape of expert_assignment becomes [Batch, Seq Len]
        expert_assignment = torch.argmax(gating_scores, dim=-1)
        # Determine capacity based on the input size and capacity factor
        capacity = max(1, math.ceil(x.size(0) * self.capacity_factor / self.num_experts))
        #capacity = max(1, capacity_pre / self.num_experts)
        outputs = torch.zeros_like(x, dtype=x.dtype)
        expert_load = torch.zeros(self.num_experts, dtype=torch.int32, device=x.device)
        #print("expert_load: ", expert_load.shape)
        #expert_index = torch.argmax(gating_scores, dim=-1)
        # Applying each expert to the corresponding elements
        #batch_size, seq_len, _ = x.size()
        #outputs = x.new_zeros(batch_size, seq_len, n_embd)
        for i in range(self.num_experts):
            # Create a mask where the current expert is assigned to the tokens based on gating scores
            # The mask has the original batch size and sequence length dims and is expanded to the hidden dim
            tokens_to_expert_mask = (expert_assignment == i).unsqueeze(-1).expand_as(x)

            # Assign and process tokens by the expert network where mask is True
            if tokens_to_expert_mask.any():
                expert_input = x.masked_select(tokens_to_expert_mask).view(-1, x.size(-1))
                expert_output = self.experts[i](expert_input).to(x.dtype)  # Process selected input through the current expert

                # Scatter computed expert outputs back to their original positions in the outputs tensor
                outputs = outputs.masked_scatter(tokens_to_expert_mask, expert_output.view(*expert_input.shape))

            # Select elements for this expert
            #mask = (expert_index == i)
            #if mask.any():
            #    outputs[mask] = self.experts[i](x[mask])
        return outputs


class BlockWithMoE(nn.Module):
    """ Transformer block with MoE instead of a standard FeedForward layer """

    def __init__(self, n_embd, n_head, num_experts, expert_layer_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.moe = MoE(n_embd, num_experts, expert_layer_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.cross_attention = CrossAttention(n_embd, head_size=head_size)

    def forward(self, x, context=None):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        if context is not None:
            x = x + self.cross_attention(x, context)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CrossAttention(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x1, x2):
        # x1 is the query, x2 is the key and value
        k = self.key(x2)
        q = self.query(x1)
        v = self.value(x2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out
    
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """" multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, input_size),
        )
    
    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, num_experts=8, expert_layer_size=n_embd):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.n_embd = n_embd
        # self.sa_head = Head(n_embd)
        self.blocks = nn.Sequential(
            BlockWithMoE(n_embd, n_head=2, num_experts=num_experts, expert_layer_size=expert_layer_size),
            BlockWithMoE(n_embd, n_head=2, num_experts=num_experts, expert_layer_size=expert_layer_size),
            BlockWithMoE(n_embd, n_head=2, num_experts=num_experts, expert_layer_size=expert_layer_size),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        aux_loss = 0.0
        x = self.blocks(x) # (B,T,C)
        # If your MoE blocks return a tuple (output, aux_loss):
        if isinstance(x, tuple):
            x, aux_loss = x  # Unpack the output and auxiliary loss
            aux_loss = aux_loss.sum()  # Sum auxiliary losses from multiple layers (if applicable)
        
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            #print(loss)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, itos, temperature=.1, prob_threshold=0.2):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply temperature and softmax to get probabilities
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            max_prob, idx_next = torch.max(probs, dim=-1, keepdim=True)
            if max_prob.item() < prob_threshold:
                return "This is not in my training data, please contact Peyton for more information."
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Check if the generated token is the [END] token
            if itos[idx_next.item()] == "[END]":
                break

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
