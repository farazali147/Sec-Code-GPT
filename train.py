import os
import time
import math
import pickle
import numpy as np
import torch
from model.gpt import GPT, GPTConfig

# --- Hyperparameters (Tune these for your laptop) ---
batch_size = 32        # How many code snippets the model reads at once
block_size = 256       # Context length (how far back it looks)
max_iters = 5000       # Total training steps
eval_interval = 250    # How often to check validation loss (Detailed check)
learning_rate = 3e-4   # How fast the model learns
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available(): device = 'mps' # For Mac Users

# Optimization for CPU: Reduced eval_iters from 200 to 20
# This makes the "validation check" 10x faster so you don't wait forever.
eval_iters = 20        

n_embd = 384           # Size of the internal vector
n_head = 6             # Number of attention heads
n_layer = 6            # Number of transformer layers
dropout = 0.2          # Regularization

print(f"🚀 Training on device: {device}")

# --- 1. The Memory-Mapped Data Loader ---
data_dir = os.path.join('data', 'seccode')

def get_batch(split):
    # We recreate the memmap every batch to avoid memory leaks
    filename = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Randomly select a starting point
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Stack them into a tensor
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Move to GPU/CPU
    x, y = x.to(device), y.to(device)
    return x, y

# --- 2. Load Vocabulary ---
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"✅ Found vocabulary: {vocab_size} tokens")
else:
    print("❌ ERROR: meta.pkl not found. Did you run prepare_scale.py?")
    exit()

# --- 3. Initialize Model ---
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd,
    dropout=dropout
)
model = GPT(config)
model.to(device)

print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 4. Loss Estimation Helper ---
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 5. Training Loop ---
print("🔥 Starting Training Loop (Press Ctrl+C to stop early)...")
t0 = time.time()

for iter in range(max_iters):

    # Every interval, perform a detailed check (Validation)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"\n📊 Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save checkpoint
        if iter > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(data_dir, 'ckpt.pt'))
            print("💾 Checkpoint saved.")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # --- THE HEARTBEAT (New Addition) ---
    # This prints every 10 steps so you know it's working
    if iter % 10 == 0:
        print(f"step {iter}: batch loss {loss.item():.4f}", end='\r')
    # ------------------------------------

print(f"✅ Training Finished in {(time.time()-t0)/60:.2f} minutes!")