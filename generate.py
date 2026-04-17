import os
import pickle
import torch
import sys
from model.gpt import GPT, GPTConfig

# --- EMERGENCY SETTINGS (Strict Mode) ---
# Force CPU for stability
device = 'cpu' 
# Top-K = 5: Only pick from the top 5 most likely characters. Prevents "randomness".
top_k = 5          
# Temp = 0.2: Make the model boring and precise. No creativity allowed.
temperature = 0.2   

# --- Paths ---
data_dir = os.path.join('data', 'seccode')
ckpt_path = os.path.join(data_dir, 'ckpt.pt')
meta_path = os.path.join(data_dir, 'meta.pkl')

print(f"✨ SecCodeGPT: Initializing on {device}...")

# --- 1. Load Vocabulary ---
if not os.path.exists(meta_path):
    print("❌ Error: meta.pkl not found")
    sys.exit()

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- 2. Load Model ---
if not os.path.exists(ckpt_path):
    print("❌ Error: ckpt.pt not found. Did you train the model?")
    sys.exit()

print("🧠 Loading neural network weights...")
# weights_only=False is required for custom classes like GPTConfig
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
config = checkpoint['config']

model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval() 

print("✅ System Online.")

# --- 3. Interactive Loop ---
print("\n" + "="*50)
print("   💀 SecCodeGPT: Offensive Security Copilot")
print("   STRICT MODE ACTIVE (Temp: 0.2, Top-K: 5)")
print("   Type 'exit' to quit.")
print("="*50)

while True:
    user_input = input("\n[PROMPT] > ")
    
    if user_input.lower() in ['exit', 'quit']:
        break
    
    if not user_input.strip():
        continue

    # Auto-Complete: If user types "scan", we give them a full python header
    # This helps the model NOT fail.
    if user_input == "scan":
        prompt = "import socket\nimport sys\n\n# Port Scanner\ndef scan(ip, port):\n    s = socket.socket("
    elif not user_input.startswith(("#", "//", "import", "def", "void")):
        prompt = f"# {user_input}\n"
    else:
        prompt = user_input

    print(f"\n--- GENERATING... ---\n")
    print(prompt, end="") 

    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    try:
        generated_indices = model.generate(
            context, 
            max_new_tokens=200, # Shortened to prevent infinite loops
            temperature=temperature, 
            top_k=top_k
        )
        
        full_text = decode(generated_indices[0].tolist())
        new_text = full_text[len(prompt):]
        print(new_text)
        
    except KeyboardInterrupt:
        print("\n[!] Generation stopped by user.")
    
    print("\n" + "-"*50)