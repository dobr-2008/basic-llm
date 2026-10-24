import torch 
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = ByteLevelBPETokenizer(
    "bpe_tokenizer-vocab.json", "bpe_tokenizer-merges.txt"
)
vocab_size = tokenizer.get_vocab_size()
embedding_dim = 128
maxlen = 20

class positionalembedding(nn.Module):

  def __init__(self, maxlen, vocab_size, embed_dim):
    super().__init__()
    self.token_emb = nn.Embedding(vocab_size, embed_dim)
    self.pos_emb = nn.Embedding(maxlen, embed_dim)

  def forward(self, x):
    seq_len = x.size(1)
    positions = torch.arange(0, seq_len, device=x.device)
    positions = self.pos_emb(positions)
    x = self.token_emb(x)
    return x + positions



class TransformerBlock(nn.Module):

  def __init__(self, embed_dim, num_heads):
    super().__init__()
    self.attention = nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, batch_first=True
    )
    self.ln1 = nn.LayerNorm(embed_dim)
    self.dense = nn.Sequential(
        nn.Linear(embed_dim, 128),
        nn.ReLU(),
        nn.Linear(128, embed_dim),
    )
    self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, x):
    seq_len = x.size(1)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=x.device), diagonal=1
    ).bool()
    attn_out, _ = self.attention(x, x, x, attn_mask=mask)
    x = self.ln1(x + attn_out)

    dense_out = self.dense(x)
    x = self.ln2(x + dense_out)
    return x


class mTransformer(nn.Module):

  def __init__(self, vocab_size, embed_dim, maxlen, num_heads):
    super().__init__()
    self.pos_embedding = positionalembedding(maxlen, vocab_size, embed_dim)
    self.transformer = TransformerBlock(embed_dim, num_heads)
    self.fc_out = nn.Linear(embed_dim, vocab_size)

  def forward(self, x):
    x = self.pos_embedding(x)
    x = self.transformer(x)
    x = self.fc_out(x)
    return x


model = mTransformer(vocab_size, embedding_dim, maxlen, num_heads=8).to(device)
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

# generating of text happens here!
def generate_text(prompt, max_new_tokens=30):
  encoded = tokenizer.encode(prompt).ids
  input_tensor = torch.tensor([encoded], dtype=torch.long, device=device)

  for _ in range(max_new_tokens):
    
    context = input_tensor[:, -maxlen:]

    with torch.no_grad():
      output = model(context)
     
      next_token_logits = output[:, -1, :]
      next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

      
      input_tensor = torch.cat([input_tensor, next_token], dim=1)

  
  generated_ids = input_tensor[0].tolist()
  return tokenizer.decode(generated_ids)


#where you prompts in 
prompt = "how are you "
print(f"Prompt: {prompt}")
print(f"Generated: {generate_text(prompt)}")
