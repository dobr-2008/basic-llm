import os 
import torch 
import torch.nn as nn
import ast
from tokenizers import ByteLevelBPETokenizer

device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

tokenizer_file = "bpe_tokenizer_vocab.json" 
corpus_file = "words_pos.txt"

filtered_tokens = "filtered_tokens.txt"

tokenizer_file = "bpe_tokenizer-vocab.json"
corpus_file = "words_pos.txt"

if os.path.exists(tokenizer_file):
  print("Loading existing 30k BPE tokenizer...")
  tokenizer = ByteLevelBPETokenizer(
      "bpe_tokenizer-vocab.json", "bpe_tokenizer-merges.txt"
  )
else:
  print("Training new 30k BPE Tokenizer on corpus...")
  tokenizer = ByteLevelBPETokenizer()
  tokenizer.train(
      files=[corpus_file],
      vocab_size=30000,
      min_frequency=2,
      special_tokens=["<|pad|>", "<|endoftext|>"],
  )
  tokenizer.save_model(".", "bpe_tokenizer")
  print("tokenizer trained and saved successfully!")


print("encoding")
with open(corpus_file, encoding="utf-8") as f:
  raw_text = f.read()

encoded_output = tokenizer.encode(raw_text)
all_token_ids = encoded_output.ids
vocab_size = tokenizer.get_vocab_size()
print(f"Total tokenized corpus length: {len(all_token_ids)} tokens")
print(f"Vocabulary size: {vocab_size}")



class positionalembedding(nn.Module):

    def __init__(self , maxlen , vocab_size , embed_dim):
       super().__init__()
       self.token_emb = nn.Embedding(vocab_size, embed_dim)
       self.pos_emb = nn.Embedding(maxlen , embed_dim)

    def forward (self , x):
       seq_len = x.size(1)
       positions = torch.arange(0 , seq_len , device=x.device)
       positions = self.pos_emb(positions)
       x = self.token_emb(x)
       return x + positions
  
embedding_dim = 128
maxlen = 20

class TransformerBlock(nn.Module):

   def __init__(self, embed_dim , num_heads):
      super().__init__()

      self.attention = nn.MultiheadAttention(
         embed_dim=embed_dim , num_heads=num_heads , batch_first=True
      )

      self.ln1 = nn.LayerNorm(embed_dim)
      self.dense = nn.Sequential(
         nn.Linear(embed_dim, 128),
         nn.ReLU(),
         nn.Linear(128, embed_dim),
      )

      self.ln2 = nn.LayerNorm(embed_dim)
   def forward(self , x ):
      seq_len = x.size(1)
      mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
      attn_out, _ = self.attention( x , x , x , attn_mask = mask)
      x = self.ln1(x + attn_out)

      dense_out = self.dense(x)
      x = self.ln2(x + dense_out)
      return x 

encoded_output = tokenizer.encode(raw_text)
all_token_ids = encoded_output.ids
vocab_size = tokenizer.get_vocab_size()

class mTransformer(nn.Module):
  def __init__(self , vocab_size , embed_dim , maxlen , num_heads):
     super().__init__()
     self.pos_embedding = positionalembedding(maxlen , vocab_size , embed_dim)
     self.transformer = TransformerBlock(embed_dim, num_heads)
     self.fc_out = nn.Linear(embed_dim, vocab_size)

  def forward(self, x ):
     x = self.pos_embedding(x)
     x = self.transformer(x)
     x = self.fc_out(x)
     return x 

model = mTransformer(
    vocab_size, embedding_dim, maxlen, num_heads=8
).to(device)
print(model)

sample_text = "this is test text"
encodes_ids = tokenizer.encode(sample_text).ids

input_tensor = torch.tensor([encodes_ids] , dtype=torch.long, device=device)

if input_tensor.size(1) < maxlen:
   padding = torch.zeros(
      (1, maxlen - input_tensor.size(1)), dtype=torch.long ,device = device 
   )
   input_tensor = torch.cat([input_tensor, padding] , dim = 1)
elif input_tensor.size(1) > maxlen:
   input_tensor = input_tensor[:, :maxlen]

model.eval()
with torch.no_grad():
   model_output = model(input_tensor)

print(model_output.shape)

#model training 

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim 


class TextDataset(Dataset):
   def  __init__(self , token_ids , maxlen):
      self.token_ids = token_ids
      self.maxlen = maxlen

   def  __len__(self):
      return len (self.token_ids) - self.maxlen

   def __getitem__(self , idx):
      x = self.token_ids[idx : idx + self.maxlen]

      y = self.token_ids[idx + 1 : idx + self.maxlen + 1]
      return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

#windows doesnt allow to run num workers without putting it onto python main guard 
if __name__ == "__main__":
  dataset = TextDataset(all_token_ids, maxlen=20)
  train_loader = DataLoader(dataset, batch_size=128, shuffle=True , num_workers=2 , pin_memory= True)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr = 0.001)

  epochs = 3 
  model.train()

  scaler = torch.amp.GradScaler("cuda")

  print("started training")
  for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      inputs = inputs.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()

      with torch.amp.autocast("cuda"):
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      total_loss += loss.item()

      if batch_idx % 500 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}]"
            f" | Loss: {loss.item():.4f}") 

    avg_loss = total_loss / len(train_loader)
    print(f"--> Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}\n")
      
  print("training finished")

  torch.save(model.state_dict(), "transformer_model.pth")
  print("Model saved successfully!")

