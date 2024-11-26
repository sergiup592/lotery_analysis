import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class LotteryDataset(Dataset):
    def __init__(self, data: List[int], seq_length: int = 10):
        # Normalize to [0, 1] range
        self.data = torch.tensor([x/50.0 for x in data], dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x.unsqueeze(-1), y.unsqueeze(-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Initialize weights with smaller values
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.W_o(context)
        
        return output

class NumberPredictor(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding with layer normalization
        self.embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(d_model, num_heads),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),  # Changed to GELU
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 50),  # Output for each possible number (1-50)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize weights with smaller values
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, x):
        x = self.embedding(x)
        
        # Create attention mask for causality
        seq_length = x.size(1)
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        mask = ~mask.to(x.device)
        
        for layer in self.layers:
            # Self-attention with mask
            attn_out = layer['attention'](x, mask)
            x = layer['norm1'](x + attn_out)
            
            # Feed forward
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)
        
        # Output logits (not normalized)
        return self.output_net(x)

def train_model(data: List[int], epochs: int = 200, seq_length: int = 10,
                batch_size: int = 64, learning_rate: float = 0.0005,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    # Create dataset
    dataset = LotteryDataset(data, seq_length)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = NumberPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                  epochs=epochs,
                                                  steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            target_indices = ((batch_y * 50).long().clamp(0, 49)).squeeze(-1)
            
            loss = criterion(predictions.view(-1, 50), target_indices.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_x)
                target_indices = ((batch_y * 50).long().clamp(0, 49)).squeeze(-1)
                
                loss = criterion(predictions.view(-1, 50), target_indices.view(-1))
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}:')
            print(f'  Training Loss: {avg_train_loss:.4f}')
            print(f'  Validation Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def predict_next_numbers(model: NumberPredictor, sequence: List[int],
                        n_predict: int = 7, seq_length: int = 10,
                        temperature: float = 0.8,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Normalize input sequence
        input_seq = torch.tensor([x/50.0 for x in sequence[-seq_length:]], 
                               dtype=torch.float32)
        input_seq = input_seq.unsqueeze(0).unsqueeze(-1).to(device)
        
        for _ in range(n_predict):
            logits = model(input_seq)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample with nucleus sampling (top-p)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample and convert to actual number
            next_index = torch.multinomial(probs, 1).item()
            next_number = next_index + 1  # Convert to 1-50 range
            predictions.append(next_number)
            
            # Update input sequence
            input_seq = torch.cat([input_seq[:, 1:],
                                 torch.tensor([[[next_number/50.0]]]).to(device)], dim=1)
    
    return predictions

def extract_numbers_to_list(input_file):
    all_numbers = []
    current_set = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if 'th' in line:
                if current_set:
                    all_numbers = current_set + all_numbers
                current_set = []
                continue
                
            try:
                num = int(line.strip())
                if 0 < num <= 50:  # Ensure numbers are between 1 and 50
                    current_set.append(num)
            except ValueError:
                continue
    
    if current_set:
        all_numbers = current_set + all_numbers
    
    return all_numbers

if __name__ == "__main__":
    input_file = '../lottery_data/lottery_numbers.txt'
    numbers = extract_numbers_to_list(input_file)
    
    print(f"Training on {len(numbers)} numbers...")
    
    model = train_model(
        data=numbers,
        epochs=2000,
        seq_length=10,
        batch_size=64,
        learning_rate=0.0005
    )
    
    # Generate multiple sets of predictions with different temperatures
    temperatures = [0.8, 1.0, 1.2]
    for temp in temperatures:
        predictions = predict_next_numbers(model, numbers, n_predict=7, temperature=temp)
        print(f"\nPredicted next 7 numbers (temperature={temp}):", predictions)