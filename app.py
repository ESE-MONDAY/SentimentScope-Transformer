import os
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

# ---------------------------------------------------------
# 1. Model Architecture Definitions
# ---------------------------------------------------------
class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer('casual_attention_mask', casual_attention_mask)

    def forward(self, input_tensor):
        batch_size, tokens_num, _ = input_tensor.shape
        Q = self.Q_weights(input_tensor)
        K = self.K_weights(input_tensor)
        V = self.V_weights(input_tensor)

        attention_scores = Q @ K.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0,
            float('-inf')
        )
        attention_scores = attention_scores / (K.shape[-1] ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config["heads_num"])])
        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input_tensor):
        heads_outputs = [head(input_tensor) for head in self.heads]
        x = torch.cat(heads_outputs, dim=-1)
        x = self.linear(x)
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input_tensor):
        return self.linear_layers(input_tensor)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

    def forward(self, input_tensor):
        x = input_tensor + self.multi_head(self.layer_norm_1(input_tensor))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class DemoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])
        self.layers = nn.Sequential(*[Block(config) for _ in range(config["layers_num"])])
        self.layer_norm = nn.LayerNorm(config["d_embed"])
        self.classification_head = nn.Linear(config["d_embed"], config["num_classes"], bias=False)

    def forward(self, token_ids):
        _, tokens_num = token_ids.shape
        x = self.token_embedding_layer(token_ids)
        positions = torch.arange(tokens_num, device=token_ids.device)
        pos_embed = self.positional_embedding_layer(positions)
        x = x + pos_embed.unsqueeze(0)
        
        x = self.layers(x)
        x = self.layer_norm(x)
        pooled = torch.mean(x, dim=1)
        logits = self.classification_head(pooled)
        return logits

# ---------------------------------------------------------
# 2. FastAPI Application Setup & Model Loading
# ---------------------------------------------------------
app = FastAPI(title="SentimentScope API", version="1.0")

# Global variables for model, tokenizer, and device
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tokenizer_loaded: bool
    device: str

@app.on_event("startup")
def load_artifacts():
    global model, tokenizer, config
    checkpoint_path = "demogpt_sentiment_model.pt"
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint file '{checkpoint_path}' not found.")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Initialize model and load weights
    model = DemoGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("/health", response_model=HealthResponse)
def health_check():
    is_model_ready = model is not None
    is_tokenizer_ready = tokenizer is not None

    if not (is_model_ready and is_tokenizer_ready):
        raise HTTPException(
            status_code=503,
            detail="Service Unhealthy: Model artifacts are not fully loaded."
        )

    return HealthResponse(
        status="healthy",
        model_loaded=is_model_ready,
        tokenizer_loaded=is_tokenizer_ready,
        device=str(device)
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    inputs = tokenizer(
        [request.text],
        padding="max_length",
        truncation=True,
        max_length=config["context_size"],
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item() * 100

    labels = ["Negative", "Positive"]
    return PredictionResponse(
        text=request.text,
        sentiment=labels[prediction],
        confidence=round(confidence, 2)
    )