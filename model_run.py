import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 32
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        energy = self.v(torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs)))
        attention_weights = F.softmax(energy.squeeze(2), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attention_weights

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (h, c) = self.lstm(embedded)
        return outputs, (h, c)

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(1)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        context = context.unsqueeze(1)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (h_new, c_new) = self.lstm(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        h_new = h_new.squeeze(0)
        c_new = c_new.squeeze(0)
        logits = self.fc_out(output.squeeze(1))
        return logits, h_new, c_new, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(vocab_size, embed_size, hidden_size, pad_idx)
        self.decoder = DecoderRNN(vocab_size, embed_size, hidden_size, pad_idx)

    def forward(self, src, tgt):
        encoder_outputs, (h, c) = self.encoder(src)
        h = h.squeeze(0)
        c = c.squeeze(0)
        outputs = []
        input_token = tgt[:, 0]
        for t in range(1, tgt.shape[1]):
            logits, h, c, _ = self.decoder(input_token, h, c, encoder_outputs)
            outputs.append(logits.unsqueeze(1))
            input_token = tgt[:, t]
        outputs = torch.cat(outputs, dim=1)
        return outputs

def text_to_tensor(text: str, max_len=MAX_LEN, vocab2idx=None) -> torch.Tensor:
    tokens = [vocab2idx[BOS_TOKEN]]
    for ch in text:
        if ch in vocab2idx:
            tokens.append(vocab2idx[ch])
        else:
            tokens.append(vocab2idx[PAD_TOKEN])
    tokens.append(vocab2idx[EOS_TOKEN])
    
    if len(tokens) < max_len:
        tokens += [vocab2idx[PAD_TOKEN]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return torch.tensor(tokens, dtype=torch.long)

def tensor_to_text(tensor: torch.Tensor, idx2vocab=None) -> str:
    chars = []
    for idx in tensor:
        token = idx2vocab[idx.item()]
        if token == BOS_TOKEN:
            continue
        if token == EOS_TOKEN or token == PAD_TOKEN:
            break
        chars.append(token)
    return "".join(chars)


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    vocab2idx = checkpoint["vocab2idx"]
    idx2vocab = checkpoint["idx2vocab"]
    model = Seq2Seq(
        vocab_size=len(vocab2idx),
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"],
        pad_idx=checkpoint["pad_idx"]
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab2idx, idx2vocab

def correct_word(model, vocab2idx, idx2vocab, word, max_len=MAX_LEN):
    src_tensor = text_to_tensor(word, max_len, vocab2idx).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, (h, c) = model.encoder(src_tensor)
        h = h.squeeze(0)
        c = c.squeeze(0)
    
    input_token = torch.tensor([vocab2idx[BOS_TOKEN]], device=DEVICE)
    decoded_indices = []
    
    for _ in range(max_len):
        with torch.no_grad():
            logits, h, c, _ = model.decoder(input_token, h, c, encoder_outputs)
            next_token = logits.argmax(1)  # (1,)
            decoded_indices.append(next_token.item())
            if next_token.item() == vocab2idx[EOS_TOKEN]:
                break
            input_token = next_token

    corrected_word = tensor_to_text(torch.tensor(decoded_indices), idx2vocab)
    return corrected_word


if __name__ == "__main__":
    MODEL_SAVE_PATH = "translit_model.pt"

    model, vocab2idx, idx2vocab = load_model(MODEL_SAVE_PATH)
    print("The model and dictionaries have been loaded successfully.")

    test_words = ["ucun", "gore", "dovlet", "bagli", "kisi"]
    for word in test_words:
        corrected = correct_word(model, vocab2idx, idx2vocab, word)
        print(f"Input: {word} -> Corrected: {corrected}")