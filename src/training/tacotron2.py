import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Helpers
# -------------------------
class Conv1dBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, C, T)
        return self.act(self.bn(self.conv(x)))

# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, conv_channels=256, n_convs=3, lstm_hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        convs = []
        in_ch = embed_dim
        for _ in range(n_convs):
            convs.append(Conv1dBNRelu(in_ch, conv_channels, kernel_size=5, padding=2))
            in_ch = conv_channels
        self.convs = nn.Sequential(*convs)
        self.birnn = nn.LSTM(conv_channels, lstm_hidden, batch_first=True, bidirectional=True)

    def forward(self, input_ids, input_lengths):
        # input_ids: (B, T_text)
        x = self.embedding(input_ids)  # (B, T, E)
        x = x.transpose(1, 2)  # (B, E, T)
        x = self.convs(x)      # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        # pack/unpack for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=True)
        outputs, _ = self.birnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs: (B, T, 2*lstm_hidden)
        return outputs  # memory for attention

# -------------------------
# Simple Bahdanau attention
# -------------------------
class Attention(nn.Module):
    def __init__(self, query_dim, memory_dim, attn_dim):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)
        self.memory_layer = nn.Linear(memory_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, memory, mask=None):
        # query: (B, query_dim) - current decoder hidden
        # memory: (B, T_mem, memory_dim)
        # mask: (B, T_mem) with 1 for valid positions
        q = self.query_layer(query).unsqueeze(1)          # (B,1,attn_dim)
        m = self.memory_layer(memory)                     # (B, T_mem, attn_dim)
        scores = self.v(torch.tanh(q + m)).squeeze(-1)    # (B, T_mem)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)      # (B, T_mem)
        context = torch.bmm(attn_weights.unsqueeze(1), memory).squeeze(1)  # (B, memory_dim)
        return context, attn_weights

# -------------------------
# Prenet
# -------------------------
class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128], dropout=0.5):
        super().__init__()
        layers = []
        in_sz = in_dim
        for sz in sizes:
            layers.append(nn.Linear(in_sz, sz))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_sz = sz
        self.prenet = nn.Sequential(*layers)

    def forward(self, x):
        return self.prenet(x)

# -------------------------
# Decoder (autoregressive)
# -------------------------
class Decoder(nn.Module):
    def __init__(self, n_mels=80, prenet_sizes=[256,128], attn_rnn_dim=512, decoder_rnn_dim=512, memory_dim=512):
        super().__init__()
        self.n_mels = n_mels
        self.prenet = Prenet(n_mels, prenet_sizes)
        self.attention_rnn = nn.LSTMCell(prenet_sizes[-1] + memory_dim, attn_rnn_dim)
        self.attention_layer = Attention(attn_rnn_dim, memory_dim, attn_dim=128)
        self.decoder_rnn = nn.LSTMCell(attn_rnn_dim + memory_dim, decoder_rnn_dim)
        self.mel_proj = nn.Linear(decoder_rnn_dim + memory_dim, n_mels)
        self.stop_proj = nn.Linear(decoder_rnn_dim + memory_dim, 1)

    def initialize_states(self, memory):
        B = memory.size(0)
        device = memory.device
        self.attn_hidden = torch.zeros(B, self.attention_rnn.hidden_size, device=device)
        self.attn_cell = torch.zeros(B, self.attention_rnn.hidden_size, device=device)
        self.dec_hidden = torch.zeros(B, self.decoder_rnn.hidden_size, device=device)
        self.dec_cell = torch.zeros(B, self.decoder_rnn.hidden_size, device=device)
        self.attention_weights = None
        self.attention_context = torch.zeros(B, memory.size(2), device=device)

    def forward(self, memory, memory_mask, mels=None, max_iters=1000, teacher_forcing_ratio=0.9):
        """
        memory: (B, T_mem, memory_dim)
        memory_mask: (B, T_mem) boolean (True where valid)
        mels: (B, n_mels, T_target) or None for inference
        Returns:
            mel_outputs: (B, n_mels, T_out)
            stop_outputs: (B, T_out)
        """
        B = memory.size(0)
        device = memory.device
        self.initialize_states(memory)

        # prepare inputs
        if mels is not None:
            # convert to (B, T, n_mels)
            mels_t = mels.transpose(1, 2)
            T_target = mels_t.size(1)
        else:
            mels_t = None
            T_target = 0

        mel_outputs = []
        stop_outputs = []
        attn_weights_all = []

        # initial input frame = zeros
        prev_mel = torch.zeros(B, self.n_mels, device=device)

        for t in range(max_iters):
            # prenet
            prenet_out = self.prenet(prev_mel)  # (B, prenet_out)
            # attention rnn input: prenet_out + previous context
            attn_in = torch.cat([prenet_out, self.attention_context], dim=-1)
            self.attn_hidden, self.attn_cell = self.attention_rnn(attn_in, (self.attn_hidden, self.attn_cell))
            # compute attention
            context, attn_weights = self.attention_layer(self.attn_hidden, memory, mask=memory_mask)
            # decoder rnn input: attn_hidden + context
            dec_in = torch.cat([self.attn_hidden, context], dim=-1)
            self.dec_hidden, self.dec_cell = self.decoder_rnn(dec_in, (self.dec_hidden, self.dec_cell))
            # project to mel and stop
            proj_input = torch.cat([self.dec_hidden, context], dim=-1)
            mel_frame = self.mel_proj(proj_input)    # (B, n_mels)
            stop_logit = self.stop_proj(proj_input).squeeze(-1)  # (B,)

            mel_outputs.append(mel_frame.unsqueeze(-1))  # keep time axis
            stop_outputs.append(stop_logit.unsqueeze(-1))
            attn_weights_all.append(attn_weights.unsqueeze(1))

            # decide next input (teacher forcing)
            if (mels_t is not None) and (torch.rand(1).item() < teacher_forcing_ratio) and (t < T_target):
                prev_mel = mels_t[:, t, :]  # teacher forcing
            else:
                prev_mel = mel_frame.detach()

            # stopping criterion (optionally) - if all stop_logits > 0 (sigmoid>0.5)
            if mels is None:
                probs = torch.sigmoid(stop_logit)
                if (probs > 0.5).all():
                    break

        mel_out = torch.cat(mel_outputs, dim=-1)  # (B, n_mels, T_out)
        stop_out = torch.cat(stop_outputs, dim=-1).squeeze(1) if len(stop_outputs) > 0 else torch.zeros(B,0, device=device)
        attn_all = torch.cat(attn_weights_all, dim=1) if len(attn_weights_all) > 0 else None
        return mel_out, stop_out, attn_all

# -------------------------
# Tacotron2 wrapper
# -------------------------
class Tacotron2(nn.Module):
    def __init__(self, vocab_size, n_mels=80, encoder_params=None, decoder_params=None):
        super().__init__()
        enc_p = encoder_params or {}
        dec_p = decoder_params or {}
        # encoder memory_dim must match decoder attention memory_dim
        self.encoder = Encoder(vocab_size, embed_dim=enc_p.get("embed_dim", 256),
                               conv_channels=enc_p.get("conv_channels", 256),
                               n_convs=enc_p.get("n_convs", 3),
                               lstm_hidden=enc_p.get("lstm_hidden", 256))
        # memory_dim is 2*lstm_hidden
        memory_dim = 2 * enc_p.get("lstm_hidden", 256)
        self.decoder = Decoder(n_mels=n_mels,
                               prenet_sizes=dec_p.get("prenet_sizes", [256,128]),
                               attn_rnn_dim=dec_p.get("attn_rnn_dim", 512),
                               decoder_rnn_dim=dec_p.get("decoder_rnn_dim", 512),
                               memory_dim=memory_dim)

    def forward(self, text_ids, text_lens, mels=None, teacher_forcing_ratio=0.9, max_iters=1000):
        """
        text_ids: (B, T_text) LongTensor, sorted by length desc
        text_lens: (B,) lengths
        mels: (B, n_mels, T) target mel for teacher forcing
        """
        # encoder
        memory = self.encoder(text_ids, text_lens)  # (B, T_mem, memory_dim)

        # mask for memory (True for valid)
        B, T_mem, _ = memory.size()
        device = memory.device
        # build mask from text_lens
        idxs = torch.arange(T_mem, device=device).unsqueeze(0).expand(B, -1)
        memory_mask = idxs < text_lens.unsqueeze(1)

        # decode
        mel_out, stop_out, attn = self.decoder(memory, memory_mask, mels=mels,
                                               max_iters=max_iters, teacher_forcing_ratio=teacher_forcing_ratio)
        return mel_out, stop_out, attn
