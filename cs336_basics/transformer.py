import torch
import torch.nn as nn
import einops

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()
        self.bias = None

    def reset_parameters(self):
        std = torch.sqrt(torch.tensor(2/(self.in_features+self.out_features)))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x):
        return einops.einsum(x, self.weight, '... d_in,  d_out d_in -> ... d_out')
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)
    
    def forward(self, x):
        return self.embedding[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x / rms * self.gain
        return result.to(in_dtype)

class SWIGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
    
    def forward(self, x):
        hidden_1 = x @ self.w1.T # ...*dff
        hidden_3 = x @ self.w3.T # ...*dff
        silu = hidden_1 * torch.sigmoid(hidden_1) #...*dff
        gated_hidden = silu * hidden_3 
        out = gated_hidden @ self.w2.T
        return out


class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        pos = torch.arange(0, max_seq_len, device=device)
        i = torch.arange(0, d_k // 2, device=device)
        inverse_freq = theta ** (-2 * i / d_k)
        angles = torch.einsum("p, f -> p f", pos, inverse_freq)
        self.register_buffer("cos_cached", torch.cos(angles))
        self.register_buffer("sin_cached", torch.sin(angles))

    def forward(self, x, token_positions):
        seq_len = token_positions.shape[-1]
        cos_half = self.cos_cached[:seq_len, :] 
        sin_half = self.sin_cached[:seq_len, :]
        cos_half, sin_half = map(lambda t: einops.rearrange(t, 's d -> 1 s d'), [cos_half, sin_half])
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x1_rotated = x1 * cos_half - x2 * sin_half
        x2_rotated = x1 * sin_half + x2 * cos_half
        result = torch.stack([x1_rotated, x2_rotated], dim=-1)   
        result = result.flatten(-2)  
        return result

class Softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        maxe, _ = torch.max(x, self.dim, keepdim=True)
        x = x - maxe
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, self.dim, True)

class SDPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(-1)
    
    def forward(self, q, k, v, mask=None):
        dq = q.shape[-1]
        qk = einops.einsum(q, k, '... sq dq, ... sk dq -> ... sq sk')
        qk = qk / torch.sqrt(torch.tensor(dq))
        if mask is not None:
            qk = qk.masked_fill(~mask, float('-inf'))

        attention = einops.einsum(self.softmax(qk), v, '... sq sk, ... sk dv -> ... sq dv')
        return attention

class MHA(nn.Module):
    def __init__(self, d_model, 
                 num_heads, 
                 max_seq_len, 
                 theta, 
                 q_proj_weight,
                 k_proj_weight, 
                 v_proj_weight, 
                 o_proj_weight):
        super().__init__()
        self.num_heads = num_heads
        dk = d_model // num_heads
        self.rope = RoPE(theta, dk, max_seq_len)
        self.sdpa = SDPA()
        self.wq = nn.Parameter(q_proj_weight)
        self.wk = nn.Parameter(k_proj_weight)
        self.wv = nn.Parameter(v_proj_weight)
        self.wo = nn.Parameter(o_proj_weight)
    
    def forward(self, x, token_positions=None):
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)

        q = x @ self.wq.T
        q = einops.rearrange(q, '... s (h dk) -> ... h s dk', h=self.num_heads)
        q = self.rope(q, token_positions)

        k = x @ self.wk.T
        k = einops.rearrange(k, '... s (h dk) -> ... h s dk', h=self.num_heads)
        k = self.rope(k, token_positions)

        v = x @ self.wv.T
        v = einops.rearrange(v, '... s (h dv) -> ... h s dv', h=self.num_heads)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
        attn = self.sdpa(q, k, v, mask)

        output = einops.rearrange(attn, '... h s dv -> ... s (h dv)')
        output = output @ self.wo.T

        return output

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights):
        super().__init__()
        q_proj_weight = weights['attn.q_proj.weight']
        k_proj_weight = weights['attn.k_proj.weight']
        v_proj_weight = weights['attn.v_proj.weight']
        o_proj_weight = weights['attn.output_proj.weight']
        self.mha = MHA(d_model, num_heads, max_seq_len, theta, 
                       q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
        self.ln1 = RMSNorm(d_model, 1e-5)
        self.ln1.gain.data = weights['ln1.weight']
        self.ln2 = RMSNorm(d_model, 1e-5)
        self.ln2.gain.data = weights['ln2.weight']
        self.ffn = SWIGLU(d_model, d_ff)
        self.ffn.w1.data = weights['ffn.w1.weight']
        self.ffn.w2.data = weights['ffn.w2.weight']
        self.ffn.w3.data = weights['ffn.w3.weight']
    
    def forward(self, x):
       y = x + self.mha(self.ln1(x))
       out = y + self.ffn(self.ln2(y))
       return out
    

class Transformer(nn.Module):
    def __init__(
            self, 
            vocab_size,
            context_length,
            d_model, 
            num_layers, 
            num_heads,
            d_ff,
            rope_theta,
            weights):
        super().__init__()
        
        self.embedding = Embedding(vocab_size, d_model)
        self.embedding.embedding.data = weights['token_embeddings.weight']
        
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_weights = {  
                k.replace(f'layers.{layer_idx}.', ''): v 
                for k, v in weights.items() 
                if f'layers.{layer_idx}.' in k
            }
            block = Block(d_model, num_heads, d_ff, context_length, rope_theta, layer_weights)
            self.layers.append(block)
        
        self.ln_final = RMSNorm(d_model, eps=1e-5)
        self.ln_final.gain.data = weights['ln_final.weight']  
        
        self.head = Linear(d_model, vocab_size)
        self.head.weight.data = weights['lm_head.weight']  
    
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x) 
        
        x = self.ln_final(x)
        logits = self.head(x)
        
        return logits