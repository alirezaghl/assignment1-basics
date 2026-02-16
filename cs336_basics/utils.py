import torch
import torch.nn as nn
import einops
import math
from typing import Optional
from collections.abc import Callable, Iterable


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets):
        maxe, _ = torch.max(inputs, dim=-1, keepdim=True)
        inputs = inputs - maxe
        selected_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))
        loss = -selected_logits.squeeze(-1) + torch.logsumexp(inputs, dim=-1)
        return loss.mean()
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f'invalid learning rate: {lr}')
        
        defaults = {
            'lr': lr,             
            'betas': betas,
            'weight_decay': weight_decay,  
            'eps': eps             
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']          
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']  
            eps = group['eps']        
        
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['t'] = 1
                    state['m'] = torch.zeros_like(p.data)  
                    state['v'] = torch.zeros_like(p.data)  
                
                t = state['t']
                m = state['m']
                v = state['v']
                
                m = beta1 * m + (1 - beta1) * grad 
                v = beta2 * v + (1 - beta2) * grad**2
                
                m_prime = m / (1 - beta1**t)
                v_prime = v / (1 - beta2**t)

                p.data -= lr * m_prime / (torch.sqrt(v_prime) + eps)
                p.data -= lr * weight_decay * p.data
                
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        
        return loss

def learning_rate_schedule(t, amax, amin, tau_w, tau_c):
    
    if t < tau_w:
        lr = (t / tau_w) * amax
        return lr
    
    if tau_w <= t <= tau_c:
        lr = amin + 0.5 * (1 + math.cos((t - tau_w) / (tau_c - tau_w) * math.pi)) * (amax - amin)
        return lr
    
    if t > tau_c:
        lr = amin
        return lr

def gradient_clipping(parameters, max_l2_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            norm = torch.linalg.norm(p.grad.data)
            total_norm += norm**2
    
    total_norm = math.sqrt(total_norm)

    if total_norm <= max_l2_norm:
        return 
    
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    