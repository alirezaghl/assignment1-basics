import regex as re
import pickle

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.vocab_reversed = {v: k for k, v in vocab.items()}
        
        next_id = max(vocab.keys()) + 1 if vocab else 256
        for special in self.special_tokens:
            special_bytes = special.encode('utf-8')
            if special_bytes not in self.vocab_reversed:
                self.vocab[next_id] = special_bytes
                self.vocab_reversed[special_bytes] = next_id
                next_id += 1
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = '|'.join(re.escape(tok) for tok in sorted_special)
            parts = re.split(f'({pattern})', text)
        else:
            parts = [text]
        
        byte_special_tokens = [token.encode('utf-8') for token in self.special_tokens]
        pretokens = []  
        
        for part in parts:
            if not part:
                continue
            
            part_bytes = part.encode('utf-8')
            
            if part_bytes in byte_special_tokens:
                pretokens.append([self.vocab_reversed[part_bytes]])
                continue
            
            matches = re.findall(PAT, part)
            
            for match in matches:
                pretoken = []
                for b in match.encode('utf-8'):
                    pretoken.append(self.vocab_reversed[bytes([b])])
                pretokens.append(pretoken)
        
        for i, pretoken in enumerate(pretokens):
            for merge in self.merges:
                merged_bytes = merge[0] + merge[1]
                new_index = self.vocab_reversed[merged_bytes]
                
                new_pretoken = []
                j = 0
                while j < len(pretoken):
                    if (j < len(pretoken) - 1 and 
                        (self.vocab[pretoken[j]], self.vocab[pretoken[j+1]]) == merge):
                        new_pretoken.append(new_index)
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1
                pretoken = new_pretoken
            
            pretokens[i] = pretoken
        
        tokens = [token for pretoken in pretokens for token in pretoken]
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8', errors='replace')
        return text
    
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
