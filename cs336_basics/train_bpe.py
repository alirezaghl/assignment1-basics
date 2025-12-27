import os
import time
import pickle
import regex as re
from multiprocessing import Pool
import multiprocessing as mp
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_file_chunk(args: tuple) -> dict[tuple[int, ...], int]:
    input_path, start, end, special_tokens = args
    
    chunk_counts = {}
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    chunk_text = chunk_text.replace('\r\n', '\n')
    
    pattern = '|'.join(re.escape(tok) for tok in special_tokens)
    parts = re.split(pattern, chunk_text)  
    
    for part in parts:       
        matches = re.findall(PAT, part)     
        for match in matches:
            token_tuple = tuple(match.encode('utf-8'))
            chunk_counts[token_tuple] = chunk_counts.get(token_tuple, 0) + 1
    
    return chunk_counts


def pretokenize_parallel(input_path: str, special_tokens: list[str]) -> dict:

    num_processes = 8
        
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, special_tokens[0].encode("utf-8")
        )
    
    task_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(pretokenize_file_chunk, task_args)
    
    pretokenize = {}
    for chunk_dict in results:
        for token, count in chunk_dict.items():
            pretokenize[token] = pretokenize.get(token, 0) + count
    
    return pretokenize


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    initial_vocab_size = 256 + len(special_tokens)
    
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for idx, special in enumerate(special_tokens):
        vocab[256 + idx] = special.encode('utf-8')
    
    pretokenize = pretokenize_parallel(input_path, special_tokens)
    
    merges = []
    num_merges_needed = vocab_size - initial_vocab_size
    next_id = 256 + len(special_tokens)
    
    for _ in range(num_merges_needed):
        pair_counts = {}
        for token_seq, count in pretokenize.items():
            for j in range(len(token_seq) - 1):
                pair = (token_seq[j], token_seq[j+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        
        if not pair_counts:
            break
        
        most_frequent_pair = max(
            pair_counts.items(),
            key=lambda x: (
                x[1],  
                vocab[x[0][0]],  
                vocab[x[0][1]] 
            )
        )[0]
        
        merges.append(most_frequent_pair)
        
        new_token = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]
        vocab[next_id] = new_token
        
        new_pretokenize = dict()
        for token_seq, count in pretokenize.items():
            new_ids = list(token_seq)
            j = 0   
            while j < len(list(new_ids)) - 1:
                pair = (new_ids[0], new_ids[j+1])
                if pair == most_frequent_pair:
                    new_ids[j] = next_id
                    new_ids.pop(j+1)
                else:
                    j+=1


           
        
        pretokenize = new_pretokenize
        next_id += 1
    
    merged = [(vocab[p0], vocab[p1]) for p0, p1 in merges]
    
    return vocab, merged