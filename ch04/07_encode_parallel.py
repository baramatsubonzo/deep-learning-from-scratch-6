import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import numpy as np
import shutil

import pickle
import regex as re
from tqdm import tqdm

from multiprocessing import Pool
from storybot.tokenizer import (
    pretokenize,
    count_pairs,
    merge,
    find_chunk_boundaries,
)

class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>"):
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))

        def get_merge_priority(pair):
            return self.merge_rules.get(pair, float('inf'))

        while len(ids) > 1:
            counts = count_pairs(ids)

            best_pair = min(counts, key=get_merge_priority)

            if best_pair not in self.merge_rules:
                break

            new_id = self.merge_rules[best_pair]
            ids = merge(ids, best_pair, new_id)
        return ids

    def encode(self, input_text, show_progress=False):
        pattern = '(' + re.escape(self.end_token) + ')'
        texts = re.split(pattern, input_text)
        all_ids = []

        texts = tqdm(texts, desc="Encoding") if show_progress else texts

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)
        return all_ids

    @staticmethod
    def load_from(filepath):
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)

    def _encode_chunk(self, args):
        file_path, start, end, cache_dir, chunk_idx = args

        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            ids = self.encode(chunk_text)

        cache_file = os.path.join(cache_dir, f"chunk_{chunk_idx:05d}.npy")
        np.array(ids, dtype=np.uint16).tofile(cache_file)

        return cache_file, len(ids)

    def encode_file(self, file_path, output_file, num_processes=8, num_chunks=64, cache_dir="bpe_cache"):
        os.makedirs(cache_dir, exist_ok=True)

        try:
            chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
            total_chunks = len(chunk_boundaries) - 1

            chunk_info_list = []
            for i in range(total_chunks):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                chunk_info_list.append((file_path, start, end, cache_dir, i))

            with Pool(processes=num_processes) as pool:
                cache_results = list(tqdm(
                    pool.imap(self._encode_chunk, chunk_info_list), 
                    total=len(chunk_info_list), 
                    desc="Encoding chunks"
                ))

            cache_files = [r[0] for r in cache_results]
            token_counts = [r[1] for r in cache_results]
            total_tokens = sum(token_counts)

            dtype = np.uint16
            arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(total_tokens,))

            idx = 0
            for cache_file in cache_files:
                chunk_data = np.fromfile(cache_file, dtype=dtype)
                arr[idx : idx + len(chunk_data)] = chunk_data
                idx += len(chunk_data)
            arr.flush()
            del arr
        finally:
            shutil.rmtree(cache_dir)

        return total_tokens

if __name__ == '__main__':
    tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

    tokenizer.encode_file(
        "storybot/tiny_stories_train.txt",
        "storybot/tiny_stories_train.bin", num_processes=8
    )

    tokenizer.encode_file(
        "storybot/tiny_stories_valid.txt",
        "storybot/tiny_stories_valid.bin", num_processes=8
    )

