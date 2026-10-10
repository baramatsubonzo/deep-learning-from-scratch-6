import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from collections import defaultdict

import regex as re
import pickle
from multiprocessing import Pool
from tqdm import tqdm

def count_pairs(ids, weight=1, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

def pretokenize(text):
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)

def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        texts = chunk_text.split(end_token)

        for text in texts:
            for pretoken in pretokenize(text):
                pretoken_counts[pretoken] += 1
        return pretoken_counts

def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:  # ファイルをバイナリモードで開く
        # ファイルサイズを取得
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks

        # チャンクの開始位置を計算（等間隔）
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)  # 最後にファイル終端を追加

        buffer_size = 4096  # 境界から先読みするバイト数

        # 境界位置の調整（終了トークンを探す）
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)  # 境界の推定位置から開始

            while True:
                buffer = file.read(buffer_size)  # バッファサイズ分を読む

                # ファイル終端に達した場合
                if buffer == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # 読み取ったチャンクで終了トークンを検索
                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    # 見つかった場合、その位置を新しい境界とする
                    chunk_boundaries[bi] = chunk_position + end_position
                    break

                # 見つからなかった場合、次のバッファ位置に移動
                chunk_position += buffer_size

    # 重複を除去し、ソートして返す
    return sorted(set(chunk_boundaries))

def train_bpe(file_path, vocab_size, end_token="<|endoftext|>", num_processes=8, num_chunks=64):
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    total_chunks = len(chunk_boundaries) - 1

    chunk_info_list = []
    for i in range(total_chunks):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i+1]
        chunk_info_list.append((file_path, start, end, end_token))

    with Pool(processes=num_processes) as pool:
        all_results = list(tqdm(
            pool.imap(pretoken_chunk, chunk_info_list),
            total = len(chunk_info_list),
            desc="Pretokenizing"
        ))


    pretoken_counts = defaultdict(int)

    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    ids_counts = {
        tuple(pretoken.encode("utf-8")): count
        for pretoken, count in pretoken_counts.items()
    }

    num_merges = vocab_size - 256 - 1
    merge_rules = {}
    pair_to_ids = defaultdict(set)

    pair_counts = defaultdict(int)

    for ids, count in ids_counts.items():
        count_pairs(ids, count, pair_counts)
        for pair in zip(ids, ids[1:]):
            pair_to_ids[pair].add(ids)

    for step in tqdm(range(num_merges), desc="Training BPE"):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        affected_ids = pair_to_ids[best_pair]
        del pair_to_ids[best_pair]

        for ids in affected_ids:
            ids_count = ids_counts[tuple(ids)]
            new_ids = merge(ids, best_pair, new_id)

            del ids_counts[tuple(ids)]
            ids_counts[tuple(new_ids)] = ids_count
            old_counts = count_pairs(ids)
            for pair, count in old_counts.items():
                pair_counts[pair] -= count * ids_count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_ids[pair].discard(tuple(ids))

            new_counts = count_pairs(new_ids)
            for pair, count in new_counts.items():
                pair_counts[pair] += count * ids_count
                pair_to_ids[pair].add(tuple(new_ids))

    return merge_rules

# vocab_size = 1000
# file_path = "codebot/tiny_codes.txt"
# text = open(file_path).read()
# merge_rules = train_bpe(text, vocab_size)

if __name__ == '__main__':
    vocab_size = 10000
    file_path = "storybot/tiny_stories_train.txt"

    merge_rules = train_bpe(file_path, vocab_size, num_processes=8)

    with open("storybot/merge_rules.pkl", "wb") as f:
        pickle.dump(merge_rules, f)