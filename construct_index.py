import argparse
import csv
import gzip
import io
import pickle

import annoy
import numpy as np
from tqdm import tqdm

from common import Attenuator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=1_700_000)
    parser.add_argument("--dict-file", type=np.load, default="./index/dictionary.npz")
    parser.add_argument("--attenuator", type=np.load, default="./index/attenuator.npy")
    parser.add_argument("--tree-file", default=None)
    args = parser.parse_args()
    index_name = args.tree_file or f"./index/posts.skip{args.offset}.annoy.idx"
    vocab = dict(zip(args.dict_file["tags"], args.dict_file["vecs"]))
    embed = Attenuator(vocab, attenuator=args.attenuator)
    args.dict_file.close()

    print("embed posts...")
    ann = annoy.AnnoyIndex(f=embed.n_dim, metric="angular")
    csv.field_size_limit(1 << 20)

    ann.on_disk_build(index_name)
    with gzip.open("./db_export/posts.csv.gz") as istrm:
        istrm = io.TextIOWrapper(istrm, encoding="utf8")
        posts = csv.DictReader(istrm)
        print(f"skip first {args.offset}")
        for i, _ in enumerate(tqdm(posts, total=args.offset)):
            if i == args.offset:
                break
        print("embed posts...")
        for post in tqdm(posts, total=2_800_000 - args.offset):
            if post["is_deleted"] == "t":
                continue
            x = embed(post["tag_string"].split())
            ann.add_item(int(post["id"]) - args.offset, x)
    print("constructing index...")
    ann.build(n_trees=16)
    print(f"index written to {index_name}")


if __name__ == "__main__":
    main()
