import argparse
import csv
import gzip
import io
import pickle

import annoy
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=1_700_000)
    parser.add_argument("--dict-file", type=np.load, default="./dictionary.npz")
    parser.add_argument("--attenuator", type=np.load, default="./attenuator.npy")
    parser.add_argument("--tree-file", default=None)
    args = parser.parse_args()
    index_name = args.tree_file or f"./posts.skip{args.offset}.annoy.idx"
    dictionary = dict(zip(args.dict_file["tags"], args.dict_file["vecs"]))
    args.dict_file.close()
    n_dim = next(iter(dictionary.values())).shape[0]

    print("embed posts...")
    ann = annoy.AnnoyIndex(f=n_dim, metric="angular")
    csv.field_size_limit(1 << 20)

    ann.on_disk_build(index_name)
    with gzip.open("./posts.csv.gz") as istrm:
        istrm = io.TextIOWrapper(istrm, encoding="utf8")
        posts = csv.DictReader(istrm)
        print(f"skip first {args.offset}")
        for i, _ in enumerate(tqdm(posts, total=args.offset)):
            if i == args.offset:
                break
        print("embed posts...")
        A = args.attenuator
        for post in tqdm(posts, total=2_800_000 - args.offset):
            if post["is_deleted"] == "t":
                continue
            mu = np.zeros(n_dim)
            k = 0
            for t in post["tag_string"].split():
                if t in dictionary:
                    mu += dictionary[t]
                    k += 1
            if k != 0:
                mu /= k
                mu = A @ mu
                ann.add_item(int(post["id"]) - args.offset, mu)
    print("constructing index...")
    ann.build(n_trees=16)
    print(f"index written to {index_name}")


if __name__ == "__main__":
    main()
