import argparse
import csv
import gzip
import io

import numpy as np
from tqdm import tqdm


def tag_hit_generator(tagset, saturation_hits=8, goal_saturation=0.9):
    tagset = set(tagset)
    with gzip.open("./posts.csv.gz") as istrm:
        posts = csv.DictReader(io.TextIOWrapper(istrm, encoding="utf8"))
        goal = {t: saturation_hits for t in tagset}
        with tqdm(total=saturation_hits * len(tagset)) as progress:
            for post in posts:
                tags = post["tag_string"].split()
                yield tags
                saturation = 1 - len(goal) / len(tagset)
                if saturation > goal_saturation:
                    return
                for t in tags:
                    if t not in goal:
                        continue
                    progress.update()
                    if t in goal:
                        goal[t] -= 1
                    if goal[t] == 0:
                        del goal[t]
                del tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-top", type=int, default=2048)
    parser.add_argument("--k-dim", type=int, default=64)
    parser.add_argument("--dump-coocs", action="store_true", default=False)
    args = parser.parse_args()
    csv.field_size_limit(1 << 20)
    with gzip.open("./tags.csv.gz") as istrm:
        print("read tags...")
        istrm = io.TextIOWrapper(istrm, encoding="utf8")
        all_tags = list(csv.DictReader(istrm))

    all_tags.sort(key=lambda tag: int(tag["post_count"]))
    all_tags.reverse()

    tag_idx = dict()
    # tag_idf = np.zeros(args.k_top)
    for tag in all_tags:
        if tag["category"] not in "045":
            continue
        t = tag["name"]
        if not t.isprintable():
            continue
        i = len(tag_idx)
        tag_idx[t] = i
        if len(tag_idx) == args.k_top:
            break

    print("generate cooccurrence matrix...")
    C = np.zeros((args.k_top, args.k_top))
    for tags in tag_hit_generator(tag_idx):
        for t in tags:
            if t not in tag_idx:
                continue
            for w in tags:
                if w not in tag_idx:
                    continue
                i, j = tag_idx[t], tag_idx[w]
                C[i, j] += 1

    print("factorize...")
    C /= (C.max(axis=-1) + 1e-16)[:, None]
    U, _, _ = np.linalg.svd(C, full_matrices=False, hermitian=True)

    tags, indx = zip(*tag_idx.items())
    vecs = U[np.array(indx)]
    tags = np.array(tags)
    np.savez("./dictionary.npz", tags=tags, vecs=vecs, allow_pickle=False)
    print("embeddings written to dictionary.npz")


if __name__ == "__main__":
    main()
