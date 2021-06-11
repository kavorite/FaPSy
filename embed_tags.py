import argparse
import csv
import gzip
import io

import numpy as np
from tqdm import tqdm


def tag_hit_generator(
    tagset, saturation_hits=8, goal_saturation=0.9, print_progress=True
):
    tagset = set(tagset)
    with gzip.open("./db_export/posts.csv.gz") as istrm:
        posts = csv.DictReader(io.TextIOWrapper(istrm, encoding="utf8"))
        goal = {t: saturation_hits for t in tagset}
        with tqdm(
            total=saturation_hits * len(tagset), disable=not print_progress
        ) as progress:
            for post in posts:
                tags = post["tag_string"].split()
                yield post
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


def tags_by_freq(k=-1):
    csv.field_size_limit(1 << 20)
    with gzip.open("./db_export/tags.csv.gz") as istrm:
        istrm = io.TextIOWrapper(istrm, encoding="utf8")
        all_tags = list()
        for tag in csv.DictReader(istrm):
            if tag["category"] not in "045":
                continue
            t = tag["name"]
            if not t.isprintable():
                continue
            all_tags.append(tag)
        all_tags.sort(key=lambda tag: int(tag["post_count"]))
        all_tags.reverse()
    return all_tags[:k]


class TagCoocs:
    def __init__(self, tags, n_dim):
        self.vocab = {t: i for i, t in enumerate(tags)}
        self.coocs = np.zeros((len(tags), len(tags)))
        self.n_dim = n_dim

    def tally(self, post):
        tags = post["tags"].split()
        for t in tags:
            if t not in self.vocab:
                continue
            for w in tags:
                if w not in self.vocab:
                    continue
                i, j = self.vocab[t], self.vocab[w]
                self.coocs[i, j] += 1

    def embed(self):
        k = self.n_dim
        C = self.coocs
        C /= (C.max(axis=-1) + 1e-16)[:, None]
        U, s, _ = np.linalg.svd(C, full_matrices=False, hermitian=True)
        S = np.diag(s)[..., :k]
        D = U @ S
        return D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-top", type=int, default=2048)
    parser.add_argument("--n-dim", type=int, default=64)
    parser.add_argument("--dump-coocs", action="store_true", default=False)
    args = parser.parse_args()
    print("generate cooccurrence matrix...")
    coocs = TagCoocs(tags_by_freq(args.k_top), args.n_dim)
    print("find post candidates...")
    for post in tag_hit_generator(coocs.vocab):
        coocs.tally(coocs, post)

    print("factorize...")
    D = coocs.embed()
    tags, indx = zip(*coocs.vocab.items())
    vecs = D[np.array(indx)]
    tags = np.array(tags)
    np.savez("./index/dictionary.npz", tags=tags, vecs=vecs, allow_pickle=False)
    print("embeddings written to ./index/dictionary.npz")


if __name__ == "__main__":
    main()
