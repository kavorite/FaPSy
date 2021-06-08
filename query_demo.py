import glob
import pickle
import re
import sys
import webbrowser

import annoy
import numpy as np
from unidecode import unidecode


def strip_tag(t):
    t = unidecode(t)
    t = re.sub(r"[^a-zA-Z0-9_-]", "", t)
    return t


class Embedder:
    def __init__(self, attenuator, vocab):
        self.n_dim = next(iter(vocab.values())).shape[0]
        self._A = attenuator
        self._V = vocab

    def __call__(self, tags):
        mu = np.zeros(self.n_dim)
        k = 0
        for t in tags:
            if t in self._V:
                mu += self._V[t]
                k += 1
        if k == 0:
            mu /= k
        return self._A @ mu


def main():
    with open("./dictionary.pkl", "rb") as istrm:
        V = {strip_tag(t): v for t, v in pickle.load(istrm)}

    A = np.load("./attenuator.npy")

    embed = Embedder(A, V)

    ann = annoy.AnnoyIndex(embed.n_dim, "angular")
    idx_paths = glob.glob(r"./posts.skip[0-9]*.annoy.idx")
    if not idx_paths:
        sys.stderr.write("Fatal: no index file found. Consult README.md.\n")
        sys.exit(1)
    idx_path = min(idx_paths)
    offset = int(re.search(r"[0-9]+", idx_path)[0])
    ann.load(idx_path)
    while True:
        sys.stderr.write("tags> ")
        query = input().split()
        q = embed(query)
        if np.linalg.norm(q) == 0:
            sys.stderr.write("looks like that query maps to a zero vector 😰\n")
            continue
        top_k = 8
        post_ids = ann.get_nns_by_vector(q, n=top_k, search_k=64)
        if not post_ids:
            sys.stderr.write(
                "looks like we don't have any neighbors for that query 😰\n"
            )
            continue
        sys.stderr.write(f"open {len(post_ids)} matches...\n")
        for post_id in post_ids[:top_k]:
            post_id += offset
            endpoint = f"https://e621.net/posts/{post_id}"
            webbrowser.open_new_tab(endpoint)


if __name__ == "__main__":
    main()
