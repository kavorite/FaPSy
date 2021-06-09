import glob
import json
import os
import re
import urllib.error
import urllib.request

import annoy
import numpy as np


class E621Error(Exception):
    pass


def index_offset(index_path):
    match = re.search(r"[0-9]+", index_path)
    if match:
        return int(match[0])


def search_index(root):
    idx_paths = glob.glob(f"{root}/posts.skip[0-9]*.annoy.idx")
    idx_paths += glob.glob(f"{root}/*.annoy.idx")
    idx_paths = np.array(idx_paths, dtype=object)
    if not idx_paths:
        return
    scores = []
    for idx in idx_paths:
        score = 1.0
        score *= np.log(os.stat(idx).st_mtime)
        score *= index_offset(idx) or 1.0
        scores.append(score)
    return idx_paths[np.argmin(scores)]


def fetch_post(post_id):
    req = urllib.request.Request(f"https://e621.net/posts/{post_id}.json")
    req.add_header("User-Agent", "e6graph by kavorite <iseurie@gmail.com>")
    with urllib.request.urlopen(req) as rsp:
        payload = json.load(rsp)
        try:
            return payload["post"]
        except urllib.error.HTTPError as err:
            raise E621Error(payload) from err


def pagerank(A, alpha=0.85):
    A_hat = A.sum(axis=-1)[:, None]
    A_hat = alpha * A_hat + (1 - alpha) / A_hat.shape[0]
    eigvals, eigvecs = np.linalg.eigh(A)
    centralities = eigvecs[np.argmax(eigvals)]
    return centralities


class Embedder:
    def __init__(self, attenuator: np.ndarray, vocab: dict):
        n = None
        embeddings = iter(vocab.values())
        n = next(embeddings).shape[0]
        self.n_dim = n
        self.A = attenuator
        self.V = vocab

    def embed_tags(self, tags):
        v = np.zeros(self.n_dim)
        k = 0
        for t in tags:
            if t in self.V:
                v += self.V[t]
                k += 1
        if k == 0:
            v /= k
        return self.A @ v

    def __call__(self, tags):
        return self.embed_tags(tags)


class PostGraph:
    def __init__(self, index: annoy.AnnoyIndex, embed: Embedder, offset: int = 0):
        self.index = index
        self.embed = embed
        self.offset = offset

    @classmethod
    def load_path(
        cls, root: str, index_path=None, idx_offset=None, dictpath=None, attnpath=None
    ):
        dictpath = dictpath or os.path.join(os.path.abspath(root), "dictionary.npz")
        dictfile = np.load(dictpath)
        V = dict(zip(dictfile["tags"], dictfile["vecs"]))
        attnpath = attnpath or os.path.join(os.path.abspath(root), "attenuator.npy")
        A = np.load(attnpath)
        embed = Embedder(A, V)
        index_path = index_path or search_index(root)
        if not index_path:
            raise ValueError(f"no index files found in")
        offset = idx_offset or index_offset(index_path or "") or 0
        index = annoy.AnnoyIndex(embed.n_dim, metric="angular")
        index.load(index_path)
        return cls(index, embed, offset)

    def tag_neighbors(self, tags, n, *args, **kwargs):
        q = self.embed(tags)
        ids = self.index.get_nns_by_vector(q, n, *args, **kwargs)
        return np.array(ids) + self.offset

    def post_neighbors(self, post_id, n, *args, **kwargs):
        try:
            ids = self.index.get_nns_by_item(post_id, n, *args, **kwargs)
            return np.array(ids) + self.offset
        except IndexError:
            post = fetch_post(post_id)
            return self.tag_neighbors(self, post["tag_string"])

    def post_centroids(self, post_ids, return_weights=True):
        neighborhood = np.array(sorted(set(post_ids)))
        n = len(neighborhood)
        A = np.zeros((n, n))
        I = {p: i for i, p in enumerate(neighborhood)}
        for q in neighborhood:
            for p in neighborhood:
                cost = self.index.get_distance(p, q)
                A[I[p]] = cost
                A[I[q]] = cost
        centralities = pagerank(A) + 1e-16
        centralities -= centralities.min()
        centralities /= centralities.max()
        by_centrality = np.argsort(centralities)[::-1]
        if return_weights:
            return neighborhood[by_centrality], centralities
        else:
            return neighborhood[by_centrality]
