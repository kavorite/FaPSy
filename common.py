import glob
import json
import os
import re
import urllib.error
import urllib.request
from typing import Iterable

import annoy
import cv2
import numpy as np
import scipy.fftpack


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
    def __init__(
        self,
        dictionary: dict,
        attenuator: np.ndarray = None,
        recognizer: np.ndarray = None,
    ):
        n = None
        embeddings = iter(dictionary.values())
        n = next(embeddings).shape[0]
        self.n_dim = n
        self.A = attenuator
        self.R = recognizer
        self.V = dictionary

    def embed_img(self, img_str, highfreq_factor=4):
        if self.recognizer is None:
            status = """
            cannot map images to label space without pretrained recognizer
            """
            raise ValueError(" ".join(status.split()))
        hash_size = int(np.ceil(np.sqrt(self.n_dim ** 2)))
        img_str = np.frombuffer(img_str, dtype=np.uint8)
        img = cv2.imdecode(img_str, cv2.IMREAD_GRAY)
        dim = hash_size * highfreq_factor
        img = cv2.resize(img, (dim, dim))
        dct = scipy.fftpack.dct(scipy.fftpack.dct(img, axis=0), axis=1)
        lo = dct[:hash_size, :hash_size]
        return lo.flatten()

    def embed_tags(self, tags, weights=None):
        v = np.zeros(self.n_dim)
        k = 0
        weights = weights or np.ones(self.n_dim)
        for t, s in zip(tags, weights):
            if t in self.V:
                v += s * self.V[t]
                k += 1
        if k != 0:
            v /= k
        if self.A is not None:
            v = self.A @ v
        v *= weights
        return v.flatten()

    def __call__(self, tags_or_image):
        if isinstance(tags_or_image, bytes):
            return self.embed_img(tags_or_image)
        elif isinstance(tags_or_image, str):
            return self.embed_tags(tags_or_image.split())
        elif isinstance(tags_or_image, Iterable):
            return self.embed_tags(tags_or_image)
        else:
            status = """
                inputs to embed() must be tagsets or image content in a
                cv2-supported format
                """
            raise TypeError(" ".join(status.split()))


class PostGraph:
    def __init__(self, index: annoy.AnnoyIndex, embed: Embedder, offset: int = 0):
        self.index = index
        self.embed = embed
        self.offset = offset

    @classmethod
    def load_path(
        cls, root: str, index_path=None, idx_offset=None, dictpath=None, attnpath=None
    ):
        root = os.path.abspath(root)
        dictpath = dictpath or os.path.join(root, "dictionary.npz")
        dictfile = np.load(dictpath)
        V = dict(zip(dictfile["tags"], dictfile["vecs"]))
        attnpath = attnpath or os.path.join(root, "attenuator.npy")
        A = np.load(attnpath)
        embed = Embedder(A, V)
        index_path = index_path or search_index(root)
        if not index_path:
            raise ValueError(f"no index files found in")
        offset = idx_offset or index_offset(index_path or "") or 0
        index = annoy.AnnoyIndex(embed.n_dim, metric="angular")
        index.load(index_path)
        return cls(index, embed, offset)

    def neighbors(self, query, n, *args, **kwargs):
        q = self.embed(query)
        ids, *rest = self.index.get_nns_by_vector(q, n, *args, **kwargs)
        ids = np.array(ids) + self.offset
        return (ids, *rest) if rest else ids

    def centroids(self, query_nodes, return_weights=True):
        neighborhood = np.array(sorted(set(query_nodes)))
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
