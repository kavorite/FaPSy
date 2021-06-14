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


def pagerank(A, alpha=0.85):
    A_hat = A.sum(axis=-1)[:, None]
    A_hat = alpha * A_hat + (1 - alpha) / A_hat.shape[0]
    eigvals, eigvecs = np.linalg.eigh(A)
    centralities = eigvecs[np.argmax(eigvals)]
    return centralities


class EmbeddingObjective:
    def __init__(self, dictfile=np.load("./index/dictionary.npz")):
        self.tags = dictfile["tags"]
        self.frqs = dict(zip(self.tags, dictfile["frqs"]))
        self.vecs = dict(zip(self.tags, dictfile["vecs"]))

    def embed(self, tags):
        tags = [t for t in tags if t in self.vecs]
        if not tags:
            return None
        frqs = [self.frqs[t] for t in tags]
        return self.vecs[tags[np.argmin(frqs)]]

    def __call__(self, tags):
        return self.embed(tags)


class Recognizer:
    def __init__(self, thumb_dim, highfreq_factor, recognizer: np.ndarray = None):
        self.thumb_dim = thumb_dim
        self.highfreq_factor = highfreq_factor
        self.R = recognizer

    def embed(self, img_str):
        d = self.thumb_dim
        dim = np.round(d * self.highfreq_factor).astype(int)
        img = np.frombuffer(img_str, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (dim, dim))
        dct = scipy.fftpack.dct(scipy.fftpack.dct(img, axis=0), axis=1)
        low = dct[:d, :d]
        x = low.flatten()
        if self.R is not None:
            x = self.R @ x
        return x

    def phash(self, img_str):
        x = self.embed(img_str)
        mask = x > np.median(x)
        return mask

    def __call__(self, img_str):
        return self.embed(img_str)


class Attenuator:
    def __init__(
        self,
        dictionary: dict,
        attenuator: np.ndarray = None,
    ):
        self.n_dim = min(attenuator.shape)
        self.A = attenuator
        self.V = dictionary

    def embed(self, tags, weights=None):
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

    def __call__(self, tags, weights=None):
        if isinstance(tags, str):
            tags = tags.split()
        return self.embed(tags, weights)


class PostGraph:
    @staticmethod
    def index_offset(index_path):
        index_name = os.path.basename(index_path)
        match = re.search(r"[0-9]+", index_name)
        if match:
            return int(match[0])

    @classmethod
    def search_index(cls, root):
        idx_paths = glob.glob(f"{root}/posts.skip[0-9]*.annoy.idx")
        idx_paths += glob.glob(f"{root}/*.annoy.idx")
        if not idx_paths:
            return
        idx_paths = np.array(idx_paths, dtype=object)
        scores = []
        for idx in idx_paths:
            score = 1.0
            score *= np.log(os.stat(idx).st_mtime)
            score *= cls.index_offset(idx) or 1.0
            scores.append(score)
        return idx_paths[np.argmin(scores)]

    def __init__(self, root: str, offset: int = None):
        root = os.path.abspath(root)
        dictpath = os.path.join(root, "dictionary.npz")
        dictfile = np.load(dictpath)
        V = dict(zip(dictfile["tags"], dictfile["vecs"]))
        A = np.load(os.path.join(root, "attenuator.npy"))
        recodict = np.load(os.path.join(root, "recognizer.npz"))
        recognizer_slots = "recognizer", "thumb_dim", "highfreq_factor"
        self.root = root
        if offset is None:
            index_name = self.search_index(root)
            offset = self.index_offset(index_name)
        self.offset = offset or self.index_offset(root) or 0
        self.attenuator = Attenuator(V, A)
        self.recognizer = Recognizer(
            **{k: v for k, v in recodict.items() if k in recognizer_slots}
        )
        self.index = annoy.AnnoyIndex(self.attenuator.n_dim, metric="angular")
        index_path = self.search_index(root)
        self.index.load(os.path.join(root, index_path or ""))

    def neighbors(self, query, n, *args, **kwargs):
        q = self.attenuator(query)
        return self.index.get_nns_by_vector(q, n, *args, **kwargs)

    def centroids(self, query_points, return_weights=True):
        neighborhood = np.array(sorted(set(query_points)))
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
