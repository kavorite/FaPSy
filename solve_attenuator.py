import csv
import gzip
import io
import pickle
import warnings

import numpy as np

from embed_tags import tag_hit_generator

csv.field_size_limit(1 << 20)
dictfile = np.load("./index/dictionary.npz")
all_tags, embeddings = dictfile["tags"], dictfile["vecs"]
embeddings = np.vstack(embeddings)
tag_idx = {t: i for i, t in enumerate(all_tags)}

print("generate contexts around rare tags...")
contexts = []
centroids = []
for post in tag_hit_generator(tag_idx):
    tags = post["tag_string"].split()
    idxs = np.array([tag_idx[t] for t in tags if t in tag_idx])
    if len(idxs) == 0:
        continue
    centroid = idxs.min()
    context = idxs[idxs != centroid]
    centroids.append(centroid)
    contexts.append(context)

# create naive vectors for each context
X = []
safe = []
with warnings.catch_warnings():
    warnings.simplefilter("error")
    for i, context in enumerate(contexts):
        try:
            X.append(embeddings[context].mean(axis=0))
            safe.append(True)
        except:
            safe.append(False)

X = np.vstack(X)
# find the rarest vector for each context
Y = embeddings[centroids][safe]

print("compute the attenuation matrix...")
A, *_ = np.linalg.lstsq(X, Y, rcond=None)
np.save("./index/attenuator.npy", A, allow_pickle=False)
print("attenuation matrix written to ./index/attenuator.npy")
