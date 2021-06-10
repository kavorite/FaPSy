import collections
import concurrent.futures as ft
import csv
import gzip
import io
import itertools as it
import os
import urllib.error
import urllib.request
from socket import timeout
from threading import BoundedSemaphore

import annoy
import numpy as np
import tensorflow as tf
import tqdm


def load_posts():
    with gzip.open("./db_export/posts.csv.gz") as istrm:
        istrm = io.TextIOWrapper(istrm, encoding="utf8")
        posts = csv.DictReader(istrm)
        for post in posts:
            md5 = post["md5"]
            ext = post["file_ext"] or "jpg"
            if post["is_deleted"] != "t":
                link = (
                    r"https://static1.e621.net/data/preview/"
                    f"{md5[0:2]}/{md5[2:4]}/{md5}.{ext}"
                )
                post["link"] = link
            yield post


def download_posts(posts, concurrency=os.cpu_count()):
    concurrency = BoundedSemaphore(concurrency)

    def download_one(link):
        with urllib.request.urlopen(link) as rsp:
            return rsp.read()

    jobs = collections.deque()
    with ft.ThreadPoolExecutor() as pool:
        for post in posts:
            if "link" not in post:
                continue
            if concurrency.acquire(blocking=False):
                jobs.append(
                    pool.submit(
                        download_one,
                    )
                )
            else:
                image_str = jobs.popleft().result()
                yield image_str


@tf.function
def phash(img_str, hash_size=8, highfreq_factor=4):
    img = tf.io.decode_image(img_str, expand_animations=False)
    img = img[..., :3]
    if tf.shape(img)[-1] > 1:
        img = tf.image.rgb_to_grayscale(img)
    n = hash_size * highfreq_factor
    img = tf.image.resize(img, (n, n))
    img = tf.squeeze(img, axis=-1)
    dct = tf.signal.dct(img)
    lo = dct[:hash_size, :hash_size]
    lo = tf.sort(tf.reshape(lo, [tf.reduce_prod(tf.shape(lo))]))
    k = hash_size // 2
    if hash_size % 2 == 0:
        mu = 0.5 * lo[k + 1] + 0.5 * lo[k + 0]
    else:
        mu = lo[k]

    return lo > mu


@tf.function
def pack(hash_mask):
    packed = 0
    for i, b in enumerate(hash_mask.numpy()):
        hashes[i] |= b & (1 << i)
    return packed


hash_size = 8
posts, posts2 = it.tee(load_posts(), 2)
post_ids = (int(post["id"]) for post in posts)
post_ids = tf.data.Dataset.from_generator(
    lambda: post_ids, output_signature=tf.TensorSpec(shape=[], dtype=tf.int32)
)

hashes = (
    tf.data.Dataset.from_generator(
        lambda: download_posts(posts2),
        output_signature=tf.TensorSpec(shape=[], dtype=tf.string),
    )
    .map(
        phash,
        num_parallel_calls=os.cpu_count() // 2,
    )
    .prefetch(tf.data.AUTOTUNE)
    .zip(post_ids)
    # .map(pack, num_parallel_calls=os.cpu_count() // 2)
    # .apply(tf.data.experimental.ignore_errors())
)

index = annoy.AnnoyIndex(f=hash_size, metric="hamming")
index.on_disk_build("./index/iqdb.annoy.idx")
for post_id, x in hashes:
    tf.print(post_id)
    index.add_item(post_id, x)
