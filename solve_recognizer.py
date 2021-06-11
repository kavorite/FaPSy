import concurrent.futures as ft
import csv
import itertools as it
import queue
import threading

import annoy
import cv2
import numpy as np
import scipy.fftpack
import tqdm
from requests_futures.sessions import FuturesSession

from common import Embedder
from embed_tags import tag_hit_generator, tags_by_freq

csv.field_size_limit(1 << 20)


def img_embed(img_str, hash_size=8, highfreq_factor=4):
    dim = hash_size * highfreq_factor
    img = np.frombuffer(img_str, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (dim, dim))
    dct = scipy.fftpack.dct(scipy.fftpack.dct(img, axis=0), axis=1)
    lo = dct[:hash_size, :hash_size]
    return lo.flatten()


def phash(img_str, hash_size=8, highfreq_factor=4):
    x = img_embed(img_str, hash_size, highfreq_factor)
    mask = x > np.median(x)
    return mask


def post_content(posts, concurrency=32):
    posts_ready = threading.Semaphore()
    output = queue.Queue(maxsize=concurrency)
    client = FuturesSession(ft.ThreadPoolExecutor(concurrency))
    jobs = dict()

    def download_one(post):
        def on_complete(job):
            image_str = job.result().content
            post = jobs[job]
            posts_ready.release()
            output.put((post, image_str))

        job = client.get(post["link"], timeout=2)
        job.add_done_callback(on_complete)
        jobs[job] = post

    with client:
        for post in posts:
            download_one(post)
            if posts_ready.acquire(blocking=False):
                yield output.get()


def prep_posts(posts):
    for post in posts:
        md5 = post["md5"]
        ext = post["file_ext"] or "jpg"
        if post["is_deleted"] != "t":
            link = (
                r"https://static1.e621.net/data/preview/"
                f"{md5[0:2]}/{md5[2:4]}/{md5}.{ext}"
            )
            yield {"id": post["id"], "link": link, "tags": post["tag_string"].split()}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k-tags", type=int, default=2048)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--highfreq-factor", type=int, default=4)
    parser.add_argument("--dictionary", type=np.load, default="./index/dictionary.npz")
    args = parser.parse_args()
    dictfile = args.dictionary
    dictionary = dict(zip(dictfile["tags"], dictfile["vecs"]))
    all_tags = tags_by_freq(args.k_tags)
    tag_indx = {tag["name"]: i for i, tag in enumerate(all_tags)}

    def tag_embed(tags):
        tags = [t for t in tags if t in tag_indx]
        if not tags:
            return None
        idxs = [tag_indx[t] for t in tags]
        return dictionary[tags[np.argmin(idxs)]]

    image_features = []
    image_taggings = []

    print("read tags...")
    all_tags = [tag["name"] for tag in tags_by_freq(args.k_tags)]
    print("find post candidates...")
    posts = list(prep_posts(tag_hit_generator(all_tags)))
    hs = args.hash_size
    hf = args.highfreq_factor
    print("embed image previews; embed post tags...")
    for post, image_str in tqdm.tqdm(post_content(posts), total=len(posts)):
        y = tag_embed(post["tags"])
        if y is None:
            continue
        try:
            x = img_embed(image_str, hs, hf)
        except cv2.error:
            continue
        image_features.append(x)
        image_taggings.append(y)
    X = np.vstack(image_features)
    Y = np.vstack(image_taggings)
    print("solve for recognizer...")
    R, *_ = np.linalg.lstsq(X, Y, rcond=None)
    opath = "./index/recognizer.npy"
    np.save(opath, R)
    print(f"recognizer written to {opath}")


if __name__ == "__main__":
    main()
