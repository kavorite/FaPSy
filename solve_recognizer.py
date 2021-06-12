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

from common import EmbeddingObjective, Recognizer
from embed_tags import tag_hit_generator, tags_by_freq

csv.field_size_limit(1 << 20)


def preview_content(posts, concurrency=32):
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


def ensure_preview(posts):
    for post in posts:
        md5 = post["md5"]
        ext = post["file_ext"] or "jpg"
        if post["is_deleted"] != "t":
            link = (
                r"https://static1.e621.net/data/preview/"
                f"{md5[0:2]}/{md5[2:4]}/{md5}.{ext}"
            )
            yield {"link": link, **post}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k-tags", type=int, default=2048)
    parser.add_argument("--thumb-dim", type=int, default=8)
    parser.add_argument("--highfreq-factor", type=int, default=4)
    parser.add_argument(
        "--dictionary", type=EmbeddingObjective, default="./index/dictionary.npz"
    )
    args = parser.parse_args()
    naive_recognizer = Recognizer(args.thumb_dim, args.highfreq_factor)
    objective = args.dictionary
    all_tags = tags_by_freq(args.k_tags)

    image_features = []
    image_taggings = []

    print("read tags...")
    all_tags = [tag["name"] for tag in tags_by_freq(args.k_tags)]
    print("find post candidates...")
    posts = list(ensure_preview(tag_hit_generator(all_tags)))
    print("embed image previews; embed post tags...")
    for post, image_str in tqdm.tqdm(preview_content(posts), total=len(posts)):
        y = objective(post["tag_string"].split())
        if y is None:
            continue
        try:
            x = naive_recognizer(image_str, args.thumb_dim, args.highfreq_factor)
        except cv2.error:
            continue
        image_features.append(x)
        image_taggings.append(y)
    X = np.vstack(image_features)
    Y = np.vstack(image_taggings)
    print("solve for recognizer...")
    R, *_ = np.linalg.lstsq(X, Y, rcond=None)
    opath = "./index/recognizer.npz"
    np.savez(
        opath,
        recognizer=R,
        thumb_dim=args.thumb_dim,
        highfreq_factor=args.highfreq_factor,
    )
    print(f"recognizer written to {opath}")


if __name__ == "__main__":
    main()
