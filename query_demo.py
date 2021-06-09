import glob
import pickle
import re
import sys
import webbrowser

import annoy
import numpy as np
from unidecode import unidecode

from common import Embedder, PostGraph


def strip_tag(t):
    t = unidecode(t)
    t = re.sub(r"[^a-zA-Z0-9_-]", "", t)
    return t


def main():
    graph = PostGraph.load_path("./")
    print(graph.offset)
    stripdict = {strip_tag(t): t for t in graph.embed.V}
    while True:
        sys.stderr.write("tags> ")
        query = (q if q not in stripdict else stripdict[q] for q in input().split())
        top_k = 8
        post_ids = graph.tag_neighbors(query, top_k)
        if len(post_ids) == 0:
            sys.stderr.write(
                "looks like we don't have any neighbors for that query 😰\n"
            )
            continue
        hits = [f"https://e621.net/posts/{post_id}" for post_id in post_ids[:top_k]]
        sys.stderr.write(f"open {len(post_ids)} closest:\n")
        sys.stderr.write("\n".join(hits))
        for endpoint in hits:
            webbrowser.open_new_tab(endpoint)


if __name__ == "__main__":
    main()
