import argparse
import csv
import gzip
import io
import pickle

import scipy.sparse
import scipy.sparse.linalg
from tqdm import tqdm


def tag_hit_generator(tagset, saturation_hits=8, goal_saturation=0.9):
    tagset = set(tagset)
    with gzip.open("./posts.csv.gz") as istrm:
        posts = csv.DictReader(io.TextIOWrapper(istrm, encoding="utf8"))
        goal = {t: saturation_hits for t in tagset}
        with tqdm(total=saturation_hits * len(tagset)) as progress:
            for post in posts:
                tags = post["tag_string"].split()
                yield tags
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-top", type=int, default=8192)
    parser.add_argument("--k-dim", type=int, default=32)
    args = parser.parse_args()
    csv.field_size_limit(1 << 20)
    with gzip.open("./tags.csv.gz") as istrm:
        print("read tags...")
        istrm = io.TextIOWrapper(istrm, encoding="utf8")
        all_tags = list(csv.DictReader(istrm))

    all_tags.sort(key=lambda tag: int(tag["post_count"]))
    all_tags.reverse()

    tag_idx = dict()
    # tag_idf = np.zeros(args.k_top)
    for tag in all_tags:
        if tag["category"] not in "0":
            continue
        t = tag["name"]
        if not t.isprintable():
            continue
        i = len(tag_idx)
        # idf = np.log(2.8e6 / float(tag["post_count"]))
        # tag_idf[i] = idf
        tag_idx[t] = i
        if len(tag_idx) == args.k_top:
            break

    print("generate cooccurrence matrix...")
    coo_idx = []
    coo_val = []
    for tags in tag_hit_generator(tag_idx):
        for t in tags:
            if t not in tag_idx:
                continue
            for w in tags:
                if w not in tag_idx:
                    continue
                i, j = tag_idx[t], tag_idx[w]
                coo_idx.append((i, j))
                coo_val.append(1.0)

    print("factorize...")
    C = scipy.sparse.coo_matrix((coo_val, zip(*coo_idx))).tocsr()
    row_norms = C.max(axis=-1).A.flatten()
    N = scipy.sparse.spdiags(1.0 / row_norms, 0, *C.shape)
    C = C * N
    U, _, _ = scipy.sparse.linalg.svds(C, args.k_dim, return_singular_vectors="u")
    # W = tag_idf[:, None]
    # NOTE: downweighting by idf is no longer necessary with linear attenuation
    # U *= W
    dictionary = tuple((t, U[i]) for t, i in tag_idx.items())
    with open("./dictionary.pkl", "wb+") as ostrm:
        pickle.dump(dictionary, ostrm)
    print("embeddings written to dictionary.pkl")


if __name__ == "__main__":
    main()
