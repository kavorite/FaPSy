## e6graph

This repository is a suite of tools for generating a graph from E621 metadata
for sublinear querying of a post's neighbors by ID, as part of [`e6px`][e6px],
an implementation of a graph-based, online recommender system inspired by
[`pinterest/pixie`][pixie]. To efficiently sample candidates from the local
"neighborhood" of a query node for further filtering, we need to encode the
topology of the network for programs to read very quickly. So here goes.

Basically the way you use all this stuff is:

1. run `hydrate_data.py`. This downloads necessary archives from
   [a database dump][db_export].
2. run `embed_tags.py`. This will give you dense vectors for tags using
   factorization of a pointwise mutual information matrix derived from their
   cooccurrences and `pickle.dump` them as a `tuple` in `./dictionary.pkl`.
3. run `solve_attenuator.py`. This solves a least-squares regression problem to
   produce a linear operator which is good for predicting rare tags from blobs
   of common ones. This is useful both for constructing and querying the graph:
   normally when embeddings are produced by averaging, signal is lost because
   dissonant noise cancels out most of the semantic information. The attenuator
   is computed to minimize this effect using least-squares optimization.
   Thanks to [Arora, et al.][alacarte] for the tech.
4. run `construct_index.py`. This will build an index file that can be queried
   for the **attenuated mean** of the tag-vectors for any given post using
   [`spotify/annoy`][annoy], yielding the post's top `k` approximate nearest
   neighbors in sublinear time. Its output IDs will need to be summed with the
   offset to actually resolve to E621 posts because `annoy` allocates space for
   all IDs in the keyspace up to and including the maximum, even if you don't
   actually backfill all of that space. This is probably something to do with
   the fact that it just memory-maps files to save space in memory, which is
   also what makes it tractable to build an index this large without having
   access to ridiculous amounts of RAM. In this case, it's a liability, but one
   that we can easily overlook, and well worth the cost. I've looked into
   alternatives. Some are faster, some are more accurate, some even save more
   space. But all of them are also more difficult to use, feature worse
   documentation, and cost more compute. `annoy` is the best, at least
   without domain-specific optimization for our use-case or use of extremely
   powerful hardware. Trust me on this.

[alacarte]: http://www.offconvex.org/2018/09/18/alacarte/
[annoy]: https://github.com/spotify/annoy
[db_export]: https://e621.net/db_export/
[pixie]: https://medium.com/pinterest-engineering/introducing-pixie-an-advanced-graph-based-recommendation-system-e7b4229b664b
[e6px]: https://github.com/kavorite/e6px
