## e6graph

This repository is a suite of tools for generating a graph from E621 metadata.
If you run each script in the top-level directory, you will be capable of
deploying an implementation of a graph-based, online recommender system
inspired by [`pinterest/pixie`][pixie]. To efficiently sample candidates from
the local "neighborhood" of a query node for further filtering, we need to
encode the topology of the network for programs to read very quickly, for which
I chose to use [`spotify/annoy`][annoy], and also employed techniques from
[Krawetz][phash], and [Arora, et al.][alacarte]. I'll keep inlining references
throughout this document at my discretion. You can find a comprehensive list at
the end in the [markdown source][this].

### The approach

#### (in excruciating detail)

1. run `hydrate_data.py`. This downloads necessary archives from
   [a database dump][db_export].
2. run `embed_tags.py`. This will give you dense vectors for tags using
   factorization of a pointwise mutual information matrix derived from their
   cooccurrences and dump embeddings á la [LSI][stop_using_word2vec] to
   `./index/dictionary.npz`. These are a bit of an idiosyncrasy, because in
   my case, each tag cooccurs with other tags to form the distributional
   representation of each term as a "document," but particularly for this
   application, I can't think of any reason this isn't an appropriate
   formulation.
3. run `solve_attenuator.py`. This solves a least-squares optimization to
   induce a linear operator which is good for predicting rare tags from blobs
   of common ones. This is useful both for constructing and querying the graph:
   normally when embeddings are produced by averaging tag vectors, signal is
   lost because dissonant noise cancels out most of the semantic information.
   The attenuator is computed and used during one-shot embedding of tagsets to
   minimize this effect. Thanks to [Arora, et al.][alacarte] for the tech.
4. run `construct_index.py`. This will build a [`spotify/annoy`][annoy] index
   file that can be queried for the **attenuated mean** of the tag-vectors for
   any given post using `annoy`, yielding the post's top `k` approximate nearest
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
   documentation, and cost more compute. `annoy` is the best, at least without
   domain-specific optimization for our use-case or use of extremely powerful
   hardware.
5. run `solve_recognizer.py`. This is technically not essential, but the bot
   demo won't work without it: it's what allows arbitrary images to be embedded
   in the same feature space as their tags without use of an image-query
   database. It learns a _separate_ new linear operator which maps the discrete
   cosine transforms of horizontal and vertical image scan-line signals to their
   attenuated tag embeddings, allowing any image vector obtained in the same way
   to be used in a content-based nearest-neighbors query. Thanks to
   [Dr. Krawetz][phash] for this tech.

[alacarte]: http://www.offconvex.org/2018/09/18/alacarte/
[annoy]: https://github.com/spotify/annoy
[db_export]: https://e621.net/db_export/
[pixie]: https://medium.com/pinterest-engineering/introducing-pixie-an-advanced-graph-based-recommendation-system-e7b4229b664b
[phash]: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
[stop_using_word2vec]: https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/
[this]: /kavorite/e6graph/blob/main/README.md
