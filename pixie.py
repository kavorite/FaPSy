import dataclasses
from collections import Counter
from typing import Callable, Iterable, Optional

import numpy as np
from expiringdict import ExpiringDict


class Node:
    pass


@dataclasses.dataclass
class Traversal:
    """Early-stopping parameters for a Pixie random walk.


    goal_hits (int): desired number of visits on each final candidate
    node_goal (int): number of candidates to saturate with goal_hits
    max_steps (int):
        upper bound on the iterations of the algorithm. This should probably
        be set as an exponent of some linear combination of node_goal and
        goal_hits.
    tour_hops (int):
        upper bound on the number of steps in a single random walk.
    """

    neighbors: Callable[[Node], float]
    user_pref: Callable[[Node], float]
    rng: Optional[np.random.Generator] = np.random.default_rng(seed=0)

    goal_hits: int = 16
    node_goal: int = 32
    max_steps: int = 256
    tour_hops: int = 8


def single_random_walk(
    origin: Node,
    traversal: Traversal,
):
    """
    Use a biased random walk to sample a new post to be the successive steps in
    a Markov chain proceeding from the given query nodes.

    Parameters:
        origin: The starting query node.
        neighbors (node -> list): Network function
            Function that returns a list of neighboring nodes given a node.
        user_pref (node -> number): Personalization function
            Callback for computing the user's preference for a given node.
            Pinterest's implementation of this function is parametrized over
            the end-user's past interactions with the network; individual nodes
            are weighted both by the age and sentiment valence of these
            interactions.
        traversal (PixieTraversal): Traversal parameters
            These are good for early stopping, describing how the graph is
            constructed, and how many nodes should be considered in order for
            the output to be useful.
        rng (np.random.Generator): Random number generator
    Returns:
        visits (dict):
            A dictionary mapping the visited nodes to the number of
            visits they received.
    """

    def personalized_neighbor(node):
        nodes = np.array(list(traversal.neighbors(node)), dtype=object)
        prefs = np.ones(len(nodes))
        prefs *= np.array(list(map(traversal.user_pref, nodes)))
        picks = np.where(prefs >= np.median(prefs))
        nodes = nodes[picks]
        prefs = prefs[picks]
        prefs /= prefs.sum()
        return traversal.rng.choice(nodes, p=prefs)

    visits = dict()
    goals_hit = 0
    total_hit = 0
    while True:
        tour = traversal.rng.integers(1, traversal.tour_hops + 1)
        dest = origin
        for _hop in range(tour):
            if dest != origin:
                visits[dest] = visits.get(dest, 0) + 1
            dest = personalized_neighbor(dest)
            total_hit += 1
            if total_hit >= traversal.max_steps:
                return visits
            if visits.get(dest, 0) == traversal.node_goal:
                goals_hit += 1
            if goals_hit >= traversal.goal_hits:
                return visits


def random_walk(
    query_nodes: Iterable[Node],
    traversal: Traversal,
):
    V = Counter()
    query_nodes = np.array(list(query_nodes), dtype=object)
    query_prefs = np.array([traversal.user_pref(q) for q in query_nodes])
    selection = np.where(query_prefs >= np.median(query_prefs))
    query_prefs = query_prefs[selection]
    query_nodes = query_nodes[selection]
    max_steps = np.ones(len(query_nodes)) * traversal.max_steps // len(query_nodes)
    for q, n in zip(query_nodes, max_steps):
        traversal = dataclasses.replace(traversal, max_steps=n)
        tour_hits = single_random_walk(q, traversal)
        for p, k in tour_hits.items():
            if p != q and k != 0:
                V[p] += np.sqrt(float(k))

    return {p: k ** 2 for p, k in V.items()}
