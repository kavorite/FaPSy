import dataclasses
from typing import Any, Callable, Iterable

import numpy as np


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
    hop_steps (int):
        Constant stride describing the length of each hop. Each node will
        receive hop_steps iterations away from it over the course of a tour, so
        make this small (e.g. =2 for bipartite graphs, as used in Pixie).
    """

    neighbors: Callable[[Node], float]
    user_pref: Callable[[Node], float]

    goal_hits: int = 16
    node_goal: int = 32
    max_steps: int = 256
    tour_hops: int = 8
    hop_steps: int = 1


def single_random_walk(
    origin: Node,
    traversal: Traversal,
    rng: np.random.Generator = np.random.default_rng(seed=0),
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
        neighbors = list(traversal.neighbors(node))
        scores = np.full((len(neighbors),), 1 / len(neighbors))
        scores += np.array(list(map(traversal.user_pref, neighbors)))
        scores -= scores.min()
        scores /= scores.max()
        return rng.choice(neighbors, p=scores)

    visits = dict()
    goals_hit = 0
    total_hit = 0
    while True:
        tour = int(rng.random() * traversal.tour_hops)
        tour = min(traversal.max_strides - total_hit, tour)
        for _hop in range(tour):
            dest = origin
            for _step in range(traversal.hop_steps):
                visits[dest] = visits.get(dest, 0) + 1
                dest = personalized_neighbor(dest)
                total_hit += 1
                if total_hit >= traversal.max_steps:
                    return visits
            if visits[dest] == traversal.node_goal:
                goals_hit += 1
            if goals_hit >= traversal.goal_hits:
                return visits


def random_walk(
    query_nodes: Iterable[Node],
    node_degrees: Iterable[float],
    traversal: Traversal,
    rng: np.random.Generator = np.random.default_rng(seed=0),
):
    max_degree = np.max(node_degrees)
    node_degrees = np.abs(node_degrees)
    node_prefs = node_degrees * (max_degree - np.log(node_degrees))
    max_steps = np.array([traversal.user_pref(q) for q in query_nodes])
    max_steps *= node_prefs
    max_steps *= traversal.max_steps
    max_steps /= node_prefs.sum()

    V = dict()
    for q, n in zip(query_nodes, max_steps):
        traversal = dataclasses.replace(traversal, max_steps=n, rng=rng)
        tour_hits = single_random_walk(q, traversal)
        for p, k in zip(query_nodes, tour_hits):
            if p not in V:
                V[p] = 0
            V[p] += np.sqrt(k)

    return {p: k ** 2 for p, k in V.items()}
