# (c) Copyright Riverlane 2020-2025.
from typing import FrozenSet, Iterator, List, Sequence

from deltakit_core.decoding_graphs import DecodingHyperEdge


def decompositions(
    target_edge: DecodingHyperEdge, edges: Sequence[DecodingHyperEdge]
) -> Iterator[FrozenSet[DecodingHyperEdge]]:
    """Generate decompositions of the given edge into the parts from edges.
    The union of all vertices in the parts must equal the vertices in the
    original edges for it to be considered a valid decomposition.

    Parameters
    ----------
    target_edge : DecodingHyperEdge
        The hyperedge to decompose into parts.
    edges : Sequence[DecodingHyperEdge]
        The edges in which to search for a decomposition.

    Yields
    ------
    Iterator[FrozenSet[DecodingHyperEdge]]
        A generator of frozensets with each item a different decomposition
        of the input edge.
    """

    # explore tree of possible decompositions

    def _decompositions(
        vertices: FrozenSet[int],
        legal_edges: List[DecodingHyperEdge],
        chosen_edges: FrozenSet[DecodingHyperEdge],
    ) -> Iterator[FrozenSet[DecodingHyperEdge]]:
        if len(vertices) == 0:  # done on this branch
            yield chosen_edges
            return
        if len(legal_edges) == 0:  # dead branch
            return
        first, *others = legal_edges
        if first.vertices.issubset(vertices):
            # use it
            yield from _decompositions(
                vertices - first.vertices, others, chosen_edges | {first}
            )
        # don't use it (even if you could)
        yield from _decompositions(vertices, others, chosen_edges)

    yield from _decompositions(
        target_edge.vertices,
        [edge for edge in edges if edge.vertices.issubset(target_edge)],
        frozenset(),
    )
