# (c) Copyright Riverlane 2020-2025.
"""
Sub-package for defining decoding graphs and related data types; these structures are
used by many decoders to define their understanding of the code and noise model being
decoded.
"""

from deltakit_core.decoding_graphs._data_qubits import (
    DecodingEdge,
    DecodingHyperEdge,
    EdgeRecord,
    EdgeT,
    OrderedDecodingEdges,
    errors_to_syndrome,
)

from deltakit_core.decoding_graphs._decoding_graph import (
    AnyEdgeT,
    DecodingCode,
    DecodingHyperGraph,
    DecodingHyperMultiGraph,
    HyperLogicals,
    HyperMultiGraph,
    NXCode,
    NXDecodingGraph,
    NXDecodingMultiGraph,
    NXLogicals,
)

from deltakit_core.decoding_graphs._dem_parsing import (
    DemParser,
    DetectorCounter,
    DetectorRecorder,
    LogicalsInEdges,
    dem_to_decoding_graph_and_logicals,
    dem_to_hypergraph_and_logicals,
    observable_warning,
)

from deltakit_core.decoding_graphs._syndromes import (
    Bitstring,
    DetectorRecord,
    FixedWidthBitstring,
    OrderedSyndrome,
    get_round_words,
)

from deltakit_core.decoding_graphs._weighted_graphs import (
    change_graph_error_probabilities,
    vector_weights,
)

from deltakit_core.decoding_graphs._decoding_graph_tools import (
    compute_graph_distance,
    compute_graph_distance_for_logical,
    filter_to_data_edges,
    filter_to_measure_edges,
    graph_to_json,
    has_contiguous_nodes,
    hypergraph_to_weighted_edge_list,
    inverse_logical_at_boundary,
    is_single_connected_component,
    single_boundary_is_last_node,
    unweight_graph,
    worst_case_num_detectors,
)

from deltakit_core.decoding_graphs._explained_dem_parsing import (
    extract_logicals,
    parse_explained_dem,
)

from deltakit_core.decoding_graphs._hypergraph_decomposition import decompositions

# List only public members in `__all__`.
__all__ = [
    "DecodingEdge",
    "DecodingHyperEdge",
    "EdgeRecord",
    "EdgeT",
    "OrderedDecodingEdges",
    "errors_to_syndrome",
    "AnyEdgeT",
    "DecodingCode",
    "DecodingHyperGraph",
    "DecodingHyperMultiGraph",
    "HyperLogicals",
    "HyperMultiGraph",
    "NXCode",
    "NXDecodingGraph",
    "NXDecodingMultiGraph",
    "NXLogicals",
    "DemParser",
    "DetectorCounter",
    "DetectorRecorder",
    "LogicalsInEdges",
    "dem_to_decoding_graph_and_logicals",
    "dem_to_hypergraph_and_logicals",
    "observable_warning",
    "Bitstring",
    "DetectorRecord",
    "FixedWidthBitstring",
    "OrderedSyndrome",
    "get_round_words",
    "change_graph_error_probabilities",
    "vector_weights",
    "compute_graph_distance",
    "compute_graph_distance_for_logical",
    "filter_to_data_edges",
    "filter_to_measure_edges",
    "graph_to_json",
    "has_contiguous_nodes",
    "hypergraph_to_weighted_edge_list",
    "inverse_logical_at_boundary",
    "is_single_connected_component",
    "single_boundary_is_last_node",
    "unweight_graph",
    "worst_case_num_detectors",
    "extract_logicals",
    "parse_explained_dem",
    "decompositions",
]
