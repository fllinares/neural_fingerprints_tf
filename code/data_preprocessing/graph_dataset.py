import operator, copy
import numpy as np
import tensorflow as tf


def make_batches(graphs, batch_size, random_state=0):
    # Number of graphs
    n_graphs = len(graphs)
    # Number of batches
    n_batches = n_graphs/batch_size + ((n_graphs % batch_size) != 0)

    # Number of node features, edge features and targets that the graphs have, respectively
    num_node_features, num_edge_features, n_targets = graphs[0]['shape'][2:5]

    # Set random seed to shuffle examples prior to forming batches
    np.random.seed(random_state)
    perm = np.random.permutation(n_graphs)

    # Prepare batches
    batches = []
    for i in xrange(n_batches):
        # Create dictionary representing the batch of graphs
        batch = {}

        # Retrieve the graphs (randomly) assigned to the i-th batch
        batch_idx = perm[(i*batch_size):min((i+1)*batch_size, n_graphs)]
        batch_graphs = operator.itemgetter(*batch_idx)(graphs)
        n_graphs_batch = len(batch_graphs)

        # Retrieve the number of nodes and edges in each graph of the batch
        n_nodes = np.array(map(lambda g: g['shape'][0], batch_graphs))
        n_edges = np.array(map(lambda g: g['shape'][1], batch_graphs))
        # Compute cumulative number of nodes and edges, i.e. total number of nodes and edges in all previous
        # graphs in the batch
        cum_n_nodes = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))
        cum_n_edges = np.concatenate(([0], np.cumsum(n_edges)[:-1]))

        # Concatenate NumPy arrays
        batch['node_features'] = np.concatenate(map(lambda g: g['node_features'], batch_graphs), axis=0)
        batch['edge_features'] = np.concatenate(map(lambda g: g['edge_features'], batch_graphs), axis=0)
        batch['adj_mat'] = np.concatenate(map(lambda (g, offset): g['adj_mat'] + offset,
                                              zip(batch_graphs, cum_n_nodes)), axis=0)
        batch['inc_mat'] = np.concatenate(map(lambda (g, node_offset, edge_offset): g['inc_mat'] + [node_offset, edge_offset],
                                              zip(batch_graphs, cum_n_nodes, cum_n_edges)), axis=0)
        batch['target'] = np.concatenate(map(lambda g: g['target'], batch_graphs), axis=0)
        batch['id'] = np.array(map(lambda g: g['id'], batch_graphs)) # IDs of graphs in the batch

        # 1D numpy arrays indicating the index of the graph (within the batch) to which each node and belong to
        batch['node_graph_map'] = np.repeat(np.arange(n_graphs_batch, dtype=np.int64), n_nodes)
        batch['edge_graph_map'] = np.repeat(np.arange(n_graphs_batch, dtype=np.int64), n_edges)

        # Add shape information
        batch['shape'] = np.array((np.sum(n_nodes), np.sum(n_edges), num_node_features, num_edge_features,
                                   n_targets, n_graphs_batch), np.int64)

        # Add batch to list of batches
        batches.append(batch)

    return batches

# -------------------------------------- I/O FUNCTIONS -------------------------------------------------------------


# Write all graphs to a TFRecord file
def write_graphs_to_tfrecord(graphs, filename):
    # Create writer
    writer = tf.python_io.TFRecordWriter(filename)

    # Iterate across all graphs in the dataset
    for g in graphs:
        # Convert graph to TensorFlow Example
        example = graph_to_example(g)
        # Write serialised example to file
        writer.write(example.SerializeToString())

    # Closer writer
    writer.close()


# Read graphs from TFRecord file
def read_graphs_from_tfrecord(filename):
    # Create record iterator
    record_iterator = tf.python_io.tf_record_iterator(path=filename)

    # Initialise list of read graphs
    graphs = []
    # For each record in the file, parse the corresponding molecular graph (represented as a Python dictionary)
    for string_record in record_iterator:
        g = string_record_to_graph(string_record)
        graphs.append(g)

    return graphs


# Encode a graph as a TensorFlow Example object
def graph_to_example(g):
    feature_dict = {k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[g[k].tobytes()])) for k in g.keys()}

    # Create TensorFlow Example object
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example


# Decode a graph from a TensorFlow record
def string_record_to_graph(string_record):
    # Create a TensorFlow example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Represent molecular graph as a dictionary
    g = {}

    # Retrieve shape information
    shape_bytes = example.features.feature['shape'].bytes_list.value[0]
    g['shape'] = np.fromstring(shape_bytes, dtype=np.int64)
    n_nodes, n_edges, num_node_features, num_edge_features, n_targets, n_graphs = g['shape']

    # Retrieve NumPy arrays representing the graph
    if 'node_features' in example.features.feature:
        node_features_string = example.features.feature['node_features'].bytes_list.value[0]
        g['node_features'] = np.reshape(np.fromstring(node_features_string, dtype=np.float32),
                                        (n_nodes, num_node_features))
    if 'edge_features' in example.features.feature:
        edge_features_string = example.features.feature['edge_features'].bytes_list.value[0]
        g['edge_features'] = np.reshape(np.fromstring(edge_features_string, dtype=np.float32),
                                        (n_edges, num_edge_features))
    if 'adj_mat' in example.features.feature:
        adj_mat_string = example.features.feature['adj_mat'].bytes_list.value[0]
        g['adj_mat'] = np.reshape(np.fromstring(adj_mat_string, dtype=np.int64), (-1, 2))
    if 'inc_mat' in example.features.feature:
        inc_mat_string = example.features.feature['inc_mat'].bytes_list.value[0]
        g['inc_mat'] = np.reshape(np.fromstring(inc_mat_string, dtype=np.int64), (-1, 2))
    if 'target' in example.features.feature:
        target_string = example.features.feature['target'].bytes_list.value[0]
        g['target'] = np.reshape(np.fromstring(target_string, dtype=np.float32), (n_graphs, n_targets))
    if 'node_graph_map' in example.features.feature:
        node_graph_map_string = example.features.feature['node_graph_map'].bytes_list.value[0]
        g['node_graph_map'] = np.fromstring(node_graph_map_string, dtype=np.int64)
    if 'edge_graph_map' in example.features.feature:
        edge_graph_map_string = example.features.feature['edge_graph_map'].bytes_list.value[0]
        g['edge_graph_map'] = np.fromstring(edge_graph_map_string, dtype=np.int64)
    if 'id' in example.features.feature:
        id_string = example.features.feature['id'].bytes_list.value[0]
        g['id'] = np.fromstring(id_string, dtype=np.int64)

    return g
