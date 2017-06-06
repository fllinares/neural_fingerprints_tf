import numpy as np
import tensorflow as tf
from basic_model import BasicModel


class NeuralFingerprints(BasicModel):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ INITIALIZATION FUNCTIONS ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, config, input_batches_file, is_training):
        BasicModel.__init__(self, config, input_batches_file, is_training)

    def parse_config_train(self):
        # Path to training-related hyperparameters and configuration settings
        prefix = 'train'

        # Set the random seed for (partial) reproducibility
        np.random.seed(self.getitem('config', 'seed', prefix))
        tf.set_random_seed(self.getitem('config', 'seed', prefix))

    def parse_config_graph_convolutional_layers(self):
        # Path to hyperparameters and configuration settings for the graph convolutional layers
        prefix = 'model/graph_conv_layers'

        # Parse options regarded the use of Batch Normalization in the graph convolutional layers
        self.normalizer_fn_graph_conv = None
        self.normalizer_params_graph_conv = None
        if self.getitem('config', 'use_batch_norm', prefix):
            self.normalizer_fn_graph_conv = tf.contrib.layers.batch_norm
            self.normalizer_params_graph_conv = {'decay': self.getitem('config', 'batch_norm_decay', prefix),
                                                 'center': True,
                                                 'scale': True,
                                                 'is_training': self.is_training,
                                                 'updates_collections': None}

        # Parse options regarding the use of L1/L2 regularization of the weights in the graph convolutional layers
        regularizer_list = []
        l1_reg_val = self.getitem('config', 'l1_reg', prefix)
        if l1_reg_val > 0.0:
            regularizer_list.append(tf.contrib.layers.l1_regularizer(l1_reg_val))
        l2_reg_val = self.getitem('config', 'l2_reg', prefix)
        if l2_reg_val > 0.0:
            regularizer_list.append(tf.contrib.layers.l2_regularizer(l2_reg_val))
        self.weights_regularizer_graph_conv = tf.contrib.layers.sum_regularizer(regularizer_list)

        # Parse options regarding the initialization of the weights in the graph convolutional layers
        if self.getitem('config', 'weights_initializer', prefix) == 'xavier':
            self.weights_initializer_graph_conv = tf.contrib.layers.xavier_initializer()
        else:
            range_val = np.exp(float(self.getitem('config', 'weights_initializer', prefix)))
            self.weights_initializer_graph_conv = tf.random_uniform_initializer(minval=-range_val, maxval=range_val)

    def parse_config_fingerprint_output_layers(self):
        # Path to hyperparameters and configuration settings for the fingerprint output layers
        prefix = 'model/fingerprint_output_layers'

        # Parse options regarding the use of L1/L2 regularization of the weights in the fingerprint output layers
        regularizer_list = []
        l1_reg_val = self.getitem('config', 'l1_reg', prefix)
        if l1_reg_val > 0.0:
            regularizer_list.append(tf.contrib.layers.l1_regularizer(l1_reg_val))
        l2_reg_val = self.getitem('config', 'l2_reg', prefix)
        if l2_reg_val > 0.0:
            regularizer_list.append(tf.contrib.layers.l2_regularizer(l2_reg_val))
        self.weights_regularizer_fp_out = tf.contrib.layers.sum_regularizer(regularizer_list)

        # Parse options regarding the initialization of the weights in the fingerprint output layers
        if self.getitem('config', 'weights_initializer', prefix) == 'xavier':
            self.weights_initializer_fp_out = tf.contrib.layers.xavier_initializer()
        else:
            range_val = np.exp(float(self.getitem('config', 'weights_initializer', prefix)))
            self.weights_initializer_fp_out = tf.random_uniform_initializer(minval=-range_val, maxval=range_val)

    def parse_config_mlp(self):
        # Path to hyperparameters and configuration settings for the output MLP
        prefix = 'model/mlp'

        # Parse options regarded the use of Batch Normalization in the output MLP
        self.normalizer_fn_mlp = None
        self.normalizer_params_mlp = None
        if self.getitem('config', 'use_batch_norm', prefix):
            self.normalizer_fn_mlp = tf.contrib.layers.batch_norm
            self.normalizer_params_mlp = {'decay': self.getitem('config', 'batch_norm_decay', prefix),
                                          'center': True,
                                          'scale': True,
                                          'is_training': self.is_training,
                                          'updates_collections': None}

        # Parse options regarding the use of L1/L2 regularization of the weights in the output MLP
        regularizer_list = []
        l1_reg_val = self.getitem('config', 'l1_reg', prefix)
        if l1_reg_val > 0.0:
            regularizer_list.append(tf.contrib.layers.l1_regularizer(l1_reg_val))
        l2_reg_val = self.getitem('config', 'l2_reg', prefix)
        if l2_reg_val > 0.0:
            regularizer_list.append(tf.contrib.layers.l2_regularizer(l2_reg_val))
        self.weights_regularizer_mlp = tf.contrib.layers.sum_regularizer(regularizer_list)

        # Parse options regarding the initialization of the weights in the output MLP
        if self.getitem('config', 'weights_initializer', prefix) == 'xavier':
            self.weights_initializer_mlp = tf.contrib.layers.xavier_initializer()
        else:
            range_val = np.exp(float(self.getitem('config', 'weights_initializer', prefix)))
            self.weights_initializer_mlp = tf.random_uniform_initializer(minval=-range_val, maxval=range_val)

    def parse_config_loss(self):
        self.loss_fn = self.string_to_tf_loss(self.getitem('config', 'model/loss_fn'))

    def parse_config(self):
        self.parse_config_train()
        self.parse_config_graph_convolutional_layers()
        self.parse_config_fingerprint_output_layers()
        self.parse_config_mlp()
        self.parse_config_loss()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------ METHODS TO BUILD THE COMPUTATIONAL GRAPH ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def build_inputs(self):
        # Path to hyperparameters and configuration settings for the input
        prefix_node = 'model/input/node_features'
        prefix_edge = 'model/input/edge_features'

        # In the current implementation, all input data is contained in a single file: self.input_batches_file
        # This file will contain one TFRecord per input batch
        # TODO: Change reading system to allow shuffling of batches across epochs
        self.filename_queue = tf.train.string_input_producer([self.input_batches_file], num_epochs=None)
        self.reader = tf.TFRecordReader()
        _, serialized_example = self.reader.read(self.filename_queue)

        # Load TFRecords encoding batches as generated by class GraphDataset. Some fields in the TFRecord are
        # unnecessary and, therefore, are not decoded.
        feature_dict = {'node_features': tf.FixedLenFeature([], tf.string),
                        'adj_mat': tf.FixedLenFeature([], tf.string),
                        'node_graph_map': tf.FixedLenFeature([], tf.string),
                        'target': tf.FixedLenFeature([], tf.string)}
        # The edge features and incidence matrix are only useful if edge labels are to be taken into consideration
        if self.getitem('config', 'use', prefix_edge) and self.num_edge_features > 0:
            feature_dict['edge_features'] = tf.FixedLenFeature([], tf.string)
            feature_dict['inc_mat'] = tf.FixedLenFeature([], tf.string)

        # Parse example
        features = tf.parse_single_example(serialized_example, features=feature_dict)

        # Return input batch as a dictionary
        self.input['node_features'] = tf.reshape(tf.decode_raw(features['node_features'], tf.float32),
                                                 tf.cast(tf.stack([-1, self.num_node_features]), tf.int32))
        n_nodes = tf.cast(tf.shape(self.input['node_features'])[0], tf.int64)
        self.input['adj_mat_indices'] = tf.reshape(tf.decode_raw(features['adj_mat'], tf.int64),
                                                   tf.cast(tf.stack([-1, 2]), tf.int32))
        n_edges = tf.cast(tf.shape(self.input['adj_mat_indices'])[0]/2, tf.int64)
        self.input['adj_mat'] = tf.SparseTensor(indices=self.input['adj_mat_indices'],
                                                values=tf.ones(tf.reshape(tf.cast(2*n_edges, tf.int32), (1,)), tf.float32),
                                                dense_shape=tf.stack([n_nodes, n_nodes]))
        self.input['node_graph_map'] = tf.decode_raw(features['node_graph_map'], tf.int64)
        self.input['target'] = tf.reshape(tf.decode_raw(features['target'], tf.float32),
                                          tf.cast(tf.stack([-1, self.n_targets]), tf.int32))
        if self.getitem('config', 'use', prefix_edge) and self.num_edge_features > 0:
            self.input['edge_features'] = tf.reshape(tf.decode_raw(features['edge_features'], tf.float32),
                                                     tf.cast(tf.stack([-1, self.num_edge_features]), tf.int32))
            self.input['inc_mat_indices'] = tf.reshape(tf.decode_raw(features['inc_mat'], tf.int64),
                                                       tf.cast(tf.stack([2 * n_edges, 2]), tf.int32))
            self.input['inc_mat'] = tf.SparseTensor(indices=self.input['inc_mat_indices'],
                                                    values=tf.ones(tf.reshape(tf.cast(2*n_edges, tf.int32), (1,)), tf.float32),
                                                    dense_shape=tf.stack([n_nodes, n_edges]))

        # Finally, add dropout to inputs
        self.input['node_features_dropout'] = tf.nn.dropout(self.input['node_features'],
                                                            self.getitem('config', 'keep_prob', prefix_node))
        if self.getitem('config', 'use', prefix_edge) and self.num_edge_features > 0:
            self.input['edge_features_dropout'] = tf.nn.dropout(self.input['edge_features'],
                                                                self.getitem('config', 'keep_prob', prefix_edge))

    def graph_convolution_layer(self, node_emb, scope, edge_emb=None):
        # Path to hyperparameters and configuration settings for the graph convolutional layers
        prefix = 'model/graph_conv_layers'

        with tf.variable_scope(scope, reuse=not self.is_training):
            # Compute the extended node embedding as the concatenation of the original node embedding and the sum of
            # the node embeddings of all distance-one neighbors in the graph.
            ext_node_emb = tf.concat([node_emb, tf.sparse_tensor_dense_matmul(self.input['adj_mat'], node_emb)], axis=1)
            # If edge labels are to be considered by the model, concatenate as well the (pre-computed) sum of the
            # feature vectors labelling all edges connected to each node
            if edge_emb is not None:
                ext_node_emb = tf.concat([ext_node_emb, edge_emb], axis=1)

            # Compute output by applying a fully connected layer to the extended node embedding
            out = tf.contrib.layers.fully_connected(inputs=ext_node_emb,
                                                    num_outputs=self.getitem('config', 'num_outputs', prefix),
                                                    activation_fn=self.string_to_tf_act(self.getitem('config', 'activation_fn', prefix)),
                                                    weights_initializer=self.weights_initializer_graph_conv,
                                                    weights_regularizer=self.weights_regularizer_graph_conv,
                                                    biases_initializer=tf.constant_initializer(0.1, tf.float32),
                                                    normalizer_fn=self.normalizer_fn_graph_conv,
                                                    normalizer_params=self.normalizer_params_graph_conv,
                                                    trainable=self.getitem('config', 'trainable', prefix))

            # Apply dropout (if necessary). Alternatively, could have also forced keep_prob to 1.0 when is_training is
            # False
            if self.is_training:
                out = tf.nn.dropout(out, self.getitem('config', 'keep_prob', prefix))

        return out

    def output_embedding_layer(self, node_emb, scope):
        # Path to hyperparameters and configuration settings for the fingerprint output layers
        prefix = 'model/fingerprint_output_layers'

        with tf.variable_scope(scope, reuse=not self.is_training):
            # Compute node-level activation

            node_fp = tf.contrib.layers.fully_connected(inputs=node_emb,
                                                        num_outputs=self.getitem('config', 'num_outputs', prefix),
                                                        activation_fn=self.string_to_tf_act(self.getitem('config', 'activation_fn', prefix)),
                                                        weights_initializer=self.weights_initializer_fp_out,
                                                        weights_regularizer=self.weights_regularizer_fp_out,
                                                        biases_initializer=tf.constant_initializer(0.0, tf.float32),
                                                        trainable=self.getitem('config', 'trainable', prefix))

            # Apply dropout (if necessary). Alternatively, could have also forced keep_prob to 1.0 when is_training is
            # False
            if self.is_training:
                node_fp = tf.nn.dropout(node_fp, self.getitem('config', 'keep_prob', prefix))

            # Compute the graph-level activation as the sum of the node-level activations for all nodes in the graph
            graph_fp = tf.segment_sum(data=node_fp, segment_ids=self.input['node_graph_map'])

        return graph_fp, node_fp

    def build_graph_fingerprint(self):
        # Total number of graph convolution layers
        n_layers = self.getitem('config', 'model/graph_conv_layers/n_layers')

        # Create output dictionaries for graph convolutional layers and fingerprint output layers
        self.output['graph_conv_layers'] = {}
        self.output['fingerprint_output_layers'] = {}

        # Input node embeddings
        node_emb = self.input['node_features']

        # Pre-compute the sum of the feature vectors labelling all edges connected to each node (if necessary)
        self.output['graph_conv_layers']['edge_emb'] = None
        if self.getitem('config', 'model/input/edge_features/use') and self.num_edge_features > 0:
            self.output['graph_conv_layers']['edge_emb'] = tf.sparse_tensor_dense_matmul(self.input['inc_mat'],
                                                                                         self.input['edge_features'])
        # Compute node and graph level fingerprints for the input layer
        graph_fp, node_fp = self.output_embedding_layer(node_emb, 'output_embedding_layer_0')

        # List of node-level embeddings per layer (output of graph convolutional layers), node-level fingerprints per
        # layer (per-node output of fingerprint output layers) and graph-level fingerprints per layer (total output of
        # fingerprint output layers)
        self.output['graph_conv_layers']['node_emb'] = [node_emb]
        self.output['fingerprint_output_layers']['node_fp'] = [node_fp]
        self.output['fingerprint_output_layers']['graph_fp'] = [graph_fp]

        # Create all graph convolutional layers and their respective fingerprint output layers
        for layer_idx in xrange(1, n_layers+1):
            node_emb = self.graph_convolution_layer(node_emb=self.output['graph_conv_layers']['node_emb'][-1],
                                                    scope='graph_conv_layer_%d' % layer_idx,
                                                    edge_emb=self.output['graph_conv_layers']['edge_emb'])
            graph_fp, node_fp = self.output_embedding_layer(node_emb=self.output['graph_conv_layers']['node_emb'][-1],
                                                            scope='output_embedding_layer_%d' % layer_idx)
            # Append outputs to lists
            self.output['graph_conv_layers']['node_emb'].append(node_emb)
            self.output['fingerprint_output_layers']['node_fp'].append(node_fp)
            self.output['fingerprint_output_layers']['graph_fp'].append(graph_fp)

        # Obtain graph fingerprint as the sum of the graph activations across all layers
            self.output['fingerprint_output_layers']['fingerprint'] = tf.add_n(self.output['fingerprint_output_layers']['graph_fp'])

    def fully_connected_mlp_layer(self, input, scope):
        # Path to hyperparameters and configuration settings for the output MLP
        prefix = 'model/mlp'

        with tf.variable_scope(scope, reuse=not self.is_training):
            out = tf.contrib.layers.fully_connected(inputs=input,
                                                    num_outputs=self.getitem('config', 'num_outputs', prefix),
                                                    activation_fn=self.string_to_tf_act(self.getitem('config', 'activation_fn', prefix)),
                                                    weights_initializer=self.weights_initializer_mlp,
                                                    weights_regularizer=self.weights_regularizer_mlp,
                                                    biases_initializer=tf.constant_initializer(0.1, tf.float32),
                                                    normalizer_fn=self.normalizer_fn_mlp,
                                                    normalizer_params=self.normalizer_params_mlp)

            # Apply dropout (if necessary). Alternatively, could have also forced keep_prob to 1.0 when is_training is
            # False
            if self.is_training:
                out = tf.nn.dropout(out, self.getitem('config', 'keep_prob', prefix))

        return out

    def build_output_mlp(self):
        # Total number of hidden layers in the output MLP
        n_layers = self.getitem('config', 'model/mlp/n_layers')

        # Create output dictionary for output MLP activations
        self.output['mlp'] = {}

        # Input to output MLP
        mlp_act = self.getitem('output', 'fingerprint_output_layers/fingerprint')
        # List of output MLP activations per layer
        self.output['mlp']['act'] = [mlp_act]

        # Create all hidden layers in the output MLP
        for layer_idx in xrange(n_layers):
            mlp_act = self.fully_connected_mlp_layer(input=mlp_act,
                                                     scope='output_mlp_layer_%d' % (layer_idx+1))
            self.output['mlp']['act'].append(mlp_act)

        # Crate the final output layer (linear and without batch normalization)
        with tf.variable_scope('final_layer', reuse=not self.is_training):
            self.output['mlp']['out'] = tf.contrib.layers.fully_connected(inputs=mlp_act,
                                                                          num_outputs=self.n_targets,
                                                                          activation_fn=None,
                                                                          weights_initializer=self.weights_initializer_mlp,
                                                                          weights_regularizer=self.weights_regularizer_mlp,
                                                                          biases_initializer=tf.constant_initializer(0.0, tf.float32))

    def build_loss(self):
        # Compute the loss due to mismatch between predictions and targets
        self.losses['prediction'] = self.loss_fn(self.input['target'], self.output['mlp']['out'])
        tf.summary.scalar(name='Predictive_loss', tensor=self.losses['prediction'])
        # Compute the total contribution of the regularization terms to the loss
        self.losses['regularization'] = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar(name='Regularization_loss', tensor=self.losses['regularization'])
        # Compute the total loss of the model, including any regularization terms
        self.losses['total'] = self.losses['prediction'] + self.losses['regularization']
        tf.summary.scalar(name='Total_loss', tensor=self.losses['total'])

    def build_optimizer(self):
        if self.is_training:
            with tf.variable_scope('optimizer', reuse=False):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.getitem('config', 'train/learning_rate'))
                self.train_op = self.optimizer.minimize(self.losses['total'], global_step=self.global_step)

    def build_summaries(self):
        self.summary_op = tf.summary.merge_all()

    def build_init_op(self):
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def build_graph(self):
        self.build_inputs()
        self.build_graph_fingerprint()
        self.build_output_mlp()
        self.build_loss()
        self.build_optimizer()
        self.build_summaries()
        self.build_init_op()

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- INTERFACE --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def get_fingerprint(self):
        return self.getitem('output', 'fingerprint_output_layers/fingerprint')
