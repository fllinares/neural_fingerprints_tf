import copy, os, json
import numpy as np
import tensorflow as tf


class BasicModel(object):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ INITIALIZATION FUNCTIONS ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, config, input_batches_file, is_training):
        """
        Construct an object representing the model.
        :param config: Nested dictionary containing describing the model architecture and other hyperparameters.
        :param input_batches_file: Path to file containing batches in TFRecord format, encoded using class GraphDataset
        :param is_training: A boolean indicating whether the model is being built for training (True) or evaluation
        (False)
        """
        # Create a deepcopy of the configuration dictionary
        self.config = copy.deepcopy(config)

        # Retrieve meta-data from the input data file (e.g. number of batches, shape)
        self.parse_input(input_batches_file)

        # Whether the model is being built for training (is_training=True) or evaluation (is_training=False)
        self.is_training = is_training

        # Set non-initialised attributes to None. Each particular model implementation should take care of properly
        # initializing these attributes appropriately.

        # global_step variable for the model
        self.global_step = None
        # Variable initialization node in the computational graph
        self.init_op = None
        # Nodes returning input mini-batches as read from the input pipeline
        self.input = {'node_features': None, 'adj_mat_indices': None, 'adj_mat': None, 'node_graph_map': None,
                      'target': None, 'edge_features': None, 'inc_mat_indices': None, 'inc_mat': None}
        # Dictionary of operators computing the losses of the model
        self.losses = {}
        # Training operator: minimises self.total_loss
        self.optimizer = None
        self.train_op = None
        # Operator to execute all summary nodes in the computational graph
        self.summary_op = None
        # Dictionary of operators computing the outputs of the model (predictions and intermediate representations)
        self.output = {}

        # Parse configuration options
        self.parse_config()

        # Build computational graph
        self.build_graph()

    def parse_input(self, input_batches_file):
        """
        This method takes care of setting up the input file as an attribute of the model. It also computes the total
        number of batches in the input file and retrieves the shape information of the dataset (number of node features,
        edge features and targets) from the input file.
        :param input_batches_file: Path to file containing batches in TFRecord format, encoded using class GraphDataset
        """
        # Make sure the input file containing the batches in TFRecord format exists
        if not os.path.isfile(input_batches_file):
            raise ValueError('Input file %s not found.' % input_batches_file)
        self.input_batches_file = input_batches_file

        # Compute number of batches in file self.input_batches_file. In the current implementation, there is one
        # TFRecord per batch
        self.n_batches = 0
        for record in tf.python_io.tf_record_iterator(self.input_batches_file):
            self.n_batches += 1

        # Retrieve shape information from the first TFRecord in the file: number of node features, number of edge
        # features and number of targets
        example = tf.train.Example()
        example.ParseFromString(tf.python_io.tf_record_iterator(self.input_batches_file).next())
        shape = np.fromstring(example.features.feature['shape'].bytes_list.value[0], dtype=np.int64)
        self.num_node_features, self.num_edge_features, self.n_targets = shape[2:5]

    def parse_config(self):
        """
        This method parses the config dictionary passed to the constructor, extracting all necessary model parameters.
        """
        raise NotImplementedError('The parse_config method must be overriden by the model.')

    def random_config(self):
        """
        This method generates a random configuration dictionary, to be used for hyperparameter selection with random
        sampling in the hyperparameter space.
        :return: A python dictionary with the model configuration, generated at random.
        """
        raise NotImplementedError('The random_config method must be overriden by the model.')

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------ METHODS TO BUILD THE COMPUTATIONAL GRAPH ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def build_graph(self):
        """
        This method builds the entire computational graph: input pipeline from TFRecords, predictive model, loss
        functions, optimizer, summaries and initialization operation.
        """
        raise NotImplementedError('The build_graph method must be overriden by the model.')

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------- INTERFACE --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # GENERIC FUNCTION TO RETRIEVE AN ELEMENT FROM A NESTED DICTIONARY ATTRIBUTE
    def getitem(self, attr, key, prefix='', sep='/'):
        key = os.path.join(prefix, key)
        try:
            return reduce(dict.__getitem__, key.split(sep), getattr(self, attr))
        except KeyError:
            raise ValueError('Please specify a value for key %s in attribute %s.' % (key, attr))

    @staticmethod
    def load_config(filename):
        config = None
        with open(filename, 'r') as f:
            config = json.load(f)
        return config

    def dump_config(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=4, separators=(',', ':'), sort_keys=True)

    # INTERFACE FUNCTIONS REGARDING INPUT DATA
    def get_n_batches(self):
        return self.n_batches

    def get_inputs(self, keys=None):
        if keys is None:
            return self.input.values()
        else:
            return [self.input[key] for key in keys]

    # INTERFACE FUNCTIONS REGARDING MODEL
    @staticmethod
    def get_n_trainable_params():
        n_trainable_params = 0
        for var in tf.trainable_variables():
            n_trainable_params += np.prod(var.get_shape()).value
        return n_trainable_params

    def get_global_step(self):
        return self.global_step

    def get_init_op(self):
        return self.init_op

    def get_train_op(self):
        return self.train_op

    def get_summary_op(self):
        return self.summary_op

    def get_outputs(self, keys=None, prefix='', sep='/'):
        if keys is None:
            return self.output.values()
        else:
            return [self.getitem('output', key, prefix, sep) for key in keys]

    def get_losses(self, keys=None):
        if keys is None:
            return self.losses.values()
        else:
            return [self.losses[key] for key in keys]

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- AUXILIARY METHODS ------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def string_to_tf_loss(loss_str):
        TRAIN_LOSSES = {'mse': tf.losses.mean_squared_error,
                        'softmax': tf.losses.softmax_cross_entropy,
                        'sigmoid': tf.losses.sigmoid_cross_entropy,
                        'hinge': tf.losses.hinge_loss}
        # Case-insensitive behaviour
        loss_str_lc = loss_str.lower()
        if loss_str_lc in TRAIN_LOSSES:
            return TRAIN_LOSSES[loss_str_lc]
        else:
            raise ValueError('Loss %s is not yet supported.' % loss_str)

    @staticmethod
    def string_to_tf_act(act_str):
        ACTIVATION_FNS = {'relu': tf.nn.relu,
                          'crelu': tf.nn.crelu,
                          'relu6': tf.nn.relu6,
                          'elu': tf.nn.elu,
                          'softmax': tf.nn.softmax,
                          'softplus': tf.nn.softplus,
                          'softsign': tf.nn.softsign,
                          'sigmoid': tf.sigmoid,
                          'tanh': tf.tanh,
                          'identity': tf.identity}
        # Case-insensitive behaviour
        act_str_lc = act_str.lower()
        if act_str_lc in ACTIVATION_FNS:
            return ACTIVATION_FNS[act_str_lc]
        else:
            raise ValueError('Activation %s is not yet supported.' % act_str)
