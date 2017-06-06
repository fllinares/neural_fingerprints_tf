import argparse, os, time, operator
import numpy as np
from code.data_preprocessing.graph_dataset import make_batches, read_graphs_from_tfrecord, write_graphs_to_tfrecord


def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecords_file", help="File with one TFRecord per graph in the dataset.", type=str)
    parser.add_argument("tfrecords_path", help="Directory where TFRecords representing the graph mini-batches will be "
                                               "stored.", type=str)
    parser.add_argument("batch_size", help="Number of graphs per batch.", type=int)
    # Train/Val/Tst splits can be specified either with an input file containing the indices of the graphs to be
    # included in each dataset or via desired proportions. In the latter case, a random partition will be created
    split_args = parser.add_mutually_exclusive_group(required=True)
    split_args.add_argument("--split_indices_file", type=str, help="File containing indices of graphs assigned to "
                                                                   "training, validation and test datasets, in that "
                                                                   "order. One row per split (can be empty).")
    split_args.add_argument("--split_proportions", type=float, nargs=3, help="Proportion of graphs to be used for "
                                                                             "training, validation and test, in that "
                                                                             "order. Must add up to 1.0.")
    parser.add_argument("--seed", type=int, help="Seed for tr/val/tst split. Unused if split indices are provided.",
                        default=0)
    args = parser.parse_args()

    return args


def read_splits(filename):
    # Check whether the file exists, raising an error if not
    if os.path.isfile(filename):
        # For now, we assume that the file format is correct if it exists
        with open(filename, 'r') as f:
            tr_idx = np.array(map(int, f.readline().strip().split()))
            val_idx = np.array(map(int, f.readline().strip().split()))
            tst_idx = np.array(map(int, f.readline().strip().split()))
    else:
        raise ValueError('Train/Val/Tst indices file %s does not exist.' % filename)

    return tr_idx, val_idx, tst_idx


def make_random_splits(n_graphs, p_tr, p_val, p_tst, random_state=0):
    # Make sure the proportions are positive and add up to 1.0
    if not (p_tr > 0 and p_val > 0 and p_tst > 0 and np.allclose(p_tr + p_val + p_tst, 1.0)):
        raise ValueError('Train/Val/Tst proportions must be positive and add up to 1.0.')

    # Set random seed
    np.random.seed(random_state)

    # Number of training, validation and test graphs
    n_tr, n_val = int(p_tr * n_graphs), int(p_val * n_graphs)
    n_tst = max(0, n_graphs - (n_tr + n_val))

    # Random permutation of graph indices
    perm = np.random.permutation(n_graphs)
    tr_idx, val_idx, tst_idx = np.sort(perm[:n_tr]), np.sort(perm[n_tr:(n_tr + n_val)]), np.sort(perm[(n_tr + n_val):])

    return tr_idx, val_idx, tst_idx


def load_graphs(filename):
    print 'Reading graph representations from TFRecords file %s...' % filename
    tic = time.time()
    graphs = read_graphs_from_tfrecord(filename)
    n_graphs = len(graphs)
    toc = time.time()
    print 'Read %d graphs in %0.3f seconds.\n' % (n_graphs, toc - tic)

    return graphs, n_graphs


def prepare_minibatches(graphs, batch_size, tr_idx, val_idx=None, tst_idx=None, random_state=0):
    # Prepare training set batches
    print 'Batching training set graphs...'
    tic = time.time()
    graphs_tr = operator.itemgetter(*tr_idx)(graphs)
    batches_tr = make_batches(graphs_tr, batch_size, 2*(random_state + 1))
    toc = time.time()
    print 'Batched %d training set graphs into %d batches in %0.03f seconds.\n' \
          % (len(graphs_tr), len(batches_tr), toc - tic)

    # Prepare validation set batches (if any)
    if val_idx is not None and len(val_idx) > 0:
        print 'Batching validation set graphs...'
        tic = time.time()
        graphs_val = operator.itemgetter(*val_idx)(graphs)
        batches_val = make_batches(graphs_val, batch_size, 3*(random_state + 1))
        toc = time.time()
        print 'Batched %d validation set graphs into %d batches in %0.03f seconds.\n' \
              % (len(graphs_val), len(batches_val), toc - tic)
    else:
        batches_val = None

    # Prepare test set batches (if any)
    if tst_idx is not None and len(tst_idx) > 0:
        print 'Batching test set graphs...'
        tic = time.time()
        graphs_tst = operator.itemgetter(*tst_idx)(graphs)
        batches_tst = make_batches(graphs_tst, batch_size, 4*(random_state + 1))
        toc = time.time()
        print 'Batched %d test set graphs into %d batches in %0.03f seconds.\n' \
              % (len(graphs_tst), len(batches_tst), toc - tic)
    else:
        batches_tst = None

    return batches_tr, batches_val, batches_tst


def write_minibatches_to_disk(batches, filename, dataset_type=''):
    print 'Writing %s batches to TFRecords file...' % dataset_type
    tic = time.time()
    write_graphs_to_tfrecord(batches, filename)
    toc = time.time()
    print 'Wrote %s batches to TFRecords file %s in %0.3f seconds.\n' % (dataset_type, filename, toc - tic)


def main():
    # PARSE INPUT ARGUMENTS
    args = parse_input_arguments()

    # LOAD GRAPHS FROM TFRECORDS FILE
    graphs, n_graphs = load_graphs(args.tfrecords_file)

    # MAKE TRAINING/VALIDATION/TEST SPLITS
    if args.split_indices_file is not None:
        tr_idx, val_idx, tst_idx = read_splits(args.split_indices_file)
    elif args.split_proportions is not None:
        p_tr, p_val, p_tst = args.split_proportions
        tr_idx, val_idx, tst_idx = make_random_splits(n_graphs, p_tr, p_val, p_tst, random_state=args.seed)

    # PREPARE MINI-BATCHES
    batches_tr, batches_val, batches_tst = prepare_minibatches(graphs, args.batch_size, tr_idx, val_idx, tst_idx,
                                                               random_state=args.seed)

    # CREATE OUTPUT DIRECTORY (IF IT DOES NOT EXIST)
    if not os.path.exists(args.tfrecords_path):
        os.makedirs(args.tfrecords_path)

    # WRITE MINI-BATCHES TO DISK
    write_minibatches_to_disk(batches_tr, os.path.join(args.tfrecords_path, 'batches_tr.tfrecords'),
                              dataset_type='training')
    if batches_val is not None:
        write_minibatches_to_disk(batches_val, os.path.join(args.tfrecords_path, 'batches_val.tfrecords'),
                                  dataset_type='validation')
    if batches_tst is not None:
        write_minibatches_to_disk(batches_tst, os.path.join(args.tfrecords_path, 'batches_tst.tfrecords'),
                                  dataset_type='test')

if __name__ == "__main__":
    main()
