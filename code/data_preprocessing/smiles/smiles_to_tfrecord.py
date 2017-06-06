import argparse, os, time
import numpy as np
from code.data_preprocessing.smiles.smiles_parser import SMILESParser
from code.data_preprocessing.graph_dataset import write_graphs_to_tfrecord


def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_file", help="File with one SMILES string per line.", type=str)
    parser.add_argument("output_file", help="Output file where TFRecords representing the graphs will be stored.",
                        type=str)
    parser.add_argument("config_file", help="JSON file with the configuration for the SMILES parser.", type=str)
    parser.add_argument("--targets_file",
                        help="File with a set of comma-separated targets per line. Number of lines must "
                             "match the number of lines in smiles_file.", type=str)
    args = parser.parse_args()

    return args


def main():
    # PARSE INPUT ARGUMENTS
    args = parse_input_arguments()

    # READ SMILES STRINGS
    print 'Reading SMILES strings...'
    tic = time.time()
    smiles_array = []
    with open(args.smiles_file, 'r') as f:
        for line in f:
            smiles_array.append(line.strip())
    n_smiles = len(smiles_array)
    toc = time.time()
    print 'Read %d SMILES strings in %0.3f seconds.\n' % (n_smiles, toc - tic)

    # READ TARGETS (IF ANY)
    targets_array, n_targets = None, None
    if args.targets_file is not None:
        print 'Reading targets...'
        tic = time.time()
        targets_array = np.loadtxt(args.targets_file, delimiter=',', dtype=np.float32)
        n_targets = len(targets_array)
        toc = time.time()
        print 'Read %d targets in %0.3f seconds.\n' % (n_targets, toc - tic)
    # Verify that there is one set of targets for each input SMILES string
    if targets_array is not None and n_smiles != n_targets:
        raise ValueError("smiles_file must have the same number of lines as targets_file")

    # CREATE GRAPH REPRESENTATION OF DATASET
    print 'Parsing SMILES strings...'
    tic = time.time()
    # Create SMILESParser object
    smiles_parser = SMILESParser(config=args.config_file)
    # Create a list of graph objects
    graphs = smiles_parser.parse_smiles(smiles_array=smiles_array, targets_array=targets_array)
    n_graphs = len(graphs)
    toc = time.time()
    print 'Parsed %d SMILES strings in %0.3f seconds.' % (n_smiles, toc - tic)
    print 'Parsing failed for %d/%d SMILES strings.\n' % (n_smiles - n_graphs, n_smiles)

    # CREATE OUTPUT DIRECTORY (IF IT DOES NOT EXIST)
    output_dir = os.path.split(args.output_file)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # CONVERT GRAPH REPRESENTATION TO TFRECORDS AND WRITE TO DISK
    print 'Writing graph data to TFRecords file...'
    tic = time.time()
    write_graphs_to_tfrecord(graphs, os.path.join(args.output_file))
    toc = time.time()
    print 'Wrote graph data to TFRecords file %s in %0.3f seconds.\n' % (args.output_file, toc - tic)

if __name__ == "__main__":
    main()
