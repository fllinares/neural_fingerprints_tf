This is an independent TensorFlow implementation of the approach described in the following paper:

+ D. Duvenaud\*, D. Maclaurin\*, J. Aguilera-Iparraguirre, R. Gomez-Bombarelli, T. Hirzel, A. Aspuru-Guzik and R. P. Adams.
**Convolutional Networks on Graphs for Learning Molecular Fingerprints**, *NIPS 2015*. \*Equal contributions.

The article can be found [here](https://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints), and the original implementation by the paper author's [here](https://github.com/HIPS/neural-fingerprint).

## Dependencies

The following packages are required:
+ [NumPy](http://www.numpy.org/), tested on version `1.12.1`
+ [RDKit](http://www.rdkit.org/), tested on version `2017.03.1.dev1`
+ [scikit-learn](http://scikit-learn.org/stable/), tested on version `0.18.1`
+ [TensorFlow](https://www.tensorflow.org/), tested on version `1.2.0-rc0`

## Input data format

This implementation assumes that a dataset is represented by two input files:
+ A text file describing the different molecules in the dataset. It is assumed
to contain one SMILES string, i.e. sample, per line.
+ A text file describing the corresponding target vector for each molecule in
the dataset. It is assumed to contain a comma-separated list of k >= 1 scalars 
per line, i.e. a k-dimensional target vector for each SMILES string.

#### Example datasets

Four example datasets (`cep`, `delaney`, `malaria` and `toxin`) , can be 
downloaded using the scripts provided in `code/scripts/data_download`:

```
export NF_HOMEDIR=<local_path/to/repo_folder>
bash code/scripts/data_download/download_<dataset_name>.sh
```

The scripts will store the two aforementioned text files, `smiles.dat` and 
`targets.dat`, in folder `data/smiles/<dataset_name>`. Datasets `cep`, `delaney` 
and `malaria` are 1-D regression problems while `toxin` is a binary 
classification problem.

All these four datasets are kindly provided by the authors of the algorithm in 
the [original repo](https://github.com/HIPS/neural-fingerprint/tree/master/data).
For `cep` and `malaria`, the authors used only a subset of the corresponding
datasets for their experiments. The download scripts refer to these subsets by
adding the prefix `reduced_` to `<dataset_name>`. Moreover, versions of `cep`
and `reduced_cep` without outliers (defined as samples with target value 0.0) 
are also provided.

## Data preprocessing

#### Parsing

This implementation relies on an internal TFRecord representation of the input 
molecular graphs and their corresponding targets.

The script`code/data_preprocessing/smiles/smiles_to_tfrecord.py` takes care of:
+ Parsing the two input text files, generating a list of Python dicts 
representing each graph in the dataset.
+ Converting each Python dict to the internal TFRecord format and writing all 
records to disk.

The SMILES parser can be configured by providing a JSON file describing the 
desired settings. Example configurations can be found in the directory `code/data_preprocessing/smiles/smiles_parser_config`. In particular, this 
allows the user to control which features are used to represent atoms and bonds and how they are encoded into node and edge feature vectors.

Example scripts to parse the four provided datasets can be found in `code/scripts/data_preprocessing`:

```
export NF_HOMEDIR=<local_path/to/repo_folder>
bash code/scripts/data_preprocessing/parse_<dataset_name>.sh
```

By default, these scripts will write the TFRecords to file `data/tfrecords/<dataset_name>/all.tfrecords`.

#### Batching

The next step in the preprocessing pipeline is to generate a training/validation/test split of the input dataset and precomputing mini-batches for each split of the
data. The current implementation essentially treats a mini-batch of graphs as a
single graph with multiple connected components. To avoid repeating the graph
concatenation operation for each mini-batch each epoch, mini-batches are to be pre-computed beforehand.

The script `code/data_preprocessing/batch_tfrecords.py` takes care of:
+ Generating a random training/validation/test split of the input dataset with
specified proportions or, alternatively, reading split indices from an input file.
+ Generating mini-batches of the specified size for each of the splits and writing
them to disk in the internal TFRecord format.

Example scripts to batch the four provided datasets can be found in `code/scripts/data_batching`:

```
export NF_HOMEDIR=<local_path/to/repo_folder>
bash code/scripts/data_batching/batch_<dataset_name>.sh
```

By default, these scripts will write the TFRecords to file `data/batched_tfrecords/<dataset_name>/batches_<split>.tfrecords`, where `split`
can be `tr` (train), `val` (validation) or `tst` (test).

## Building and training the model

File `code/models/neural_fingerprints.py` contains the Python class responsible 
for building the computational graph. Its constructor takes as input:
+ The path to a JSON file specifying the model hyperparameters. Please see 
the example configuration files included in folders `model_configs/<dataset_name>` 
for a reference.
+ The path to a file containing the pre-computed mini-batches of a dataset 
split, in the internal TFRecord format (i.e. the output file generated by script `code/data_preprocessing/batch_tfrecords.py`).
+ A boolean indicating whether the model is to be used for training (train split)
or evaluation (validation and test splits).

In the current implementation, the training model needs to be built first. The 
validation and test models are expected to be built afterwards, as they employ 
variable reuse to tie their model weights and biases to those of the training 
model.

Additionally, `code/train_tools/trainer.py` contains a Python class that handles
the training and evaluation loops. Its constructor takes as input the already
constructed training model and, optionally, already constructed validation and 
test models. It also allows restoring a previously trained model saved in disk.

Finally, the script `run_neural_fingerprints.py` offers a wrapper that handles
both model construction and training using the two aforementioned Python classes.

Example scrips to a train the model on the four provided datasets can be found in `code/scripts/experiments`:

```
export NF_HOMEDIR=<local_path/to/repo_folder>
bash code/scripts/experiments/launch_<dataset_name>.sh
```

By default, these scripts will store the output files in folder `output/<dataset_name>`. 
This folder will contain:
+ A subdirectory `backup`, with the five most recent trained models (the model is saved after each epoch).
+ A subdirectory `best`, with the five best performing models according to validation loss (to be used for early-stopping).
+ A file `config.json`, identical to the provided input configuration file (to help keep track of hyperparameters).
+ Subdirectories `train`, `val` and `tst` with the corresponding TensorFlow summaries.

## Authors
This implementation was written by [Felipe Llinares-Lopez](https://scholar.google.ch/citations?user=zzjTWUUAAAAJ&hl=en). It is based on the software written by David Duvenaud, Dougal Maclaurin and Ryan P. Adams. If you happen to use this TensorFlow implementation, please don't forget to cite their article.

