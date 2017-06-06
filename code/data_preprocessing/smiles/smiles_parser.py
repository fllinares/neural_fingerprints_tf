import os, copy, json
import numpy as np
# RDKit imports for parsing SMILES strings to create graphs
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

# ------------------------------------ GLOBAL VARIABLES -------------------------------------

# List of atomic symbols that will be recognized by the model, allocating a category for others
ALLOWABLE_ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                          'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                          'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Other']
# Total number of allowable, distinct atom symbols (including the category 'Other')
NUM_ALLOWABLE_ATOM_SYMBOLS = len(ALLOWABLE_ATOM_SYMBOLS)
# Maximum degree an atom can have according to the model
MAX_ATOM_DEGREE = 5
# Maximum number of implicit hydrogen atoms an atom can have according to the model
MAX_ATOM_NUM_HS = 4
# Maximum implicit valence an atom can have according to the model
MAX_ATOM_IMPLICIT_VALENCE = 5

# List of bond types that will be recognized by the model, allocating a category for others
ALLOWABLE_BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                        Chem.rdchem.BondType.AROMATIC, 'Other']
# Total number of allowable, distinct bond types (including the category 'Other')
NUM_ALLOWABLE_BOND_TYPES = len(ALLOWABLE_BOND_TYPES)


class SMILESParser(object):

    def __init__(self, config):
        # If config is a Python dictionary, create a deepcopy
        if isinstance(config, dict):
            self.config = copy.deepcopy(config)
        # Otherwise, check if config is a valid file (assumed to be a JSON file containing the config dictionary)
        elif os.path.isfile(config):
            with open(config, 'r') as f:
                self.config = json.load(f)
        # Else, raise an exception
        else:
            raise ValueError('Invalid configuration. Please specify either a Python dict or the path to a JSON file.')

        # Compute number of node and edge features
        self.num_node_features = self.get_num_node_features()
        self.num_edge_features = self.get_num_edge_features()

    def parse_smiles(self, smiles_array, targets_array=None, ids_array=None, verbose=False):
        # Number of SMILES strings to be parsed
        n_smiles = len(smiles_array)
        # If targets are provided, verify that the number of SMILES strings match the number of targets provided
        if targets_array is not None:
            if len(targets_array) != n_smiles:
                raise ValueError("smiles_array and targets_array must have the same number of elements.")
        # Otherwise, make targets_array a list of None with length matching the number of SMILES strings
        else:
            targets_array = n_smiles*[None]
        # If ids are provided, verify that the number of SMILES strings match the number of ids provided
        if ids_array is not None:
            if len(ids_array) != n_smiles:
                raise ValueError("smiles_array and ids_array must have the same number of elements.")
        # Otherwise, assign consecutive integer IDs to the SMILES strings
        else:
            ids_array = np.arange(n_smiles)

        # Initialise empty list of parsed molecular graphs
        graphs = []
        for i in xrange(n_smiles):
            # Attempt to parse the SMILES string, summarising it as a Python dict
            g = self.parse_smiles_str(smiles_str=smiles_array[i], id=ids_array[i], target=targets_array[i])
            # If the parsing attempt was successful, append it to the list of graphs
            if g is not None:
                graphs.append(g)
            # Otherwise, report the error (if verbose is enabled)
            elif verbose:
                print 'Parsing failed for SMILES string %d/%d, with value %s...' % (i+1, n_smiles, smiles_array[i])

        return graphs

    def parse_smiles_str(self, smiles_str, id, target=None):
        # Use RDKit to parse SMILES string
        mol = MolFromSmiles(smiles_str)
        if not mol:
            return None

        # Represent Hydrogen atoms explicity (if necessary)
        if self.config['explicit_Hs']:
            mol = Chem.AddHs(mol)

        # Compute number of nodes (atoms) and edges (bonds)
        n_nodes, n_edges = mol.GetNumAtoms(), mol.GetNumBonds()

        # Allocate space for Numpy arrays representing the molecular graph
        node_features = np.zeros((n_nodes, self.num_node_features), dtype=np.float32)
        edge_features = np.zeros((n_edges, self.num_edge_features), dtype=np.float32)
        adj_mat = np.zeros((2*n_edges, 2), dtype=np.int64)  # Adjacency matrix (sparse representation)
        inc_mat = np.zeros((2*n_edges, 2), dtype=np.int64)  # Incidence matrix (sparse representation)

        # Retrieve node (atom) features, if needed
        if self.num_node_features > 0:
            for i, atom in enumerate(mol.GetAtoms()):
                node_features[i] = self.get_node_features(atom)

        # Retrieve edges (bonds)
        for i, bond in enumerate(mol.GetBonds()):
            # Fill in the two pairs of indices this edge (bond) contributes to the adjacency matrix
            adj_mat[2*i] = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            adj_mat[2*i+1] = [bond.GetEndAtom().GetIdx(), bond.GetBeginAtom().GetIdx()]
            # Fill in the two pairs of indices this edge (bond) contributes to the incidence matrix
            inc_mat[2*i] = [bond.GetBeginAtom().GetIdx(), i]
            inc_mat[2*i+1] = [bond.GetEndAtom().GetIdx(), i]

            # Retrieve edge (bond) features, if needed
            if self.num_edge_features > 0:
                edge_features[i] = self.get_edge_features(bond)

        # Sort the adjacency and incidence matrices lexicographically
        adj_mat = adj_mat[np.lexsort((adj_mat[:, 1], adj_mat[:, 0]))]
        inc_mat = inc_mat[np.lexsort((inc_mat[:, 1], inc_mat[:, 0]))]

        # Represent molecular graph as a dictionary
        g = {'node_features': node_features, 'edge_features': edge_features, 'adj_mat': adj_mat, 'inc_mat': inc_mat}

        # Add target(s) (if any), making sure they are a NumPy array object with method tobytes()
        if target is not None:
            # Convert scalars to NumPy array
            if not isinstance(target, np.ndarray):
                target = np.array(target, np.float32)

            # Ensure target is of type np.float32
            target = target.astype(np.float32)

            # Flatten targets of rank >= 2
            if target.ndim > 1:
                target = target.flatten()

            # Store target as a (row) 2D NumPy array (for compatibility)
            g['target'] = np.reshape(target, (1, -1))
            n_targets = g['target'].shape[1]
        # If there are no targets, add an empty NumPy array (for compatibility)
        else:
            g['target'] = np.zeros((1, 0), dtype=np.float32)
            n_targets = 0

        # Add ID, making sure it is a NumPy array object with method tobytes()
        if not isinstance(target, np.ndarray):
            id = np.array(id, np.int64)
        g['id'] = id

        # Finally, add shape information. The last element refers to the number of graphs, and is included for
        # compatibility with batched graphs
        g['shape'] = np.array((n_nodes, n_edges, self.num_node_features, self.num_edge_features, n_targets, 1),
                              np.int64)

        return g

    def get_num_node_features(self):
        # Initialise number of node features to zero
        num_node_features = 0

        # Number of features used to encode atom symbol
        num_node_features += self.count_num_features(self.config['atom']['symbol'], NUM_ALLOWABLE_ATOM_SYMBOLS,
                                                     "atom symbol")
        # Number of features used to encode atom degree
        num_node_features += self.count_num_features(self.config['atom']['degree'], MAX_ATOM_DEGREE + 1, "atom degree")
        # Number of features used to encode the number of implicit hydrogen atoms
        num_node_features += self.count_num_features(self.config['atom']['num_Hs'], MAX_ATOM_NUM_HS + 1, "atom num Hs")
        # Number of features used to encode the implicit valence of the atom
        num_node_features += self.count_num_features(self.config['atom']['implicit_valence'],
                                                     MAX_ATOM_IMPLICIT_VALENCE + 1, "atom implicit valence")
        # Number of features used to encode the aromaticity of the atom
        num_node_features += self.count_num_features(self.config['atom']['aromaticity'], 2, "atom aromaticity")

        return num_node_features

    def get_num_edge_features(self):
        # Initialise number of edge features to zero
        num_edge_features = 0

        # Number of features used to encode the bond type
        num_edge_features += self.count_num_features(self.config['bond']['type'], NUM_ALLOWABLE_BOND_TYPES, "bond type")
        # Number of features used to encode whether the bond is conjugated or not
        num_edge_features += self.count_num_features(self.config['bond']['is_conjugated'], 2, "bond is conjugated")
        # Number of features used to encode whether the bond is in a ring or not
        num_edge_features += self.count_num_features(self.config['bond']['is_in_ring'], 2, "bond is in ring")

        return num_edge_features

    def get_node_features(self, atom):
        # Initialise atom feature vector as an empty list
        node_features = []

        # Encode atom symbol as a feature vector
        atom_symbol = atom.GetSymbol()
        # If the atom symbol is unrecognized by the model, set it to 'Other'
        if atom_symbol not in ALLOWABLE_ATOM_SYMBOLS:
            atom_symbol = 'Other'
        # Retrieve (arbitrary) integer encoding of the atom symbol
        atom_category = ALLOWABLE_ATOM_SYMBOLS.index(atom_symbol)
        # Append feature vector corresponding to atom symbol to the list of node-specific features
        node_features += self.encode_feature(self.config['atom']['symbol'], atom_category, NUM_ALLOWABLE_ATOM_SYMBOLS,
                                             "atom symbol")

        # Encode the degree of the atom
        atom_degree = atom.GetDegree()
        # Append feature vector corresponding to atom degree to the list of node-specific features
        node_features += self.encode_feature(self.config['atom']['degree'], atom_degree, MAX_ATOM_DEGREE+1,
                                             "atom degree")

        # Encode the number of implicit hydrogen atoms
        atom_num_Hs = atom.GetTotalNumHs()
        # Append feature vector corresponding to number of implicit hydrogen atoms to the list of node-specific
        # features
        node_features += self.encode_feature(self.config['atom']['num_Hs'], atom_num_Hs, MAX_ATOM_NUM_HS+1,
                                             "atom num Hs")

        # Encode the implicit valence of the atom
        atom_implicit_valence = atom.GetImplicitValence()
        # Append feature vector corresponding to the implicit valence of the atom to the list of node-specific
        # features
        node_features += self.encode_feature(self.config['atom']['implicit_valence'], atom_implicit_valence,
                                             MAX_ATOM_IMPLICIT_VALENCE+1, "atom implicit valence")

        # Encode whether the atom is aromatic or not
        atom_aromaticity = atom.GetIsAromatic()
        # Append feature vector corresponding to the aromaticity of the atom to the list of node-specific features
        node_features += self.encode_feature(self.config['atom']['aromaticity'], atom_aromaticity, 2,
                                             "atom aromaticity")

        # Check that feature vector is not empty
        if len(node_features) == 0:
            raise ValueError("Node feature vector is empty. Please select at least one source of node features.")

        return np.array(node_features, dtype=np.float32)

    def get_edge_features(self, bond):
        # Initialise bond feature vector as an empty list
        edge_features = []

        # Encode bond type as a feature vector
        bond_type = bond.GetBondType()
        # If the bond type is unrecognized by the model, set it to 'Other'
        if bond_type not in ALLOWABLE_BOND_TYPES:
            bond_type = 'Other'
        # Retrieve (arbitrary) integer encoding of the bond type
        bond_category = ALLOWABLE_BOND_TYPES.index(bond_type)
        # Append feature vector corresponding to bond type to the list of edge-specific features
        edge_features += self.encode_feature(self.config['bond']['type'], bond_category, NUM_ALLOWABLE_BOND_TYPES,
                                             "bond type")

        # Encode whether the bond is conjugated or not
        bond_is_conjugated = bond.GetIsConjugated()
        # Append the corresponding features to the list of edge-specific features
        edge_features += self.encode_feature(self.config['bond']['is_conjugated'], bond_is_conjugated, 2,
                                             "bond is conjugated")

        # Encode whether the bond is in a ring or not
        bond_is_in_ring = bond.IsInRing()
        # Append the corresponding features to the list of edge-specific features
        edge_features += self.encode_feature(self.config['bond']['is_in_ring'], bond_is_in_ring, 2,  "bond is in ring")

        # Check that feature vector is not empty
        if len(edge_features) == 0:
            raise ValueError("Edge feature vector is empty. Please select at least one source of edge features.")

        return np.array(edge_features, dtype=np.float32)

    @staticmethod
    def encode_feature(mode, val, n_vals, feat_name):
        # Make the function case-insensitive
        mode = mode.lower()

        # Sanity-check
        if val > n_vals:
            raise ValueError("%s: %d. Maximum %s recognised by the model: %d." % (feat_name, val, feat_name, n_vals-1))

        # Encode feature as an integer
        if mode == 'category':
            feature_vector = [val]
        # Encode feature as a one-hot vector
        elif mode == 'one-hot':
            feature_vector = n_vals*[0]
            feature_vector[val] = 1
        # Discard the feature
        elif mode == 'none':
            feature_vector = []
        # Unrecognised option
        else:
            raise ValueError("Unrecognised encoding format %s for %s." % (mode, feat_name))

        return feature_vector

    @staticmethod
    def count_num_features(mode, n_vals, feat_name):
        # Make the function case-insensitive
        mode = mode.lower()

        # Add one feature if it is to be encoded as an integer
        if mode == 'category':
            num_features = 1
        # Add as many features as allowable categories if it is to be encoded as a one-hot vector
        elif mode == 'one-hot':
            num_features = n_vals
        # Add no features if it is not to be included in the feature representation
        elif mode == 'none':
            num_features = 0
        # Unrecognised option
        else:
            raise ValueError("Unrecognised encoding format %s for %s." % (mode, feat_name))

        return num_features



