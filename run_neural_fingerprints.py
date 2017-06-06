import argparse
import json

import tensorflow as tf

from code.models.neural_fingerprints import NeuralFingerprints
from code.train_tools.trainer import Trainer
from code.train_tools.eval_losses import eval_losses_dict

# GLOBAL VARIABLES TO TRANSFORM INPUT STRINGS INTO TENSORFLOW FUNCTIONS
TRAIN_LOSSES = {'MSE': tf.losses.mean_squared_error,
                'softmax': tf.losses.softmax_cross_entropy,
                'sigmoid': tf.losses.sigmoid_cross_entropy,
                'hinge': tf.losses.hinge_loss}


def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="Path to file containing train batches in TFRecord format.", type=str)
    parser.add_argument("out_dir", help="Path to directory where results are to be stored.", type=str)
    parser.add_argument("train_loss", help="Loss function used to be used as training objective.", type=str,
                        choices=TRAIN_LOSSES.keys())
    parser.add_argument("eval_losses", nargs='+', help="Loss functions to be used for evaluating the model. "
                                                       "At least one must be provided. If multiple losses are provided, "
                                                       "the first one will be used to determine the best performing "
                                                       "model, while the others will still be kept track of in the "
                                                       "events files.", type=str,
                        choices=['MSE', 'RMSE', 'Corr', '0/1', 'AUC'])
    parser.add_argument("--val_path", help="Path to file containing validation batches in TFRecord format.", type=str)
    parser.add_argument("--tst_path", help="Path to file containing test batches in TFRecord format.", type=str)
    parser.add_argument("--num_epochs", help="Number of training epochs.", type=int, default=100)
    parser.add_argument("--config_path", help="Path to file containing configuration dictionary, in JSON format.",
                        type=str)
    parser.add_argument("--seed", help="Random seed, for reproducibility.", type=int, default=10)

    return parser.parse_args()


def main():
    # Parse input arguments first
    args = parse_input_arguments()

    # If a configuration file is provided, load the configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        #TODO: Implement method random_config for the NeuralFingerprints model, to be used for random search
        # hyperparameter selection
        config = NeuralFingerprints.random_config()

    # Add training loss to config dictionary
    config['model']['loss_fn'] = args.train_loss
    # Add random seed to config dictionary
    config['train']['seed'] = args.seed

    # Create NeuralFingerprints object for the training model
    model_tr = NeuralFingerprints(config=config, input_batches_file=args.train_path, is_training=True)

    # Create NeuralFingerprints object for the validation model (if necessary)
    model_val = None
    if args.val_path:
        model_val = NeuralFingerprints(config=config, input_batches_file=args.val_path, is_training=False)

    # Create NeuralFingerprints object for the test model (if necessary)
    model_tst = None
    if args.tst_path:
        model_tst = NeuralFingerprints(config=config, input_batches_file=args.tst_path, is_training=False)

    # Create Trainer object to run the main training loop
    trainer = Trainer(args.out_dir, model_tr, model_val, model_tst)

    # List of evaluation losses
    EVAL_LOSSES = eval_losses_dict()
    eval_losses = [(loss, EVAL_LOSSES[loss]) for loss in args.eval_losses]

    # Train model and evaluate it along the way (if necessary)
    trainer.train(continue_training=False, max_epochs=args.num_epochs, external_losses=eval_losses,
                  save_results_npy=True)
    # Close open files
    trainer.close_writers()


if __name__ == "__main__":
    main()
