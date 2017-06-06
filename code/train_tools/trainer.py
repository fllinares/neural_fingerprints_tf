import os, time
import numpy as np
import tensorflow as tf


class Trainer(object):

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ INITIALIZATION FUNCTIONS ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, out_dir, model_tr, model_val=None, model_tst=None, restore='no'):
        # Create output directory (if it does not exist)
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Store models and training parameters
        self.model_tr = model_tr
        self.model_val = model_val
        self.model_tst = model_tst

        # Create a TensorFlow Session
        self.sess = tf.Session()

        # Create a Saver object to backup the model after each training epoch is completed
        self.backup_saver = tf.train.Saver()
        # Create sub-directory for backup model checkpoints (if it does not exist)
        if not os.path.exists(os.path.join(self.out_dir, 'backup')):
            os.makedirs(os.path.join(self.out_dir, 'backup'))

        # If a validation model is provided, create a second Saver object to save the best performing model according
        # to the validation set loss. Used to implement early stopping
        if self.model_val is not None:
            self.best_model_saver = tf.train.Saver()
            # Create sub-directory for saving best performing model checkpoints (if it does not exist)
            if not os.path.exists(os.path.join(self.out_dir, 'best')):
                os.makedirs(os.path.join(self.out_dir, 'best'))

        # Create FileWriter objects to save summaries for training model as well as validation and test models
        # (if any)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.out_dir, 'train'), self.sess.graph)
        self.val_writer = None
        if self.model_val is not None:
            self.val_writer = tf.summary.FileWriter(os.path.join(self.out_dir, 'val'))
        self.tst_writer = None
        if self.model_tst is not None:
            self.tst_writer = tf.summary.FileWriter(os.path.join(self.out_dir, 'tst'))

        self.reset_all_counters()

        # Restore state (if necessary)
        if restore.lower() == 'backup':
            self.backup_saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(self.out_dir, 'backup')))
        elif restore.lower() == 'best':
            self.backup_saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join(self.out_dir, 'best')))

    def reset_all_counters(self):
        # Flag indicating whether the I/O pipeline has been initiated
        self.input_pipeline_started = False
        # Number of training batches processed and training epochs completed
        self.training_batches_seen, self.training_epochs_done = 0, 0
        # Number of validation batches processed and validation epochs completed
        self.validation_batches_seen, self.validation_epochs_done = 0, 0
        # Number of test batches processed and test epochs completed
        self.test_batches_seen, self.test_epochs_done = 0, 0
        # Epoch with best validation set performance, and the corresponding loss value. For early stopping.
        self.best_epoch, self.best_main_loss_val = 0, np.finfo(np.float).max
        # Loss value in the validation ad test sets (if any) per epoch
        self.val_main_loss_val_per_epoch, self.tst_main_loss_val_per_epoch = {}, {}

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- TERMINATION FUNCTIONS -----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __del__(self):
        self.close_writers()
        self.stop_input_pipeline()

    def close_writers(self):
        self.train_writer.close()
        if self.val_writer is not None:
            self.val_writer.close()
        if self.tst_writer is not None:
            self.tst_writer.close()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- I/O PIPELINE FUNCTIONS -----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def start_input_pipeline(self):
        # Do nt start the pipeline twice
        if not self.input_pipeline_started:
            # Start threads to fill input pipeline queues
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            # Mark pipeline as started
            self.input_pipeline_started = True

    def stop_input_pipeline(self):
        # Attempt to close pipeline only if started
        if self.input_pipeline_started:
            # Terminate threads taking care of input pipeline queues
            self.coord.request_stop()
            self.coord.join(self.threads)
            # Mark pipeline as closed
            self.input_pipeline_started = False

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- MAIN FUNCTIONS -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def eval(self, model):
        """
        Iterate through all mini-batches in a dataset, computing:
        (1) Average predictive loss (i.e. NLL) across all samples in the dataset
        (2) Regularization loss (i.e. negative log prior)
        (3) Total loss (i.e. loss used as a training objective by the model)

        Additionally, the function computes both targets and predictions for all samples in the dataset.

        The method returns a tf.Summary object containing the value values of all three losses, as well as the targets
        and model predictions for all samples in the dataset.

        :param model: An instantiated object of any class inheriting from BasicModel. It is assumed that the
        computational graph of the model has been already built and that all variables are initialised.
        :return: A tf.Summary object containing the value values of the average predictive loss, regularization loss
        and total loss across all samples in the dataset. Two NumPy arrays containing the targets and predictions
        respectively are also returned.
        """
        # Initialize variables
        batches_seen, n_samples = 0, 0
        pred_loss, reg_loss, total_loss = 0, 0, 0
        targets, predictions = [], []
        # Iterate across all model batches, accumulating output along the way
        while batches_seen < model.get_n_batches():
            # Compute loss (predictive, regularization and total), original targets and model predictions for the
            # mini-batch
            tensors = model.get_losses(keys=['prediction', 'regularization', 'total']) + \
                      model.get_inputs(keys=['target']) + \
                      model.get_outputs(keys=['mlp/out'])
            pred_loss_batch, reg_loss_batch, total_loss_batch, targets_batch, predictions_batch = self.sess.run(tensors)
            batch_size = targets_batch.shape[0]  # Handle potential variable-size mini-batches

            # Update losses
            pred_loss += batch_size*pred_loss_batch
            reg_loss += batch_size*reg_loss_batch  # Not strictly necessary, as it is constant across batches
            total_loss += batch_size*total_loss_batch

            # Update targets and predictions
            targets.append(targets_batch)
            predictions.append(predictions_batch)

            # Update number of examples
            n_samples += batch_size

            # Increase counter of batches processed
            batches_seen += 1

        # Create TensorFlow Summary, adding the three basic model losses
        summary = tf.Summary()
        summary.value.add(tag='Predictive_loss', simple_value=pred_loss/n_samples)
        summary.value.add(tag='Regularization_loss', simple_value=reg_loss/n_samples)
        summary.value.add(tag='Total_loss', simple_value=total_loss/n_samples)

        # Concatenate targets and predictions for each mini-batch into single NumPy arrays
        targets, predictions = np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0)

        return summary, targets, predictions

    @staticmethod
    def eval_external_losses(targets, predictions, external_losses, summary=None):
        """
        , using them to
        evaluate external loss functions. Each external loss function is represented as a tuple (name, loss_fn) where
        name is a string tagging the loss function and loss_fn is an arbitrary function of the form
        loss_val = loss_fn(targets, predictions), where targets and predictions are NumPy arrays with identical shape
        and loss_val an scalar.

        At least one such external loss function must be provided. Its value will be used to determine the best
        performing model. If len(external_losses) > 1, the additional external losses will also be evaluated.
        :param targets:
        :param predictions:
        :param external_losses: A list of tuples (name, loss_fn) of length at least 1. name must be a string tagging
        the loss and loss_fn must be a Python function with signature loss_val = loss_fn(targets, predictions), taking
        two NumPy arrays as inputs.
        :param summary: A TensorFlow Summary object to which the values of each external loss will be added as a scalar
        value. If no Summary is provided, the method will create an empty Summary and populate it with the values of
        the external losses.
        :return: A tf.Summary object containing the value values of all losses, as well as the scalar value of the
        first external loss in external_losses.
        """
        # Make sure at least one external loss has been provided
        if not external_losses:
            raise ValueError('At least one external loss must be provided as an input.')

        # If no TensorFlow summary is provided, create an empty one
        if summary is None:
            summary = tf.Summary()

        # For each provided evaluation loss, add the corresponding value to the evaluation Summary
        for i, (tag, loss_fn) in enumerate(external_losses):
            loss_val = loss_fn(targets, predictions)
            summary.value.add(tag=tag, simple_value=loss_val)
            # The first external loss is considered the main loss, used to determine the best performing model for
            # early stopping
            if i == 0:
                main_loss_val = loss_val

        return summary, main_loss_val

    def train(self, continue_training=False, max_epochs=500, external_losses=None, save_results_npy=True):
        # Initialise model
        if not continue_training:
            print 'BEGINNING MODEL TRAINING...\n'
            self.reset_all_counters()
            self.model_tr.dump_config(os.path.join(self.out_dir, 'config.json'))
            self.sess.run(self.model_tr.get_init_op())
        else:
            print 'RESUMING MODEL TRAINING FROM EPOCH %d...\n' % self.training_epochs_done

        # Start input pipeline
        self.start_input_pipeline()

        # Timing reference
        outer_tic = time.time()

        # Main training loop
        while self.training_epochs_done < max_epochs:
            # Do an entire training epoch
            print 'Starting training epoch %d/%d...' % (self.training_epochs_done+1, max_epochs)
            inner_tic = time.time()
            epoch_completed = False
            while not epoch_completed:
                # Perform a training step
                _, summary_str = self.sess.run([self.model_tr.get_train_op(), self.model_tr.get_summary_op()])
                # Update counter of batches seen and check if an epoch as been completed
                self.training_batches_seen += 1
                # Add the corresponding summary
                self.train_writer.add_summary(summary_str, self.training_batches_seen)
                epoch_completed = (self.training_batches_seen % self.model_tr.get_n_batches()) == 0
            # Update counter of epochs completed
            self.training_epochs_done += 1
            inner_toc = time.time()
            print 'Completed training epoch %d/%d in %0.3f seconds.\n' \
                  % (self.training_epochs_done, max_epochs, inner_toc - inner_tic)

            if self.model_val is not None:
                print '\tEvaluating performance in validation set...'
                inner_tic = time.time()
                # Obtain summary containing evaluation performance
                summary, targets, predictions  = self.eval(self.model_val)
                summary, main_loss_val = self.eval_external_losses(targets, predictions, external_losses, summary)
                self.val_writer.add_summary(summary, self.training_batches_seen)
                self.val_main_loss_val_per_epoch[self.training_epochs_done] = main_loss_val
                # Increase counters
                self.validation_batches_seen += self.model_val.get_n_batches()
                self.validation_epochs_done += 1
                inner_toc = time.time()
                print '\tValidation set performance evaluated in %0.3f seconds.' % (inner_toc - inner_tic)
                print '\tValidation %s: %0.3f.\n' % (external_losses[0][0], main_loss_val)
                # Early stopping: save the model if the main external loss in the validation set is smaller than the
                # best value recorded so far
                if main_loss_val < self.best_main_loss_val:
                    self.best_epoch, self.best_main_loss_val = self.training_epochs_done, main_loss_val
                    self.best_model_saver.save(sess=self.sess,
                                               save_path=os.path.join(self.out_dir, 'best', 'model.ckpt'),
                                               global_step=self.training_batches_seen)

            if self.model_tst is not None:
                print '\tEvaluating performance in test set...'
                inner_tic = time.time()
                # Obtain summary containing evaluation performance
                summary, targets, predictions = self.eval(self.model_tst)
                summary, main_loss_val = self.eval_external_losses(targets, predictions, external_losses, summary)
                self.tst_writer.add_summary(summary, self.training_batches_seen)
                self.tst_main_loss_val_per_epoch[self.training_epochs_done] = main_loss_val
                # Increase counters
                self.test_batches_seen += self.model_tst.get_n_batches()
                self.test_epochs_done += 1
                inner_toc = time.time()
                print '\tTest set performance evaluated in %0.3f seconds.\n' % (inner_toc - inner_tic)
                print '\tTest %s: %0.3f.\n' % (external_losses[0][0], main_loss_val)

            # Save model
            print '\tBacking up the current model...'
            inner_tic = time.time()
            self.backup_saver.save(sess=self.sess, save_path=os.path.join(self.out_dir, 'backup', 'model.ckpt'),
                                   global_step=self.training_batches_seen)
            inner_toc = time.time()
            print '\tBacked-up model in %0.3f seconds.\n' % (inner_toc - inner_tic)

        # Termine input pipeline
        self.stop_input_pipeline()

        outer_toc = time.time()
        print 'TRAINING COMPLETED IN %0.3f SECONDS.\n' % (outer_toc - outer_tic)
        if self.model_val is not None and self.model_tst is not None:
            print 'Best test %s: %0.3f, achieved in epoch %d.\n' \
                  % (external_losses[0][0], self.tst_main_loss_val_per_epoch[self.best_epoch], self.best_epoch)
            # Sanity-check
            assert len(self.val_main_loss_val_per_epoch) == len(self.tst_main_loss_val_per_epoch)
            results = np.stack((self.val_main_loss_val_per_epoch.values(), self.tst_main_loss_val_per_epoch.values()))
            if save_results_npy:
                np.save(os.path.join(self.out_dir, 'main_loss_results'), results)