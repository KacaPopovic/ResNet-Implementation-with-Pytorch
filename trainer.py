import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os
import numpy as np


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss

        self._optim.zero_grad()

        y_pred = self._model(x)
        y = y.type_as(y_pred)
        loss = self._crit(y_pred, y)
        loss.backward()
        self._optim.step()

        return loss.item(), y_pred

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        y_pred = self._model(x)
        y = y.type_as(y_pred)

        loss = self._crit(y_pred, y)
        return loss.item(), y_pred

    def train_epoch(self):
        # set training mode
        # self._train_dl.shuffle()
        self._model.train()
        total_loss = 0
        predictions = []
        labels = []
        # iterate through the training set
        for x, y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            loss, y_pred = self.train_step(x, y)
            total_loss += loss
            y_pred_binary = (y_pred > 0.5).float()

            predictions.append(y_pred_binary.cpu().numpy())
            labels.append(y.cpu().numpy())

        avg_loss = total_loss / len(self._train_dl)

        all_predictions = np.vstack(predictions)
        all_labels = np.vstack(labels)

        # Calculate F1 score using 'samples' averaging for multi-label classification
        mean_f1_score = f1_score(all_labels, all_predictions, average='samples', zero_division=0)

        return avg_loss, mean_f1_score

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()

        self._model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            # iterate through the validation set
            for x, y in self._val_test_dl:
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                # perform a validation step
                loss, y_pred = self.val_test_step(x, y)
                if self._cuda:
                    y_pred = y_pred.cpu()
                    y = y.cpu()
                total_loss += loss
                y_pred_binary = (y_pred > 0.5).float()

                # Accumulate flat lists of predictions and labels
                all_predictions.append(y_pred_binary.numpy())
                all_labels.extend(y.numpy())
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice.
        # You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # Convert lists of arrays into 2D arrays
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        avg_loss = total_loss / len(self._val_test_dl)

        # Calculate F1 score
        f1 = f1_score(all_labels, all_predictions, average='samples',
                      zero_division=0)  # adjust 'average' as needed for your task

        return avg_loss, f1

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses, val_losses = [], []
        train_f1_scores, val_f1_scores = [], []
        epoch = 0
        epochs_since_improvement = 0
        best_val_loss = float('inf')
        best_model_state = None  # To store the best model state

        while epoch < epochs:
            train_loss, train_f1 = self.train_epoch()
            val_loss, val_f1 = self.val_test()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1_scores.append(train_f1)
            val_f1_scores.append(val_f1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0  # Reset the counter
                # best_model_state = copy.deepcopy(self._model.state_dict())  # Save the best model state
                # Optionally save the model checkpoint here as well
            else:
                epochs_since_improvement += 1  # Increment the counter if no improvement

            print(
                f'Epoch {epoch}: Training Loss: {train_loss:.5f}, Validation Loss: {val_loss:.5f}, Training F1: {train_f1:.5f}, Validation F1: {val_f1:.5f}')

            if epochs_since_improvement >= self._early_stopping_patience:
                print("Early stopping triggered.")
                break

            epoch += 1

        # If best_model_state is not None, load it into the model
        # if best_model_state is not None:
        #   self._model.load_state_dict(best_model_state)

        return train_losses, val_losses, train_f1_scores, val_f1_scores, self._model