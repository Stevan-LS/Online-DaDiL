r"""Pytorch Lightning models module."""

import torch
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torchmetrics.functional.classification import binary_accuracy


class DeepNeuralNet(pl.LightningModule):
    r"""Deep Neural Network model with Pytorch Lightning

    This class implements a DNN composed of an encoder :math:`\phi`
    and a classifier :math:`h`. The encoder is a function of a raw
    data space :math:`\mathcal{X}` (e.g., image/pixel space), and
    maps inputs to a latent space :math:`\mathcal{Z}`, i.e., a sub
    space of an Euclidean space :math:`\mathbb{R}^{d}`.

    The classifier maps :math:`\mathcal{Z}` into a label space
    :math:`\mathcal{Y}`.

    Parameters
    ----------
    encoder : torch module
        Module implementing the encoder architecture.
    task : torch module
        Module implementing the classifier architecture.
    learning_rate_encoder : float, optional (default=1e-5)
        Learning rate for the encoder part of the network.
    learning_rate_task : float, optional (default=None)
        Learning rate for the classifier part of the network.
        If it is None, uses 10 * learning_rate_encoder.
    loss_fn : function, optional (default=None)
        Function taking as arguments the network predictions
        and the ground-truths, and returns a differentiable
        loss. If None, uses torch.nn.CrossEntropyLoss().
    input_shape : tuple, optional (default=None)
        If None, assumes a problem of image classification
        where the inputs are RGB images w/ shape (3, 224, 224).
    l2_penalty : float, optional (default=0.0)
        If positive, adds l2_penalty to the network weights.
    weight_decay : float, optional (default=0.0)
        If positive, adds weight decay to network weights.
    momentum : float, optional (default=0.9)
        Only used if optimizer is SGD. Momentum term in the optimizer.
    log_gradients : bool, optional (default=False)
        If True, logs histograms of network gradients. NOTE: the
        generated tensorboard logs may be heavy if this parameter
        is set to true.
    optimizer_name : str, optional (default='adam')
        Either 'adam' or 'sgd'. Chooses which optimization strategy
        the network will adopt.
    """
    def __init__(self,
                 encoder,
                 task,
                 learning_rate_encoder=1e-5,
                 learning_rate_task=None,
                 loss_fn=None,
                 input_shape=None,
                 l2_penalty=0.0,
                 weight_decay=0.0,
                 momentum=0.9,
                 log_gradients=False,
                 multi_class=True,
                 optimizer_name='adam'):
        super(DeepNeuralNet, self).__init__()
        self.task = task
        self.encoder = encoder
        self.learning_rate = learning_rate_encoder

        if learning_rate_task is None:
            self.learning_rate_task = learning_rate_encoder * 10
        else:
            self.learning_rate_task = learning_rate_task

        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        if input_shape is None:
            self.example_input_array = torch.randn(16, 3, 224, 224)
        else:
            self.example_input_array = torch.randn(16, *input_shape)

        self.optimizer_name = optimizer_name.lower()
        self.l2_penalty = l2_penalty
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.log_gradients = log_gradients
        self.multi_class = multi_class

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.task(self.encoder(x))

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name,
                                                 params,
                                                 self.current_epoch)

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam([
                {
                    'params': self.encoder.parameters(),
                    'lr': self.learning_rate_encoder,
                    'weight_decay': self.weight_decay
                },
                {
                    'params': self.task.parameters(),
                    'lr': self.learning_rate_task,
                    'l2_penalty': self.l2_penalty,
                    'weight_decay': self.weight_decay
                }
            ])
        else:
            return torch.optim.SGD([
                {
                    'params': self.encoder.parameters(),
                    'lr': self.learning_rate_encoder,
                    'weight_decay': self.weight_decay,
                    'momentum': self.momentum
                },
                {
                    'params': self.task.parameters(),
                    'lr': self.learning_rate_task,
                    'l2_penalty': self.l2_penalty,
                    'weight_decay': self.weight_decay,
                    'momentum': self.momentum
                }
            ])

    def __step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        if self.multi_class:
            L = self.loss_fn(target=y.argmax(dim=1), input=y_pred)
            acc = accuracy(preds=y_pred,
                           target=y.argmax(dim=1),
                           task='multiclass',
                           num_classes=int(y.shape[1]), top_k=1)
        else:
            L = self.loss_fn(torch.sigmoid(y_pred.view(-1)), y.view(-1))
            acc = binary_accuracy(preds=torch.sigmoid(y_pred.view(-1)),
                                  target=y.view(-1))

        return {'loss': L, 'acc': acc}

    def training_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.training_step_outputs.append(output)

        return output

    def validation_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.validation_step_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.test_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(
            [x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.tensor(
            [x['acc'] for x in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()

        # Logs scalars
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

        if self.log_gradients:
            # Logs histograms
            self.custom_histogram_adder()

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Validation",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Validation",
                                              avg_acc,
                                              self.current_epoch)

    def on_test_epoch_end(self):
        avg_loss = torch.tensor(
            [x['loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Test",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Test",
                                              avg_acc,
                                              self.current_epoch)


class ShallowNeuralNet(pl.LightningModule):
    r"""Shallow Neural Network model with Pytorch Lightning

    This class implements a single layer neural network which
    predicts a classes from feature vectors.

    Parameters
    ----------
    n_features : int
        Number of features in the input of the network.
    n_classes : int
        Number of classes in the output of the network.
    learning_rate : float, optional (default=None)
        Learning rate for the classifier part of the network.
        If it is None, uses 10 * learning_rate_encoder.
    loss_fn : function, optional (default=None)
        Function taking as arguments the network predictions
        and the ground-truths, and returns a differentiable
        loss. If None, uses torch.nn.CrossEntropyLoss().
    l2_penalty : float, optional (default=0.0)
        If positive, adds l2_penalty to the network weights.
    weight_decay : float, optional (default=0.0)
        If positive, adds weight decay to network weights.
    momentum : float, optional (default=0.9)
        Only used if optimizer is SGD. Momentum term in the optimizer.
    log_gradients : bool, optional (default=False)
        If True, logs histograms of network gradients. NOTE: the
        generated tensorboard logs may be heavy if this parameter
        is set to true.
    optimizer_name : str, optional (default='adam')
        Either 'adam' or 'sgd'. Chooses which optimization strategy
        the network will adopt.
    max_norm : float, optional (default=None)
        If given, constrains the network weights to have maximum norm
        equal to the given value.
    """
    def __init__(self,
                 n_features,
                 n_classes,
                 learning_rate=1e-4,
                 loss_fn=None,
                 l2_penalty=0.0,
                 momentum=0.9,
                 optimizer_name='adam',
                 log_gradients=False,
                 max_norm=None):
        super(ShallowNeuralNet, self).__init__()
        self.main = torch.nn.Linear(n_features, n_classes)
        self.learning_rate = learning_rate

        if loss_fn is None:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        self.l2_penalty = l2_penalty
        self.momentum = momentum
        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        self.log_gradients = log_gradients
        self.optimizer_name = optimizer_name
        self.n_classes = n_classes
        self.max_norm = max_norm

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def max_norm_normalization(self, w):
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(w ** 2, dim=1, keepdim=True))
            desired = torch.clamp(norm, 0, self.max_norm)
            w *= (desired / (1e-10 + norm))

    def custom_histogram_adder(self):
        if self.logger is not None:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name,
                                                     params,
                                                     self.current_epoch)

    def forward(self, x):
        if self.max_norm is not None:
            self.max_norm_normalization(self.main.weight)
        return self.main(x)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.l2_penalty)
        else:
            return torch.optim.SGD(self.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=self.l2_penalty,
                                   momentum=self.momentum)

    def __step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)

        L = self.loss_fn(target=y.argmax(dim=1), input=y_pred)
        acc = accuracy(preds=y_pred,
                       target=y.argmax(dim=1),
                       task='multiclass',
                       num_classes=self.n_classes,
                       top_k=1)

        return {'loss': L, 'acc': acc}

    def training_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.validation_step_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx):
        output = self.__step(batch, batch_idx)
        self.test_step_outputs.append(output)

        return output

    def on_train_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()

        self.log('loss', avg_loss)
        self.log('accuracy', avg_acc)

        self.history['loss'].append(avg_loss)
        self.history['acc'].append(avg_acc)

        # Logs scalars
        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Train",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Train",
                                              avg_acc,
                                              self.current_epoch)

        # Logs histograms
        if self.log_gradients:
            self.custom_histogram_adder()

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()

        self.log('val_loss', avg_loss)
        self.log('val_accuracy', avg_acc)

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Validation",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Validation",
                                              avg_acc,
                                              self.current_epoch)

    def on_test_epoch_end(self):
        avg_loss = torch.tensor([
            x['loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.tensor([
            x['acc'] for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()

        if self.logger is not None:
            self.logger.experiment.add_scalar("Loss/Test",
                                              avg_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar("Accuracy/Test",
                                              avg_acc,
                                              self.current_epoch)
