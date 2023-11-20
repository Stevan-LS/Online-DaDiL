r"""Neural Network architectures module."""

import torch
import torchvision


class CRWUEncoder(torch.nn.Module):
    r"""CRWU Encoder. Consists of a MLP
    that maps a feature vector of 2048
    dimensions into a feature vector of
    256 dimensions.

    Parameters
    ----------
    n_features : int, optional (default=2048)
        Number of features in the input of the network.
    """
    def __init__(self, n_features=2048):
        super(CRWUEncoder, self).__init__()
        self.n_features = n_features

        self.main = torch.nn.Sequential(
            torch.nn.Linear(n_features, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.main(x)


class ConvolutionalEncoder(torch.nn.Module):
    r"""Tennessee Eastman Convolutional Encoder.

    Consists of a CNN for extracting features
    from TEP raw signals.

    Parameters
    ----------
    n_features : int, optional (default=51)
        Number of features in the raw data. Corresponds
        to the number of sensors in the time series.
    n_time_steps : int, optional (default=600)
        Number of time steps in each time series.
    n_conv_blocks : int, optional (default=6)
        Number of convolutional blocks in the CNN.
    n_filters : int, optional (default=128)
        Number of filters in each convolutional layer.
    kernel_size : int, optional (default=7)
        Size of 1D kernels for convolutions. Each unit
        corresponds to a minute in the time series.
    multiply_every : int, optional (default=3)
        Multiplies the number of features every `multiply_every`
        layers.
    first_pool : bool, optional (default=False)
        If True, applies pooling after 1st layer.
    layer_norm : bool, optional (defualt=False)
        If True, applies layer normalization on conv blocks.
    batch_norm : bool, optional (defualt=False)
        If True, applies batch normalization on conv blocks.
    instance_norm : bool, optional (defualt=False)
        If True, applies instance normalization on conv blocks.
    """
    def __init__(self,
                 n_features=51,
                 n_time_steps=600,
                 n_conv_blocks=6,
                 n_filters=128,
                 kernel_size=7,
                 multiply_every=3,
                 first_pool=False,
                 layer_norm=False,
                 batch_norm=False,
                 instance_norm=False):
        super(ConvolutionalEncoder, self).__init__()

        x = torch.randn(16, n_features, n_time_steps)
        current_seq_len = n_time_steps

        if batch_norm:
            layer_list = [
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=n_filters,
                                kernel_size=kernel_size,
                                padding='same'),
                torch.nn.BatchNorm1d(num_features=n_filters, affine=False),
                torch.nn.ReLU(inplace=True),
            ]
        elif layer_norm:
            layer_list = [
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=n_filters,
                                kernel_size=kernel_size,
                                padding='same'),
                torch.nn.LayerNorm([n_filters, current_seq_len]),
                torch.nn.ReLU(inplace=True),
            ]
        elif instance_norm:
            layer_list = [
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=n_filters,
                                kernel_size=kernel_size,
                                padding='same'),
                torch.nn.InstanceNorm1d(num_features=n_filters, affine=False),
                torch.nn.ReLU(inplace=True),
            ]
        else:
            layer_list = [
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=n_filters,
                                kernel_size=kernel_size,
                                padding='same'),
                torch.nn.ReLU(inplace=True),
            ]
        if first_pool:
            layer_list.append(torch.nn.MaxPool1d(kernel_size=kernel_size,
                                                 stride=2,
                                                 padding=kernel_size // 2))

        last_nf, nf = n_filters, n_filters

        # Next conv layers
        for i in range(1, n_conv_blocks):
            if i % multiply_every == 0:
                last_nf, nf = nf, 2 * nf
            if batch_norm:
                layer_list.extend([
                    torch.nn.Conv1d(in_channels=last_nf,
                                    out_channels=nf,
                                    kernel_size=kernel_size,
                                    padding='same'),
                    torch.nn.BatchNorm1d(num_features=nf, affine=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool1d(kernel_size=kernel_size, stride=2,
                                       padding=kernel_size // 2)
                ])
            elif layer_norm:
                layer_list.extend([
                    torch.nn.Conv1d(in_channels=last_nf,
                                    out_channels=nf,
                                    kernel_size=kernel_size,
                                    padding='same'),
                    torch.nn.LayerNorm([nf, current_seq_len]),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool1d(kernel_size=kernel_size, stride=2,
                                       padding=kernel_size // 2)
                ])
            elif instance_norm:
                layer_list.extend([
                    torch.nn.Conv1d(in_channels=last_nf,
                                    out_channels=nf,
                                    kernel_size=kernel_size,
                                    padding='same'),
                    torch.nn.InstanceNorm1d(num_features=nf, affine=False),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool1d(kernel_size=kernel_size, stride=2,
                                       padding=kernel_size // 2)
                ])
            else:
                layer_list.extend([
                    torch.nn.Conv1d(in_channels=last_nf,
                                    out_channels=nf,
                                    kernel_size=kernel_size,
                                    padding='same'),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool1d(kernel_size=kernel_size, stride=2,
                                       padding=kernel_size // 2)
                ])
            last_nf = nf
            if current_seq_len % 2 == 0:
                current_seq_len = current_seq_len // 2
            else:
                current_seq_len = current_seq_len // 2 + 1
            self.main = torch.nn.Sequential(*layer_list)

        with torch.no_grad():
            h = self(x)
        self.n_out_feats = h.shape[-1]

    def forward(self, x):
        return torch.flatten(self.main(x), start_dim=1)


class TennesseeEastmannFullyConvolutionalEncoder(torch.nn.Module):
    r"""Tennessee Eastman Fully Convolutional Encoder.

    Consists of a FCN for extracting features
    from TEP raw signals.

    Parameters
    ----------
    n_features : int, optional (default=51)
        Number of features in the raw data. Corresponds
        to the number of sensors in the time series.
    n_time_steps : int, optional (default=600)
        Number of time steps in each time series.
    batch_norm : bool, optional (defualt=False)
        If True, applies batch normalization on conv blocks.
    instance_norm : bool, optional (defualt=True)
        If True, applies instance normalization on conv blocks.
    """
    def __init__(self,
                 n_features=51,
                 n_time_steps=600,
                 batch_norm=False,
                 instance_norm=True):
        super(TennesseeEastmannFullyConvolutionalEncoder, self).__init__()

        x = torch.randn(16, n_features, n_time_steps)

        if batch_norm:
            self.main = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=128,
                                kernel_size=9,
                                padding='same'),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=5,
                                padding='same'),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=256,
                                out_channels=128,
                                kernel_size=3,
                                padding='same'),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(inplace=True),
            )
        elif instance_norm:
            self.main = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=128,
                                kernel_size=9,
                                padding='same'),
                torch.nn.InstanceNorm1d([128, n_time_steps]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=5,
                                padding='same'),
                torch.nn.InstanceNorm1d([256, n_time_steps]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=256,
                                out_channels=128,
                                kernel_size=3,
                                padding='same'),
                torch.nn.InstanceNorm1d([128, n_time_steps]),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.main = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=128,
                                kernel_size=9,
                                padding='same'),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=5,
                                padding='same'),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=256,
                                out_channels=128,
                                kernel_size=3,
                                padding='same'),
                torch.nn.ReLU(inplace=True),
            )
        with torch.no_grad():
            h = self.main(x)
            h = h.mean(dim=-1)
        self.n_out_feats = h.shape[-1]

    def forward(self, x):
        return self.main(x).mean(dim=-1)


class CelebaEncoder(torch.nn.Module):
    r"""Celeba Encoder

    Parameters
    ----------
    init_num_filters : int
        Initial number of filters from encoder image channels
    lrelu_slope : float
        Positive number indicating LeakyReLU negative slope
    inter_fc_dim : int
        Intermediate fully connected dimensionality prior to embedding layer
    embedding_dim : int
        Embedding dimensionality
    """
    def __init__(self,
                 init_num_filters=16,
                 lrelu_slope=0.2,
                 inter_fc_dim=128,
                 embedding_dim=2,
                 nc=3,
                 dropout=0.05):
        super(CelebaEncoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(nc, self.init_num_filters_ * 1, 4, 2, 1,
                            bias=False),
            torch.nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            torch.nn.Dropout(dropout),

            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(self.init_num_filters_,
                            self.init_num_filters_ * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.init_num_filters_ * 2),
            torch.nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            torch.nn.Dropout(dropout),

            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(self.init_num_filters_ * 2,
                            self.init_num_filters_ * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.init_num_filters_ * 4),
            torch.nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            torch.nn.Dropout(dropout),

            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(self.init_num_filters_ * 4,
                            self.init_num_filters_ * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.init_num_filters_ * 8),
            torch.nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            torch.nn.Dropout(dropout),

            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(self.init_num_filters_ * 8,
                            self.init_num_filters_ * 8, 4, 2, 0, bias=False),
        )

        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(self.init_num_filters_ * 8, self.embedding_dim_),
            torch.nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            torch.nn.BatchNorm1d(self.embedding_dim_, affine=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.fc_out(x)
        return x


class CelebaDecoder(torch.nn.Module):
    r"""Celeba Decoder

    Parameters
    ----------
    init_num_filters : int, optional (default=16)
        Initial number of filters from encoder image channels
    lrelu_slope : float, optional (default=0.2)
        Positive number indicating LeakyReLU negative slope
    inter_fc_dim : int, optional (default=128)
        Intermediate fully connected dimensionality prior to embedding layer
    embedding_dim : int, optional (default=2)
        embedding dimensionality
    """
    def __init__(self,
                 init_num_filters=16,
                 lrelu_slope=0.2,
                 inter_fc_dim=128,
                 embedding_dim=2,
                 nc=3,
                 dropout=0.05):
        super(CelebaDecoder, self).__init__()

        self.init_num_filters_ = init_num_filters
        self.lrelu_slope_ = lrelu_slope
        self.inter_fc_dim_ = inter_fc_dim
        self.embedding_dim_ = embedding_dim

        self.features = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.init_num_filters_ * 8,
                                     self.init_num_filters_ * 8, 4, 1, 0,
                                     bias=False),
            torch.nn.BatchNorm2d(self.init_num_filters_ * 8),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),

            torch.nn.ConvTranspose2d(self.init_num_filters_ * 8,
                                     self.init_num_filters_ * 4, 4, 2, 1,
                                     bias=False),
            torch.nn.BatchNorm2d(self.init_num_filters_ * 4),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),

            torch.nn.ConvTranspose2d(self.init_num_filters_ * 4,
                                     self.init_num_filters_ * 2, 4, 2, 1,
                                     bias=False),
            torch.nn.BatchNorm2d(self.init_num_filters_ * 2),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),

            torch.nn.ConvTranspose2d(self.init_num_filters_ * 2,
                                     self.init_num_filters_ * 1, 4, 2, 1,
                                     bias=False),
            torch.nn.BatchNorm2d(self.init_num_filters_ * 1),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),

            torch.nn.ConvTranspose2d(self.init_num_filters_ * 1, nc, 4, 2, 1,
                                     bias=False),
            torch.nn.Tanh()
        )

        self.fc_in = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim_, self.init_num_filters_ * 8),
            torch.nn.LeakyReLU(self.lrelu_slope_, inplace=True),
            torch.nn.BatchNorm1d(self.init_num_filters_ * 8)
        )

    def forward(self, z):
        z = self.fc_in(z)
        z = z.view(-1, self.init_num_filters_ * 8, 1, 1)
        z = self.features(z)
        return z


class DigitsMLPEncoder(torch.nn.Module):
    r"""Multi-Layer Perceptron Encoder for the
    MNIST dataset.

    Parameters
    ----------
    n_dim : int, optional (default=784)
        Number of dimensions in the raw data vector.
    n_encoder_features : int, optional (default=512)
        Number of dimensions in the encoder latent vector.
    n_latent_space : int, optinal (default=2)
        Number of dimensions of the latent space.
    """
    def __init__(self,
                 n_dim=784,
                 n_encoder_features=512,
                 n_latent_space=2):
        super(DigitsMLPEncoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(n_dim, n_encoder_features),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_encoder_features, n_encoder_features),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_encoder_features, n_encoder_features),
            torch.nn.ReLU(True)
        )

        self.mu = torch.nn.Linear(n_encoder_features, n_latent_space)
        self.std = torch.nn.Linear(n_encoder_features, n_latent_space)

    def forward(self, x):
        z = self.main(x)
        mu, std = self.mu(z), self.std(z)

        return z, mu, std


class DigitsMLPDecoder(torch.nn.Module):
    r"""Multi-Layer Perceptron Decoder for the
    MNIST dataset.

    Parameters
    ----------
    n_dim : int, optional (default=784)
        Number of dimensions in the raw data vector.
    n_encoder_features : int, optional (default=512)
        Number of dimensions in the encoder latent vector.
    n_latent_space : int, optinal (default=2)
        Number of dimensions of the latent space.
    """
    def __init__(self,
                 n_features=784,
                 n_decoder_features=512,
                 n_latent_space=2):
        super(DigitsMLPDecoder, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(n_latent_space, n_decoder_features),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_decoder_features, n_decoder_features),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_decoder_features, n_features),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class DigitsConvolutionalEncoder(torch.nn.Module):
    r"""CNN Encoder for the MNIST dataset.

    Parameters
    ----------
    latent_dim : int, optinal (default=128)
        Number of dimensions of the latent space.
    variational : bool, optional (default=True)
        If True, architecture consists of a variational autoencoder,
        so that the output of this module is a 3-tuple, with the latent
        code z, and the mean and variance of the VAE.
    """
    def __init__(self, latent_dim=128, variational=True):
        super(DigitsConvolutionalEncoder, self).__init__()

        self.variational = variational

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=2, padding=1),
        )

        if variational:
            self.mu = torch.nn.Linear(in_features=512,
                                      out_features=latent_dim)
            self.std = torch.nn.Linear(in_features=512,
                                       out_features=latent_dim)
        else:
            self.proj = torch.nn.Linear(in_features=512,
                                        out_features=latent_dim)

    def forward(self, x):
        z = self.main(x).view(-1, 512)
        if self.variational:
            return z, self.mu(z), self.std(z)
        else:
            return self.proj(z)


class DigitsConvolutionalDecoder(torch.nn.Module):
    r"""CNN Decoder for the MNIST dataset.

    Parameters
    ----------
    latent_dim : int, optinal (default=128)
        Number of dimensions of the latent space.
    variational : bool, optional (default=True)
        If True, applies a sigmoid to the network output.
    """
    def __init__(self, latent_dim=128, variational=True):
        super(DigitsConvolutionalDecoder, self).__init__()
        self.variational = variational

        self.proj = torch.nn.Linear(in_features=latent_dim, out_features=512)

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                            padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3,
                            padding='same'),
        )

    def forward(self, x):
        z = self.proj(x).view(-1, 32, 4, 4)
        if self.variational:
            return self.main(z).sigmoid()
        return self.main(z)


def get_resnet(resnet_size=50):
    r"""Auxiliary function for constructing a resnet, given its size.

    Parameters
    ----------
    resnet_size : int, optional (default=50)
        Number of layers in the resnet. Either 18, 34, 50, 101 or 152.
    """
    if resnet_size == 18:
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1',
                                            progress=True)
        T = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    elif resnet_size == 34:
        model = torchvision.models.resnet34(weights='IMAGENET1K_V1',
                                            progress=True)
        T = torchvision.models.ResNet34_Weights.IMAGENET1K_V1.transforms()
    elif resnet_size == 50:
        model = torchvision.models.resnet50(weights='IMAGENET1K_V2',
                                            progress=True)
        T = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    elif resnet_size == 101:
        model = torchvision.models.resnet101(weights='IMAGENET1K_V2',
                                             progress=True)
        T = torchvision.models.ResNet101_Weights.IMAGENET1K_V2.transforms()
    elif resnet_size == 152:
        model = torchvision.models.resnet152(weights='IMAGENET1K_V2',
                                             progress=True)
        T = torchvision.models.ResNet152_Weights.IMAGENET1K_V2.transforms()
    else:
        raise ValueError(("Expected resnet_size to be in [18, 34, 50, 101"
                          f", 152], but got {resnet_size}"))
    model.fc = torch.nn.Identity()

    return model, T
