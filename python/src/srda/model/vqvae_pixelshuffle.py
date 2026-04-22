import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim: int, h_dim: int, res_h_dim: int, n_res_layers: int):
        super().__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_res_layers: int,
        res_h_dim: int,
        compression_factor: int,
    ):
        super().__init__()
        kernel = 4
        stride = 2
        num_comp = int(math.log2(compression_factor))
        h_dims = [(h_dim * (i + 1) + i) // num_comp for i in range(num_comp)]
        h_dims = [max(d, 1) for d in h_dims]

        blocks = []
        for i in range(num_comp):
            _in = in_dim if i == 0 else h_dims[i - 1]
            _out = h_dims[i]
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(_in, _out, kernel_size=kernel, stride=stride, padding=1),
                    nn.ReLU() if i != num_comp - 1 else nn.Identity(),
                )
            )
        blocks.append(ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e: int, e_dim: int, beta: float):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


def ICNR(
    tensor: torch.Tensor, scale_factor: int = 2, initializer=nn.init.kaiming_uniform_
):
    OUT, IN, H, W = tensor.shape
    sub = torch.zeros(OUT // scale_factor**2, IN, H, W)
    sub = initializer(sub)

    kernel = torch.zeros_like(tensor)
    for i in range(OUT):
        kernel[i] = sub[i // scale_factor**2]

    return kernel


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_res_layers: int,
        res_h_dim: int,
        compression_factor: int,
    ):
        super().__init__()
        kernel = 4
        stride = 2

        num_comp = int(math.log2(compression_factor))
        h_dims = [(h_dim * (i + 1) + i) // num_comp for i in range(num_comp)]
        h_dims = [max(d, 1) for d in h_dims]

        blocks = []
        blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
                ),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            )
        )

        for i in reversed(range(num_comp)):
            _in = h_dim if i == num_comp - 1 else h_dims[i + 1]
            _out = out_dim if i == 0 else h_dims[i]

            conv = nn.Conv2d(
                _in,
                _out * 4,
                kernel_size=kernel - 1,
                stride=stride - 1,
                padding=1,
            )
            kernel_icnr = ICNR(conv.weight)
            conv.weight.data.copy_(kernel_icnr)

            blocks.append(
                nn.Sequential(
                    conv,
                    nn.PixelShuffle(2),
                    nn.ReLU() if i != 0 else nn.Identity(),
                )
            )

        blocks.append(nn.Tanh())
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class VQVAE(nn.Module):
    """
    Inputs:
    - in_dim : the input dimension
    - out_dim : the out dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    - n_embeddings : the embedding vector numbers
    - embedding_dim : the embedding dimension. This is a channel (or channels) of latent variable
    - beta : parameter of loss function
    - compression_factor : If you set compression_factor = 4, an image whose size is 128*64 bacomes an image whose size is 32*16 in latent space.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        res_hidden_channels: int,
        num_res_layers: int,
        num_embeddings: int,
        embedding_dim: int,
        beta: float,
        compression_factor: int,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            hidden_channels,
            num_res_layers,
            res_hidden_channels,
            compression_factor,
        )
        self.pre_quantization_conv = nn.Conv2d(
            hidden_channels, embedding_dim, kernel_size=1, stride=1
        )
        self.vector_quantization = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.decoder = Decoder(
            embedding_dim,
            hidden_channels,
            out_channels,
            num_res_layers,
            res_hidden_channels,
            compression_factor,
        )

    def make_latent_variables(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        _, z_q, _, _, _ = self.vector_quantization(z_e)
        return z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        _, z_q, _, _, _ = self.vector_quantization(z)
        return self.decoder(z_q)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, embedding_loss, perplexity
