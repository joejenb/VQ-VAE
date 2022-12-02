from Residual import ResidualStack

from VectorQuantiser import VectorQuantiserEMA

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.gelu(x)
        
        x = self._conv_2(x)
        x = F.gelu(x)
        
        x = self._conv_3(x)
        x = F.gelu(x)

        x = self._conv_4(x)
        #Should have 2048 units -> embedding_dim * repres_dim^2
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.gelu(x)

        x = self._conv_trans_2(x)
        x = F.gelu(x)
        
        return self._conv_trans_3(x)

class VQVAE(nn.Module):
    def __init__(self, config, device):
        super(VQVAE, self).__init__()

        self.device = device

        self._embedding_dim = config.embedding_dim
        self._representation_dim = config.representation_dim

        self._encoder = Encoder(config.num_channels, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=config.num_hiddens, 
                                      out_channels=config.num_filters,
                                      kernel_size=1, 
                                      stride=1)

        self._vq_vae = VectorQuantiserEMA(config.num_embeddings, config.embedding_dim, 
                                            config.commitment_cost, config.decay)

        self._decoder = Decoder(config.num_filters,
                            config.num_channels,
                            config.num_hiddens, 
                            config.num_residual_layers, 
                            config.num_residual_hiddens
                        )

    def sample(self):
        z = torch.randn(1, self._embedding_dim, self._representation_dim, self._representation_dim)
        _, z_quantised, _, _ = self._vq_vae(z)

        x_sample = self._decoder(z_quantised)

        return x_sample

    def interpolate(self, x, y):
        if (x.size() == y.size()):
            zx = self._encoder(x)
            zx = self._pre_vq_conv(zx)

            zy = self._encoder(y)
            zy = self._pre_vq_conv(zy)

            z = (zx + zy) / 2

            _, z_quantised, _, _ = self._vq_vae(z)

            xy_inter = self._decoder(z_quantised)

            return xy_inter
        return x

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)

        quant_loss, z_quantised, _, _ = self._vq_vae(z)
        x_recon = self._decoder(z_quantised)

        return x_recon, quant_loss 