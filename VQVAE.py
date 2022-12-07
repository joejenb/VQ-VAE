from Residual import ResidualStack

from VectorQuantiser import VectorQuantiserEMA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from PixelCNN.PixelCNN import PixelCNN

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
                                 stride=1, padding=2)

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
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = F.relu(x)

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
                                                stride=1, padding=2)

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
        x = F.relu(x)

        x = self._conv_trans_2(x)
        x = F.relu(x)
        
        return self._conv_trans_3(x)

class VQVAE(nn.Module):
    def __init__(self, config, prior_config, device):
        super(VQVAE, self).__init__()

        self.device = device

        self._embedding_dim = config.embedding_dim
        self._num_embeddings = config.num_embeddings
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
        
        self.fit_prior = False
        self.prior = PixelCNN(prior_config, device)

        self._decoder = Decoder(config.num_filters,
                            config.num_channels,
                            config.num_hiddens, 
                            config.num_residual_layers, 
                            config.num_residual_hiddens
                        )

    def sample(self):
        z_sample_indices = self.prior.sample().type(torch.int64)
        z_sample_indices = z_sample_indices.permute(0, 2, 3, 1).contiguous()
        z_sample_indices = z_sample_indices.view(-1, 1)

        z_sample = torch.zeros(z_sample_indices.shape[0], self._num_embeddings, device=self.device)
        z_sample.scatter_(1, z_sample_indices, 1)
        
        # Quantize and unflatten
        z_quantised = torch.matmul(z_sample, self._vq_vae._embedding.weight).view(1, self._representation_dim, self._representation_dim, self._embedding_dim)
        z_quantised = z_quantised.permute(0, 3, 1, 2).contiguous()

        x_sample = self._decoder(z_quantised)

        return x_sample

    def interpolate(self, x, y):
        if (x.size() == y.size()):
            zx = self._encoder(x)
            zx = self._pre_vq_conv(zx)

            zy = self._encoder(y)
            zy = self._pre_vq_conv(zy)

            z = (zx + zy) / 2

            _, z_quantised, z_indices = self._vq_vae(z)

            z_denoised_indices = self.prior.denoise(z_indices / self._num_embeddings).type(torch.int64)
            z_denoised_indices = z_denoised_indices.permute(0, 2, 3, 1).contiguous()
            z_denoised_indices = z_denoised_indices.view(-1, 1)

            z_sample = torch.zeros(z_denoised_indices.shape[0], self._num_embeddings, device=self.device)
            z_sample.scatter_(1, z_denoised_indices, 1)
            
            # Quantize and unflatten
            z_perm_shape = (z.shape[0], self._representation_dim, self._representation_dim, self._embedding_dim)
            z_quantised = torch.matmul(z_sample, self._vq_vae._embedding.weight).view(z_perm_shape)
            z_quantised = z_quantised.permute(0, 3, 1, 2).contiguous()

            xy_inter = self._decoder(z_quantised)

            return xy_inter
        return x

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)

        quant_loss, z_quantised, z_indices = self._vq_vae(z)
        z_indices = z_indices

        if self.fit_prior:
            z_logits = self.prior((z_indices.float() / self._num_embeddings) - 0.5)

            nll = F.cross_entropy(z_logits, z_indices.detach(), reduction='none')
            z_prediction_error = nll.mean() #* torch.log2(torch.exp(torch.Tensor(1))).to(nll.device)
            z_prediction_error = z_prediction_error.mean()
            print(z_prediction_error)

            x_recon = self._decoder(z_quantised)
            return x_recon, quant_loss.detach(), z_prediction_error

        x_recon = self._decoder(z_quantised)
        return x_recon, quant_loss, 0