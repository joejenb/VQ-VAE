from Residual import ResidualStack

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
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._conv_5 = nn.Conv2d(in_channels=num_hiddens,
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
        x = F.relu(x)

        x = self._conv_5(x)
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
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
       
        self._conv_trans_4 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
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
        
        x = self._conv_trans_3(x)
        x = F.relu(x)

        return self._conv_trans_4(x)


class VAE(nn.Module):
    def __init__(self, config, device):
        super(VAE, self).__init__()

        self.device = device

        self._embedding_dim = config.embedding_dim

        self._encoder = Encoder(config.num_channels, config.num_hiddens,
                                config.num_residual_layers, 
                                config.num_residual_hiddens)

        self._pre_sample = nn.Conv2d(in_channels=config.num_hiddens, 
                                      out_channels=config.num_filters,
                                      kernel_size=1, 
                                      stride=1)

        self.mu = nn.Linear(config.embedding_dim * 64, config.embedding_dim * 64)
        self.log_var = nn.Linear(config.embedding_dim * 64, config.embedding_dim * 64)

        self.pre_decode = nn.Linear(config.embedding_dim * 64, config.embedding_dim * 64)

        self._decoder = Decoder(config.num_filters,
                            config.num_channels,
                            config.num_hiddens, 
                            config.num_residual_layers, 
                            config.num_residual_hiddens
                        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


    def interpolate(self, x, y):
        if (x.size() == y.size()):
            zx = self._encoder(x)
            zx = self._pre_sample(zx)

            zy = self._encoder(y)
            zy = self._pre_sample(zy)

            z = (zx + zy) / 2

            z_shape = z.shape
            flat_z = z.view(z_shape[0], z_shape[1], self._embedding_dim)

            flat_z_quantised = self._hopfield(flat_z)#flat_z)

            z_quantised = flat_z_quantised.view(z_shape)

            xy_recon = self._decoder(z_quantised)

            return xy_recon
        return x

    def forward(self, x):
        z = self._encoder(x)
        z = F.relu(self._pre_sample(z))

        z_shape = z.shape

        flat_z = z.view(z_shape[0], -1)

        mu = self.mu(flat_z)
        log_var = self.log_var(flat_z)
        z_sampled = self.reparameterize(mu, log_var)

        flat_z_sampled = self.pre_decode(z_sampled)

        z_sampled = flat_z_sampled.view(z_shape)

        x_recon = self._decoder(z_sampled)

        return x_recon, mu, log_var