import numpy as np
import torch
from .resnet import ResBlock, ResBlockDeconv
from vae_lidar import VAE_DEFAULT_FILENAME

class Encoder(torch.nn.Module):
    """Image encoder into a latent space.
    The output shape is 2*size_latent since the VAE makes use of mean+std.
    """
    def __init__(self, nb_chan, size_latent, dropout_rate=0.2, filename=None, device=None):
        filename = filename if filename is not None else VAE_DEFAULT_FILENAME
        filename += '_encoder'
        super(Encoder, self).__init__()
        self.nb_chan = nb_chan
        self.size_latent = size_latent
        self.dropout_rate = dropout_rate

        self.layers = torch.nn.ModuleDict({
            'resnet': torch.nn.Sequential(
                torch.nn.Conv2d(self.nb_chan, 64, kernel_size=7, stride=2, padding=3),
                torch.nn.ELU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResBlock(64, 2, batchnorm=True, dropout_rate=self.dropout_rate),
                ResBlock(128, 2, batchnorm=True, dropout_rate=self.dropout_rate),
                ResBlock(256, 2, batchnorm=True, dropout_rate=self.dropout_rate),
                ResBlock(512, 1, batchnorm=True, dropout_rate=self.dropout_rate),
                torch.nn.AdaptiveAvgPool2d((2, 2)),
                torch.nn.Flatten(),
            ),
            'mean': torch.nn.Linear(512 * 2 * 2, size_latent),
            'logvar': torch.nn.Linear(512 * 2 * 2, size_latent),
        })

        self.eval()
        self.zero_grad()


    def forward(self, input):
        features = self.layers['resnet'](input)
        mean = self.layers['mean'](features)
        logvar = self.layers['logvar'](features)
        return mean, logvar


    def encode(self, input):
        """Convenience function that returns only the mean of the latent distribution."""
        features = self.layers['resnet'](input)
        return self.layers['mean'](features)



class Decoder(torch.nn.Module):
    """Image decoder from a latent space."""
    def __init__(self, nb_chan, size_latent, shape_imgs, dropout_rate=0.2, filename=None, device=None):
        filename = filename if filename is not None else VAE_DEFAULT_FILENAME
        filename += '_decoder'
        super(Decoder, self).__init__()
        self.nb_chan = nb_chan
        self.size_latent = size_latent
        self.shape_imgs = shape_imgs
        self.dropout_rate = dropout_rate

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(size_latent, 512 * 8 * 15),
            torch.nn.Unflatten(1, (512, 8, 15)),
            ResBlockDeconv(512, 2, output_padding=1, batchnorm=True, dropout_rate=self.dropout_rate),
            ResBlockDeconv(256, 2, output_padding=1, batchnorm=True, dropout_rate=self.dropout_rate),
            ResBlockDeconv(128, 2, output_padding=1, batchnorm=True, dropout_rate=self.dropout_rate),
            ResBlockDeconv(64, 2, output_padding=1, batchnorm=True, dropout_rate=self.dropout_rate),
            torch.nn.ConvTranspose2d(32, self.nb_chan, kernel_size=5, stride=1, padding=2),
            torch.nn.Upsample(size=shape_imgs, mode='bilinear'),
            torch.nn.Sigmoid(),
        )

        self.eval()
        self.zero_grad()

    def forward(self, input):
        return self.layers(input)



class Vae(torch.nn.Module):
    """Variational AutoEncoder."""
    def __init__(self, 
                 size_latent, 
                 shape_imgs, 
                 filename, 
                 dropout_rate=0.2, 
                 device='cpu'):
        super(Vae, self).__init__()
        self.nb_chan = 1  # range images
        self.size_latent = size_latent
        self.shape_imgs = shape_imgs
        self.encoder = Encoder(self.nb_chan, size_latent, dropout_rate, filename, device)
        self.decoder = Decoder(self.nb_chan, size_latent, shape_imgs, dropout_rate,filename, device)
        self.device = device
        
        if not filename.endswith('.pth'): filename += '.pth'
        weights = torch.load(filename, map_location=self.device, weights_only=False)
        self.load_state_dict(weights)
        self.to(self.device)
        self.eval()

    def forward(self, input):
        ## encode
        mean, logvar = self.encoder(input)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sampled = eps * std + mean
        else:
            z_sampled = mean

        ## decode

        return mean, logvar
    def decode(self, z_sampled):
        output = self.decoder(z_sampled)
        return output
