import numpy as np
import torch


class ResBlock(torch.nn.Module):
    def __init__(self, size_in, stride, bottleneck=False, batchnorm=False, dropout_rate=0):
        """Residual block implementation.
        size_in         -- number of features of the input volume
        stride          -- stride for spatial dimension reduction
        bottleneck      -- use a bottleneck block (1x1, 3x3, 1x1 convolutions) or standard residual block (3x3, 3x3 convolutions)
        batchnorm       -- use batchnorm or not
        dropout_rate    -- dropout after activations
        """
        super(ResBlock, self).__init__()
        bottleneck_dim_reduction = 4  # hardcoded since I've never seen anyone use some other number
        size_inner = size_in // bottleneck_dim_reduction
        size_out = size_in * stride
        self.dropout_rate = dropout_rate

        self.norm = torch.nn.BatchNorm2d if batchnorm else torch.nn.Identity
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout_rate)


        if bottleneck:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(size_in, size_inner, kernel_size=1, padding=0, stride=stride, bias=not batchnorm),
                self.norm(size_inner),
                self.activation,
                self.dropout,
                torch.nn.Conv2d(size_inner, size_inner, kernel_size=3, padding=1, stride=1, bias=not batchnorm),
                self.norm(size_inner),
                self.activation,
                self.dropout,
                torch.nn.Conv2d(size_inner, size_out, kernel_size=1, padding=0, stride=1, bias=not batchnorm),
                self.norm(size_out),
            )
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(size_in, size_out, kernel_size=3, padding=1, stride=stride, bias=not batchnorm),
                self.norm(size_out),
                self.activation,
                self.dropout,
                torch.nn.Conv2d(size_out, size_out, kernel_size=3, padding=1, stride=1, bias=not batchnorm),
                self.norm(size_out),
            )

        if stride == 1:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Sequential(
                 torch.nn.Conv2d(size_in, size_out, kernel_size=1, padding=0, stride=stride, bias=not batchnorm),
                 self.norm(size_out)
            )


    def forward(self, input):
        output = self.layers(input)
        output += self.shortcut(input)
        output = self.activation(output)
        output = self.dropout(output)
        return output



class ResBlockDeconv(torch.nn.Module):
    def __init__(self, size_in, stride, output_padding=0, bottleneck=False, batchnorm=False, dropout_rate=0):
        super(ResBlockDeconv, self).__init__()
        bottleneck_dim_reduction = 4
        size_inner = size_in // bottleneck_dim_reduction
        size_out = size_in // stride
        self.dropout_rate = dropout_rate

        self.norm = torch.nn.BatchNorm2d if batchnorm else torch.nn.Identity
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout_rate)

        if bottleneck:
            self.layers = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(size_in, size_inner, kernel_size=1, padding=0, stride=stride, output_padding=output_padding, bias=not batchnorm),
                self.norm(size_inner),
                self.activation,
                self.dropout,
                torch.nn.ConvTranspose2d(size_inner, size_inner, kernel_size=3, padding=1, stride=1, output_padding=0, bias=not batchnorm),
                self.norm(size_inner),
                self.activation,
                self.dropout,
                torch.nn.ConvTranspose2d(size_inner, size_out, kernel_size=1, padding=0, stride=1, output_padding=0, bias=not batchnorm),
                self.norm(size_out),
            )
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(size_in, size_out, kernel_size=3, padding=1, stride=stride, output_padding=output_padding, bias=not batchnorm),
                self.norm(size_out),
                self.activation,
                self.dropout,
                torch.nn.ConvTranspose2d(size_out, size_out, kernel_size=3, padding=1, stride=1, output_padding=0, bias=not batchnorm),
                self.norm(size_out),
            )

        if stride == 1:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(size_in, size_out, kernel_size=1, padding=0, stride=stride, output_padding=output_padding, bias=not batchnorm),
                self.norm(size_out)
            )


    def forward(self, input):
        output = self.layers(input)
        output += self.shortcut(input)
        output = self.activation(output)
        output = self.dropout(output)
        return output
