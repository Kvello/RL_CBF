import torch
import os
from collision_predictor_mpc import COLPREDMPC_TMP_DIR


class BaseNN(torch.nn.Module):
    """Base NN wrapper with convenience functions regarding device and saving.loading weights."""
    def __init__(self, filename, device=None):
        """Init the Colpred object with a model architecture and the weight filename.
        The device is set to cuda if available.
        """
        super(BaseNN, self).__init__()
        if device is None:
            ## get GPU if available; else get CPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            assert device in ['cpu', 'cuda', 'cuda:0', 'cuda:1']
            self.device = device
        if filename.startswith('/'):
            self.filename = filename
        else:
            self.filename = os.path.join(COLPREDMPC_TMP_DIR, filename)
        self.eval()
        self.zero_grad()


    @property
    def device(self):
        return self._dev
    @device.setter
    def device(self, value):
        assert value in ['cpu', 'cuda', 'cuda:0', 'cuda:1']
        self = self.to(value)
        self._dev = value
        for module in self.children():
            module.device = value


    def load_weights(self, filename=None, alter_fn=None):
        """Loads the weights from a given filename, or from self.filename by default.
        A weight alteration function can be passed (eg, to handle API changes)
        """
        if filename is None: filename = self.filename
        if not filename.endswith('.pth'): filename += '.pth'
        weights = torch.load(filename, map_location=self.device, weights_only=False)
        if alter_fn: weights = alter_fn(weights)
        self.load_state_dict(weights)
        self.to(self.device)


    def save_weights(self, filename=None):
        """Save the weights to a given filename, or to self.filename by default."""
        if filename is None: filename = self.filename
        if not filename.endswith('.pth'): filename += '.pth'
        torch.save(self.state_dict(), filename)


    def save_weights_idx(self, idx):
        """Convenience function for saving weights in different files at each epoch.
        A folder named after self.filename is created, then files are named epoch_N.pth.
        """
        if not os.path.isdir(self.filename.split('.')[0]): os.makedirs(self.filename)
        outfile = self.filename + '/epoch_' + str(idx)
        self.save_weights(filename=outfile)


    def init_conv_layers(self, layer):
        """Recursive function to initialize all conv2d layers with xavier_uniform_ throughout the submodules."""
        if type(layer) in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('conv2d'))
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        for ll in layer.children():
            self.init_conv_layers(ll)
