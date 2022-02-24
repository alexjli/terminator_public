from torch import nn
from torch.utils.checkpoint import checkpoint

# resnet based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# and https://arxiv.org/pdf/1603.05027.pdf


def conv1xN(channels, N):
    return nn.Conv2d(channels, channels, kernel_size=(1, N), padding=(0, N // 2))


class Conv1DResidual(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        hdim = hparams['term_hidden_dim']
        self.bn1 = nn.BatchNorm2d(hdim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1xN(hdim, hparams['conv_filter'])
        self.bn2 = nn.BatchNorm2d(hdim)
        self.conv2 = conv1xN(hdim, hparams['conv_filter'])

    def forward(self, X):
        identity = X

        out = self.bn1(X)

        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)

        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out


class Conv1DResNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        blocks = [self._make_layer(hparams) for _ in range(hparams['matches_blocks'])]
        self.resnet = nn.Sequential(*blocks)

    def _make_layer(self, hparams):
        return Conv1DResidual(hparams)

    def forward(self, X):
        # X: num batches x num channels x TERM length x num alignments
        # out retains the shape of X
        # X = self.bn(X)
        if self.hparams['resnet_linear']:
            out = X
        else:
            out = self.resnet(X)

        # average along axis of alignments
        # out: num batches x hidden dim x TERM length
        out = out.mean(dim=-1)

        # put samples back in rows
        # out: num batches x TERM length x hidden dim
        out = out.transpose(1, 2)

        return out


def conv3x3(channels):
    return nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1))


class Conv2DResidual(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        del hparams

        self.bn1 = nn.BatchNorm2d(2) 
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(2) 
        self.bn2 = nn.BatchNorm2d(2) 
        self.conv2 = conv3x3(2) 

    def forward(self, X):
        identity = X

        out = self.bn1(X)

        out = self.relu(out)
        out = checkpoint(self.conv1, out)

        out = self.bn2(out)

        out = self.relu(out)
        out = checkpoint(self.conv2, out)

        print(out.shape, identity.shape)
        out += identity

        return out


class Conv2DResNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.embed = nn.Conv2d(1, 2, kernel_size=(3, 3), padding=(1, 1))
        blocks = [self._make_layer(hparams) for _ in range(1)]
        self.resnet = nn.Sequential(*blocks)

    def _make_layer(self, hparams):
        return Conv2DResidual(hparams)

    def forward(self, X):
        # X: num batches x num TERMs x TERM length x TERM length x hidden dim x hidden dim
        # out retains the shape of X
        # X = self.bn(X)
        if self.hparams['matches_linear']:
            out = X
        else:
            num_batches, num_terms, term_len, _, hidden_dim = X.shape[:-1]
            flat = X.view(-1, hidden_dim, hidden_dim)
            flat = flat.unsqueeze(1)
            out = self.embed(flat)
            out = self.resnet(out)
            out = out.mean(dim=1)
            out = out.view(num_batches, num_terms, term_len, term_len, hidden_dim, hidden_dim)

        return out
