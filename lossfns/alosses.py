import torch

# ---------------------------------------------------------------------------
class FBMBLoss(torch.nn.Module):
    """Frequency-band model-based loss"""

    def __init__(self, L1, L2, etaS, eta):
        super(FBMBLoss, self).__init__()
        self.L1 = L1
        self.L2 = L2
        self.etaS = etaS
        self.eta = eta

    def forward(self, pred, target, sLf, sHf):
        diff = pred - target
        To = diff * diff
        TL = sLf * sLf
        TH = sHf * sHf
        loss = torch.mean(self.etaS*To + self.eta*self.L1*TL + self.eta*self.L2*TH)
        return loss