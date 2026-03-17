import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()                                    
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice