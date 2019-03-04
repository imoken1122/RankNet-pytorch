import torch as th

class CrossEntropyLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.sigma = 1.
  def forward(self,s_i, s_j, u_i, u_j):
    s_ij = th.zeros_like(u_i)
    s_ij[u_i > u_j] = 1
    s_ij[u_i < u_j] = -1
      
    bar_P_ij = 0.5* (1 + s_ij)
    P_ij  =th.sigmoid(self.sigma*(s_i - s_j))
    C = -bar_P_ij * th.log(P_ij) - (1 - bar_P_ij)*th.log(1-P_ij)
    return C.mean()
    
