import torch
import math

def PositionalEncoding(d_p, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_p % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dimension (got dim={:d})".format(d_p))
    pe = torch.zeros(d_p, height, width)
    pe.requires_grad = False
    # Each dimension use half of d_model
    d_p = int(d_p / 2)
    div_term = torch.exp(torch.arange(0., d_p, 2) *
                        -(math.log(10000.0) / d_p))
    # div_term = 1/(d_model-1)
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    # pos_ww = torch.arange(0., d_model*2).unsqueeze(1)
    # pos_hh = torch.arange(0., d_model*2).unsqueeze(1)
    
    # print(pos_w)
    pos_w = pos_w/(pos_w.shape[0]-1)
    pos_h = pos_w/(pos_h.shape[0]-1)
    # print(pos_w)
    
    pe[0:d_p:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_p:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_p::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_p + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe