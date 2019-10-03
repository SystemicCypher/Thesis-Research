import torch as t
from memory import *
from controller import *

class DNC(t.nn.Module):
    '''
    DNC's are comprised of a controller and memory layer
    '''
    def __init__(self, memory_size, num_controller_layers ):
        super(DNC, self).__init__()

    def generate_out_interf_vec(self, hidden):
        w_y = t.randn(hidden.shape[1])
        w_i = t.randn(hidden.shape[1])

        v_t = t.einsum('h,hl->l', w_y, hidden)
        i_t = t.einsum('h,hl->l', w_i, hidden)

        return v_t, i_t

    def forward(self, input_vec):
        pass