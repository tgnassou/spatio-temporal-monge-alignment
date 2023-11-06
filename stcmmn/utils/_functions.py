from braindecode.models import ShallowFBCSPNet
import numpy as np


def compute_final_conv_length(n_list):
    length = []
    for n in n_list:
        length += [ShallowFBCSPNet(
            2,
            2,
            n_times=n,
            final_conv_length='auto',
            add_log_softmax=False,
        ).final_conv_length]

    final_conv_length = np.min(length)

    return final_conv_length
