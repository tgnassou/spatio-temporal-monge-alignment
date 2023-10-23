from braindecode.models import ShallowFBCSPNet
import numpy as np


def compute_final_conv_length(n_1, n_2):
    length_1 = ShallowFBCSPNet(
        2,
        2,
        n_times=n_1,
        final_conv_length='auto',
        add_log_softmax=False,
    ).final_conv_length

    length_2 = ShallowFBCSPNet(
            2,
            2,
            n_times=n_2,
            final_conv_length='auto',
            add_log_softmax=False,
        ).final_conv_length
    final_conv_length = np.min([length_1, length_2])

    return final_conv_length
