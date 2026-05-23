import math
import numpy as np
import torch
import config


def print_epoch(e, max_e):
    print(config.sep_width * "-")
    print("Epoch (" + str(e) + '/' + str(max_e) + ')')
    print(config.sep_width * "-")


def p_step(loss, step, epoch):
    step += 1

    if config.device == 'cuda':
        l_item = loss.cpu().detach().numpy()
        if isinstance(l_item, np.ndarray):
            l_item = l_item.item()
    else:
        l_item = loss.item()

    print('Epoch: ' + str(epoch) + ' Step ' + str(step) + ' Loss: ' + str(round(l_item, 4)))
    return step


def pe_encoding(pos, dim=None):
    """
    Sinusoidal positional encoding for [lat, lon].

    This version matches:
        PE_{2k}(pc)   = sin(lambda * pc * 10000^{-2k/d})
        PE_{2k+1}(pc) = cos(lambda * pc * 10000^{-2k/d})

    The output order follows the original code:
        [lat_i0, lon_i0, lat_i2, lon_i2, ...]
    If your paper writes Concat(PE(lat), PE(lon)), switch the two loops.
    """
    if dim is None:
        dim = config.pe_size

    p_enc = []
    for i in range(0, dim // 2, 2):
        for loc in pos:
            w_k = config.lambda_ / pow(10000, i / (dim // 2))
            p_enc.append(math.sin(float(loc) * w_k))
            p_enc.append(math.cos(float(loc) * w_k))

    return np.array(p_enc, dtype=np.float32)


def mape(y_t, p):
    y_t, p = np.array(y_t), np.array(p)
    return 100 * np.mean(np.abs((y_t - p) / (y_t + 1e-6)))


def smape(y_t, p):
    y_t, p = np.array(y_t), np.array(p)
    return 100 * np.mean(
        2 * np.abs(p - y_t) / (np.abs(y_t) + np.abs(p) + 1e-6)
    )
