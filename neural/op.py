import numpy as np
from scipy import signal


def convolve2d(arr, kernel, strides=(1, 1), rotate180=False, axes=(1, 2, 3)):
    arr = np.array(arr)
    kernel = np.array(kernel)
    fh = kernel.shape[0]
    fw = kernel.shape[1]
    sh = strides[0]
    sw = strides[1]
    m = arr.shape[0]
    h_orig = arr.shape[1]
    w_orig = arr.shape[2]
    h = ((h_orig - kernel.shape[0]) // sh) + 1
    w = ((w_orig - kernel.shape[1]) // sw) + 1
    c = kernel.shape[3]

    if rotate180:
        kernel = np.rot90(kernel, 2, axes=(0, 1))

    if kernel.shape[2] != arr.shape[3]:
        raise Exception("3rd dimension in kernel must match the 4th dimension in arr")

    out = np.zeros(shape=(m, h, w, c))

    def reduce(arr_slc, kernel_slc):
        return np.sum(arr_slc * kernel_slc, axis=axes)

    broadcast_kernel = np.expand_dims(kernel, axis=0)

    broadcast_arr = np.expand_dims(arr, axis=-1)

    for row_ind, row in enumerate(range(fh, h_orig+1, sh)):
        for col_ind, col in enumerate(range(fw, w_orig+1, sw)):
            out[:, row_ind, col_ind, :] = reduce(broadcast_arr[:, row-fh:row, col-fw:col, :, :], broadcast_kernel)

    return out


def fft_convolve2d(arr, kernel, strides=(1, 1), rotate180=False):
    arr = np.array(arr)
    kernel = np.array(kernel)
    sh = strides[0]
    sw = strides[1]
    m = arr.shape[0]
    h_orig = arr.shape[1]
    w_orig = arr.shape[2]
    h = ((h_orig - kernel.shape[0]) // sh) + 1
    w = ((w_orig - kernel.shape[1]) // sw) + 1
    c = kernel.shape[3]

    if not rotate180:
        kernel = np.rot90(kernel, 2, axes=(0, 1))

    if kernel.shape[2] != arr.shape[3]:
        raise Exception("3rd dimension in kernel must match the 4th dimension in arr")

    broadcast_kernel = np.expand_dims(kernel, axis=0)
    broadcast_arr = np.expand_dims(arr, axis=-1)

    arr_fft = np.fft.fftn(broadcast_arr, s=(m, h, w, 1, c), axes=(0, 1, 2, 3, 4))
    kernel_fft = np.fft.fftn(broadcast_kernel, s=(m, h, w, 1, c), axes=(0, 1, 2, 3, 4))

    out_fft = arr_fft * kernel_fft

    out = np.real(np.fft.ifftn(out_fft))
    out = out.squeeze(axis=-2)

    return out


def convolve2d_arr_arr(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    hl_1 = arr1.shape[1]
    hl = arr2.shape[1]

    wl_1 = arr1.shape[2]
    wl = arr2.shape[2]

    fh = hl_1 - hl + 1
    fw = wl_1 - wl + 1

    out = np.zeros(shape=(fh, fw, arr1.shape[-1], arr2.shape[-1]))

    arr1 = np.expand_dims(arr1, axis=-1)  # (m, hl-1, wl-1, cl-1, 1)
    arr2 = np.expand_dims(arr2, axis=-2)  # (m, hl, w1, 1, cl)

    def reduce(arr_slc, kernel_slc):
        return np.sum(arr_slc * kernel_slc, axis=(0, 1, 2))

    for row in range(fh):
        for col in range(fw):
            out[row, col, :, :] = reduce(arr1[:, row:row+hl, col:col+wl, :, :], arr2)

    return out


def convolve2d_ker_arr(ker, arr):
    ker = np.array(ker)
    arr = np.array(arr)

    # m = arr.shape[0]

    hl = arr.shape[1]
    wl = arr.shape[2]

    fh = ker.shape[0]
    fw = ker.shape[1]

    # fh_padded = 2 * hl - 2 + fh
    # fw_padded = 2 * wl - 2 + fw

    # hl_1 = fh_padded - hl + 1
    # wl_1 = fw_padded - wl + 1

    # out = np.zeros(shape=(m, hl_1, wl_1, ker.shape[2]))

    if ker.shape[-1] != arr.shape[-1]:
        raise Exception("4th dimension in kernel must match the 4th dimension in arr")

    # ker_padded = np.zeros(shape=(fh_padded, fw_padded, ker.shape[2], ker.shape[3]))
    #
    # ker_padded[hl-1:hl+fh-1, wl-1:wl+fw-1, :, :] = ker

    ker_padded = np.pad(ker, ((hl - 1, hl - 1), (wl - 1, wl - 1), (0, 0), (0, 0)))

    # arr = np.rot90(arr, 2, axes=(1, 2))

    arr = np.expand_dims(arr, axis=-2)  # (m, hl, w1, 1, cl)
    ker_padded = np.expand_dims(ker_padded, axis=0)  # (1, fh, fw, cl-1, cl)

    # def reduce(ker_slc, arr_slc):
    #     return np.sum(ker_slc * arr_slc, axis=(1, 2, -1))
    #
    # for row in range(hl_1):
    #     for col in range(wl_1):
    #         out[:, row, col, :] = reduce(ker_padded[:, row:row+hl, col:col+wl, :, :], arr)

    out = np.sum(signal.fftconvolve(ker_padded, arr, axes=(1, 2), mode="valid"), axis=-1)

    return out


def max_pooling2d(arr, filter_size=(2, 2), strides=(2, 2), return_mask=False):
    arr = np.array(arr)

    if type(filter_size) == int:
        fh = filter_size
        fw = filter_size
    else:
        fh = filter_size[0]
        fw = filter_size[1]

    if type(strides) == int:
        sh = strides
        sw = strides
    else:
        sh = strides[0]
        sw = strides[1]

    m = arr.shape[0]
    c = arr.shape[-1]
    h_orig = arr.shape[1]
    w_orig = arr.shape[2]

    h = ((h_orig - fh) // sh) + 1
    w = ((w_orig - fw) // sw) + 1

    out = np.zeros(shape=(m, h, w, c))

    out_max_ind = np.zeros(shape=(2, m, h, w, c))

    input_mask = None
    if return_mask:
        input_mask = np.zeros(shape=(m, h_orig, w_orig, c))

    def mask(slc, row_offset, col_offset):
        ind = np.argmax(slc.reshape(m, -1, c), axis=1)
        ind_unravel = np.unravel_index(ind, shape=(fh, fw))
        return ind_unravel[0] + row_offset, ind_unravel[1] + col_offset

    def reduce(slc):
        return np.max(slc, axis=(1, 2))

    for row_ind, row in enumerate(range(fh, h_orig+1, sh)):
        for col_ind, col in enumerate(range(fw, w_orig+1, sw)):
            max_ind = mask(arr[:, row-fh:row, col-fw:col, :], row_offset=row-fh, col_offset=col-fw)
            out_max_ind[:, :, row_ind, col_ind, :] = max_ind
            # out[:, row_ind, col_ind, :] = \
            #     arr[np.repeat(np.arange(m), c), max_ind[0].reshape(-1), max_ind[1].reshape(-1), np.tile(np.arange(c), m)]
            out[:, row_ind, col_ind, :] = reduce(arr[:, row-fh:row, col-fw:col, :])
            if return_mask:
                input_mask[
                    np.repeat(np.arange(m), c),
                    max_ind[0].reshape(-1), max_ind[1].reshape(-1),
                    np.tile(np.arange(c), m)
                ] += 1

    if return_mask:
        return out, out_max_ind, input_mask
    else:
        return out, out_max_ind


def max_upsampling2d(arr, out_size, max_ind, output_mask=None):
    arr = np.array(arr)
    max_ind = np.array(max_ind)

    if len(arr.shape) != 4:
        raise Exception("arr must be a 4D array")

    if len(max_ind.shape) != 5:
        raise Exception("max_ind must be a 5D array")

    if arr.shape != max_ind[0, :, :, :].shape:
        raise Exception("arr must have the same dimension as max_ind excluding the first dimension of max_ind")

    h_out = out_size[0]
    w_out = out_size[1]

    m = max_ind.shape[1]
    h_in = max_ind.shape[2]
    w_in = max_ind.shape[3]
    c = max_ind.shape[4]
    s = h_in * w_in

    out = np.zeros(shape=(m, h_out, w_out, c))

    max_ind_simple = max_ind.reshape((2, m, s, c))

    m_repeats = np.repeat(np.arange(m), s * c).astype(int)
    max_row_ind = max_ind_simple[0].reshape(-1).astype(int)
    max_col_ind = max_ind_simple[1].reshape(-1).astype(int)
    c_repeats = np.tile(np.arange(c), s * m).astype(int)
    arr_reshaped = arr.reshape(-1)

    max_ind_simpler = np.array([m_repeats, max_row_ind, max_col_ind, c_repeats])
    lex_ind = np.lexsort([c_repeats, max_col_ind, max_row_ind, m_repeats])

    arr_reshaped = arr_reshaped[lex_ind]
    max_ind_simpler = max_ind_simpler[:, lex_ind]

    _, ind, counts = np.unique(max_ind_simpler, return_index=True, return_counts=True, axis=1)

    m_repeats_red = max_ind_simpler[0, ind]
    max_row_ind_red = max_ind_simpler[1, ind]
    max_col_ind_red = max_ind_simpler[2, ind]
    c_repeats_red = max_ind_simpler[3, ind]
    arr_reshaped_red = arr_reshaped[ind]

    big_counts_ind = np.where(counts > 1)[0]

    for count_ind in big_counts_ind:
        arr_reshaped_red[count_ind] = np.sum(arr_reshaped[ind[count_ind]:ind[count_ind]+counts[count_ind]])

    out[m_repeats_red, max_row_ind_red, max_col_ind_red, c_repeats_red] = arr_reshaped_red

    if output_mask is not None:
        out = out * output_mask

    return out
