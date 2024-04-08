import numpy as np
from typing import Tuple, List
import neural.initializers


class StdLayer:
    def __init__(self, kernel_initializer, bias_initializer):

        if type(kernel_initializer) == str:
            self.kernel_initializer = neural.initializers.initializers_dict[kernel_initializer]()
        else:
            self.kernel_initializer = kernel_initializer

        if type(bias_initializer) == str:
            self.bias_initializer = neural.initializers.initializers_dict[bias_initializer]()
        else:
            self.bias_initializer = bias_initializer

        self.kernel = None
        self.bias = None

    def __call__(self, inputs, **kwargs):
        pass

    def _initialize_weights(self):
        pass

    def gradient(self, prev_grad, last) -> Tuple[np.ndarray, np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray]:
        pass

    def update_weights(self, new_weights: Tuple[np.ndarray, np.ndarray] or List[np.ndarray, np.ndarray]):
        self.kernel = new_weights[0]
        self.bias = new_weights[1]

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.kernel, self.bias

    @staticmethod
    def get_base():
        pass


class NoWeightsLayer:
    def __init__(self):
        pass

    def __call__(self, inputs, **kwargs):
        pass

    def gradient(self, prev_grad) -> Tuple[np.ndarray]:
        pass

    @staticmethod
    def get_base():
        pass


class FullyConnected(StdLayer):
    def __init__(self,
                 neurons: int,
                 l2_regularizer=0,
                 kernel_initializer="xavier_normal",
                 bias_initializer="zeros",
                 ):
        super(FullyConnected, self).__init__(kernel_initializer, bias_initializer)

        self.neurons = neurons
        self.l2_reg = l2_regularizer
        self.input_shape = None
        self.inputs = None

    def __call__(self, inputs, **kwargs):
        inputs = np.array(inputs)

        if self.input_shape is None:
            self.input_shape = inputs.shape[-1]
            self._initialize_weights()

        self.inputs = inputs

        w = self.kernel
        b = self.bias
        z = np.matmul(inputs, w) + b

        return z

    def _initialize_weights(self):
        self.kernel = self.kernel_initializer(shape=(self.input_shape, self.neurons))
        self.bias = self.bias_initializer(shape=(1, self.neurons))

    def gradient(self, prev_grad, last=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray]:
        w = self.kernel
        a = self.inputs

        dz_dw = np.matmul(a.T, prev_grad) + self.l2_reg * w
        dz_db = np.sum(prev_grad, axis=0, keepdims=True)

        if not last:
            dz_da = np.matmul(prev_grad, w.T)
            return dz_dw, dz_db, dz_da
        else:
            return dz_dw, dz_db

    @staticmethod
    def get_base():
        return FullyConnected.__base__


class Conv2D(StdLayer):
    def __init__(self,
                 num_of_filters,
                 filter_size,
                 padding="VALID",
                 kernel_initializer="xavier_normal",
                 bias_initializer="zeros",
                 ):
        super(Conv2D, self).__init__(kernel_initializer, bias_initializer)
        self.k = num_of_filters

        if type(filter_size) == tuple or type(filter_size) == list:
            self.f = (filter_size[0], filter_size[1])
        else:
            self.f = (filter_size, filter_size)

        self.input_channels = None
        self.input_shape = None

        self.inputs = None

        self.p = None
        if type(padding) == str:
            padding = padding.lower()
            if padding == "valid":
                self.p = (0, 0)
            elif padding == "same":
                self.p = -1
            else:
                raise Exception("Invalid Padding String")
        elif type(padding) == int:
            self.p = (padding, padding)
        else:
            self.p = padding

    def __call__(self, inputs, **kwargs):
        inputs = np.array(inputs)

        if self.input_shape is None:
            self.input_shape = inputs.shape
            self.input_channels = inputs.shape[-1]
            self._initialize_weights()

        if self.p == -1:    # SAME padding
            p_h = (self.f[0] - 1) // 2
            p_w = (self.f[1] - 1) // 2
            self.p = (p_h, p_w)

        self.inputs = np.pad(inputs, ((0, 0), (self.p[0], self.p[0]), (self.p[1], self.p[1]), (0, 0)))

        b = self.bias
        w = self.kernel
        z = neural.op.convolve2d(self.inputs, w) + b

        return z

    def _initialize_weights(self):
        self.kernel = self.kernel_initializer(shape=(self.f[0], self.f[1], self.input_channels, self.k))
        self.bias = self.bias_initializer(shape=(1, 1, 1, self.k))

    def gradient(self, prev_grad, last=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray]:
        w = self.kernel
        a = self.inputs
        p_h = self.p[0]
        p_w = self.p[1]

        dz_dw = neural.op.convolve2d_arr_arr(a, prev_grad)
        dz_db = np.sum(prev_grad, axis=(0, 1, 2))

        if not last:
            dz_da = neural.op.convolve2d_ker_arr(w, prev_grad)

            h_padded = dz_da.shape[1]
            w_padded = dz_da.shape[2]

            dz_da = dz_da[:, p_h:h_padded-p_h, p_w:w_padded-p_w, :]

            return dz_dw, dz_db, dz_da
        else:
            return dz_dw, dz_db

    @staticmethod
    def get_base():
        return Conv2D.__base__


class Activation(NoWeightsLayer):
    def __init__(self, activation):
        super(Activation, self).__init__()

        if type(activation) == str:
            self.activation = neural.activations.activations_dict[activation]()
        else:
            self.activation = activation

    def __call__(self, inputs, **kwargs):
        a = self.activation(inputs)

        return a

    def gradient(self, prev_grad) -> Tuple[np.ndarray]:
        da_dz = self.activation.gradient(prev_grad)

        return da_dz

    @staticmethod
    def get_base():
        return Activation.__base__


class Dropout(NoWeightsLayer):
    def __init__(self, dropout_rate):
        super(Dropout, self).__init__()

        self.keep_rate = 1 - dropout_rate
        self.input_shape = None
        self.dropout_mask = None

    def __call__(self, inputs, **kwargs):
        inputs = np.array(inputs)
        self.input_shape = inputs.shape

        if len(kwargs) == 0 or kwargs["train"] is True:
            self.dropout_mask = (np.random.uniform(0, 1, size=self.input_shape) < self.keep_rate).astype(np.float_)
            inputs = inputs * self.dropout_mask
            inputs = inputs / self.keep_rate
        else:
            self.dropout_mask = np.ones(shape=self.input_shape)

        return inputs

    def gradient(self, prev_grad) -> Tuple[np.ndarray]:
        grad = self.dropout_mask / self.keep_rate

        grad = grad * prev_grad

        return grad

    @staticmethod
    def get_base():
        return Dropout.__base__


class Flatten(NoWeightsLayer):
    def __init__(self):
        super(Flatten, self).__init__()

        self.input_shape = None

    def __call__(self, inputs, **kwargs):
        inputs = np.array(inputs)

        self.input_shape = inputs.shape
        outputs = inputs.reshape((self.input_shape[0], -1))

        return outputs

    def gradient(self, prev_grad) -> Tuple[np.ndarray]:
        grad = prev_grad.reshape(self.input_shape)

        return grad

    @staticmethod
    def get_base():
        return Flatten.__base__


class MaxPooling2D(NoWeightsLayer):
    def __init__(self, filter_size=(2, 2), strides=(2, 2)):
        super(MaxPooling2D, self).__init__()

        self.filter_size = filter_size
        self.strides = strides

        self.max_ind = None
        self.input_shape = None

    def __call__(self, inputs, **kwargs):
        inputs = np.array(inputs)

        self.input_shape = inputs.shape

        outputs, self.max_ind = neural.op.max_pooling2d(
            inputs,
            filter_size=self.filter_size,
            strides=self.strides
        )

        return outputs

    def gradient(self, prev_grad) -> Tuple[np.ndarray]:
        in_sz = (self.input_shape[1], self.input_shape[2])

        grad = neural.op.max_upsampling2d(prev_grad, in_sz, self.max_ind)

        return grad

    @staticmethod
    def get_base():
        return MaxPooling2D.__base__
