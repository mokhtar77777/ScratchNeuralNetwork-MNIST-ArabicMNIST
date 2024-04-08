import numpy as np


class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

    def gradient(self, prev_grad):
        pass


class Linear(ActivationFunction):
    def __init__(self):
        super(Linear, self).__init__()

        self.value = None

    def __call__(self, inputs):
        self.value = np.array(inputs)
        return self.value

    def gradient(self, prev_grad=None):
        grad_shape = self.value.shape
        grad = np.ones(shape=grad_shape)

        if prev_grad is not None:
            grad = grad * prev_grad

        return grad


class Sigmoid(ActivationFunction):
    def __init__(self):
        super(Sigmoid, self).__init__()

        self.value = None

    def __call__(self, inputs):
        inputs = np.array(inputs)
        self.value = 1 / (1 + np.exp(-inputs))
        return self.value

    def gradient(self, prev_grad=None):
        grad = self.value * (1 - self.value)

        if prev_grad is not None:
            grad = grad * prev_grad

        return grad


class Tanh(ActivationFunction):
    def __init__(self):
        super(Tanh, self).__init__()

        self.value = None

    def __call__(self, inputs):
        inputs = np.array(inputs)
        self.value = np.tanh(inputs)
        return self.value

    def gradient(self, prev_grad=None):
        grad = 1 - (self.value ** 2)

        if prev_grad is not None:
            grad = grad * prev_grad

        return grad


class Relu(ActivationFunction):
    def __init__(self):
        super(Relu, self).__init__()

        self.value = None

    def __call__(self, inputs):
        inputs = np.array(inputs)
        self.value = np.maximum(inputs, 0)

        return self.value

    def gradient(self, prev_grad=None):
        grad = (self.value > 0).astype(float)

        if prev_grad is not None:
            grad = grad * prev_grad

        return grad


class LeakyRelu(ActivationFunction):
    def __init__(self, alpha=0.3):
        super(LeakyRelu, self).__init__()

        self.alpha = alpha
        self.value = None

    def __call__(self, inputs):
        inputs = np.array(inputs)
        self.value = np.where(inputs > 0, inputs, inputs * self.alpha)

        return self.value

    def gradient(self, prev_grad=None):
        grad = np.where(self.value > 0, 1, self.alpha)

        if prev_grad is not None:
            grad = grad * prev_grad

        return grad


class Softplus(ActivationFunction):
    def __init__(self):
        super(Softplus, self).__init__()

        self.value = None
        self.exp_terms = None

    def __call__(self, inputs):
        inputs = np.array(inputs)
        self.exp_terms = np.exp(inputs)
        self.value = np.log(1 + self.exp_terms)

        return self.value

    def gradient(self, prev_grad=None):
        grad = self.exp_terms / (1 + self.exp_terms)

        if prev_grad is not None:
            grad = grad * prev_grad

        return grad


class Softmax(ActivationFunction):
    def __init__(self):
        super(Softmax, self).__init__()

        self.value = None

    def __call__(self, inputs):
        inputs = np.array(inputs)
        exp_terms = np.exp(inputs)
        exp_sum = np.sum(exp_terms, axis=1, keepdims=True)

        self.value = exp_terms / exp_sum

        return self.value

    def gradient(self, prev_grad=None):
        value_high_dim = np.expand_dims(self.value, axis=0)
        value_high_dim_twin = value_high_dim.T

        value_mask = np.eye(self.value.shape[1]) - 1
        value_mask = np.expand_dims(value_mask, axis=1)

        value_high_dim_masked = value_high_dim * value_mask

        grad = value_high_dim_masked * value_high_dim_twin

        value_mask = value_mask + 1

        trim_value_high_dim = value_high_dim * (1 - value_high_dim)

        grad = trim_value_high_dim * value_mask + grad

        if prev_grad is not None:
            prev_grad = np.expand_dims(prev_grad, axis=0)
            grad = np.einsum("abc,cbe->abe", prev_grad, grad).squeeze(axis=0)

        return grad


activations_dict = {
    "linear": Linear,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": Relu,
    "leaky_relu": LeakyRelu,
    "softplus": Softplus,
    "softmax": Softmax
}
