import numpy as np


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def gradient_step(self, old_params, grad):
        pass

    def adjust_lr(self, new_lr):
        self.lr = new_lr

    def get_lr(self):
        return self.lr


class GradientDescent(Optimizer):
    def __init__(self, lr=0.01, momentum=0):
        super(GradientDescent, self).__init__(lr)

        self.momentum = momentum
        self.velocity = None

    def gradient_step(self, old_params, grad):
        length = len(old_params)

        if self.velocity is None:
            self.velocity = [0] * length

        for param in range(length):

            self.velocity[param] = self.momentum * self.velocity[param] - self.lr * grad[param]
            old_params[param] += self.velocity[param]


class Adagrad(Optimizer):
    def __init__(self, lr=0.001, epsilon=1e-07):
        super(Adagrad, self).__init__(lr)

        self.epsilon = epsilon
        self.s = None

    def gradient_step(self, old_params, grad):
        length = len(old_params)

        if self.s is None:
            self.s = [0] * length

        for param in range(length):

            self.s[param] = self.s[param] + (grad[param] ** 2)
            cur_lr = self.lr / (np.sqrt(self.s[param]) + self.epsilon)

            old_params[param] -= (cur_lr * grad[param])


class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-07):
        super(RMSprop, self).__init__(lr)

        self.epsilon = epsilon
        self.rho = rho
        self.s = None

    def gradient_step(self, old_params, grad):
        length = len(old_params)

        if self.s is None:
            self.s = [0] * length

        for param in range(length):

            self.s[param] = self.rho * self.s[param] + (1 - self.rho) * (grad[param] ** 2)
            cur_lr = self.lr / (np.sqrt(self.s[param]) + self.epsilon)

            old_params[param] -= (cur_lr * grad[param])


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07):
        super(Adam, self).__init__(lr)

        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.cur_step = 0
        self.velocity = None
        self.s = None

    def gradient_step(self, old_params, grad):
        self.cur_step += 1

        length = len(old_params)

        if self.velocity is None:
            self.velocity = [0] * length

        if self.s is None:
            self.s = [0] * length

        for param in range(length):

            self.velocity[param] = self.beta1 * self.velocity[param] + (1 - self.beta1) * grad[param]
            self.s[param] = self.beta2 * self.s[param] + (1 - self.beta2) * (grad[param] ** 2)

            vel_cor = self.velocity[param] / (1 - self.beta1 ** self.cur_step)
            s_cor = self.s[param] / (1 - self.beta2 ** self.cur_step)

            cur_lr = self.lr / (np.sqrt(s_cor) + self.epsilon)

            old_params[param] -= (cur_lr * vel_cor)


class Nadam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07):
        super(Nadam, self).__init__(lr)

        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.cur_step = 0
        self.velocity = None
        self.s = None

    def gradient_step(self, old_params, grad):
        self.cur_step += 1

        length = len(old_params)

        if self.velocity is None:
            self.velocity = [0] * length

        if self.s is None:
            self.s = [0] * length

        for param in range(length):

            self.velocity[param] = self.beta1 * self.velocity[param] + (1 - self.beta1) * grad[param]
            self.s[param] = self.beta2 * self.s[param] + (1 - self.beta2) * (grad[param] ** 2)

            vel_cor = self.velocity[param] / (1 - self.beta1 ** self.cur_step)
            s_cor = self.s[param] / (1 - self.beta2 ** self.cur_step)

            cur_lr = self.lr / (np.sqrt(s_cor) + self.epsilon)

            old_params[param] -= (cur_lr * (self.beta1 * vel_cor + (1 - self.beta1) * grad[param]))


optimizers_dict = {
    "gd": GradientDescent,
    "sgd": GradientDescent,
    "adagrad": Adagrad,
    "RMSprop": RMSprop,
    "adam": Adam,
    "nadam": Nadam
}
