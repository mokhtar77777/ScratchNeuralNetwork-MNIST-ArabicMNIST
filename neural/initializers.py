import numpy as np


class Initializer:
    def __init__(self):
        pass

    def __call__(self, shape: tuple):
        pass


class RandomNormal(Initializer):
    def __init__(self, mean=0, std=1):
        super(RandomNormal, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, shape: tuple):
        weights = np.random.normal(loc=self.mean, scale=self.std, size=shape)

        return weights


class GlorotNormal(Initializer):
    def __init__(self):
        super(GlorotNormal, self).__init__()
        self.in_num = None
        self.out_num = None
        self.rem = None

    def __call__(self, shape: tuple):
        self.in_num = shape[-2]
        self.out_num = shape[-1]
        self.rem = np.prod(shape[: -2])

        std = np.sqrt(2 / (self.rem * (self.in_num + self.out_num)))

        weights = np.random.normal(loc=0, scale=std, size=shape)

        return weights


class HeNormal(Initializer):
    def __init__(self):
        super(HeNormal, self).__init__()
        self.in_num = None
        self.rem = None

    def __call__(self, shape: tuple):
        self.in_num = shape[-2]
        self.rem = np.prod(shape[: -2])

        std = np.sqrt(2 / (self.rem * self.in_num))

        weights = np.random.normal(loc=0, scale=std, size=shape)

        return weights


class LecunNormal(Initializer):
    def __init__(self):
        super(LecunNormal, self).__init__()
        self.in_num = None
        self.rem = None

    def __call__(self, shape: tuple):
        self.in_num = shape[-2]
        self.rem = np.prod(shape[: -2])

        std = np.sqrt(1 / (self.rem * self.in_num))

        weights = np.random.normal(loc=0, scale=std, size=shape)

        return weights


class Zeros(Initializer):
    def __init__(self):
        super(Zeros, self).__init__()

    def __call__(self, shape: tuple):
        weights = np.zeros(shape=shape, dtype=np.float_)

        return weights


class Ones(Initializer):
    def __init__(self):
        super(Ones, self).__init__()

    def __call__(self, shape: tuple):
        weights = np.ones(shape=shape, dtype=np.float_)

        return weights


initializers_dict = {
    "random_normal": RandomNormal,
    "glorot_normal": GlorotNormal,
    "xavier_normal": GlorotNormal,
    "he_normal": HeNormal,
    "lecun_normal": LecunNormal,
    "ones": Ones,
    "zeros": Zeros
}
