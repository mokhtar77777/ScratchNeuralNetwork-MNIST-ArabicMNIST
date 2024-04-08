import numpy as np
from neural.CONSTANTS import *


class Loss:
    def __init__(self):
        pass

    def __call__(self, y_predict, y_true):
        pass

    def gradient(self):
        pass


class MeanSquaredError(Loss):
    def __init__(self):
        super(MeanSquaredError, self).__init__()
        self.y_predict = None
        self.y_true = None

        self.batches = None
        self.features = None

    def __call__(self, y_predict, y_true):
        self.y_predict = np.array(y_predict)
        self.y_true = np.array(y_true)

        self.batches = self.y_true.shape[0]
        self.features = self.y_true.shape[1]

        loss = 0.5 * np.sum((self.y_predict - self.y_true) ** 2)

        loss = loss / self.batches / self.features

        return loss

    def gradient(self):
        grad = self.y_predict - self.y_true
        grad = grad / self.batches / self.features

        return grad


class BinaryHinge(Loss):
    def __init__(self):
        super(BinaryHinge, self).__init__()
        self.y_predict = None
        self.y_true = None

        self.batches = None
        self.features = None

    def __call__(self, y_predict, y_true):
        self.y_predict = np.array(y_predict)
        self.y_true = np.array(y_true)

        self.y_true = np.where(self.y_true <= 0.5, -1, 1)

        self.batches = self.y_true.shape[0]
        self.features = self.y_true.shape[1]

        loss = np.sum(np.maximum(1 - self.y_true * self.y_predict, 0))

        loss = loss / self.features / self.batches

        return loss

    def gradient(self):
        gradient_mask = ((1 - self.y_true * self.y_predict) > 0)
        grad = gradient_mask * (-self.y_true)
        grad = grad / self.features / self.batches

        return grad


class BinaryCrossEntropy(Loss):
    def __init__(self, from_logits=False):
        super(BinaryCrossEntropy, self).__init__()

        self.from_logits = from_logits

        self.y_predict = None
        self.y_true = None

        self.features = None
        self.batches = None

    def __call__(self, y_predict, y_true):
        self.y_predict = np.array(y_predict)
        self.y_true = np.array(y_true)

        self.y_true = np.where(self.y_true <= 0.5, 0, 1)

        self.features = self.y_true.shape[1]
        self.batches = self.y_true.shape[0]

        if not self.from_logits:
            self.y_predict = np.where(self.y_predict > TENDS_TO_ONE, TENDS_TO_ONE, self.y_predict)
            self.y_predict = np.where(self.y_predict < TENDS_TO_ZER0, TENDS_TO_ZER0, self.y_predict)
            loss = np.sum(-self.y_true * np.log(self.y_predict) - (1 - self.y_true) * np.log(1 - self.y_predict))

        else:
            loss = np.sum(self.y_predict - self.y_true * self.y_predict + np.log(1 + np.exp(-self.y_predict)))

        loss = loss / self.features / self.batches

        return loss

    def gradient(self):
        if not self.from_logits:
            grad = (1 - self.y_true) / (1 - self.y_predict) - self.y_true / self.y_predict
        else:
            sigmoid_out = 1 / (1 + np.exp(-self.y_predict))
            grad = sigmoid_out - self.y_true

        grad = grad / self.features / self.batches

        return grad


class OneHotCrossEntropy(Loss):
    def __init__(self, from_logits=False):
        super(OneHotCrossEntropy, self).__init__()
        self.from_logits = from_logits

        self.y_predict = None
        self.y_true = None

        self.batches = None

    def __call__(self, y_predict, y_true):
        self.y_predict = np.array(y_predict)
        self.y_true = np.array(y_true)

        self.batches = self.y_true.shape[0]

        self.exp_terms = None
        self.exp_terms_sum = None

        if not self.from_logits:
            self.y_predict = np.where(self.y_predict > TENDS_TO_ONE, TENDS_TO_ONE, self.y_predict)
            self.y_predict = np.where(self.y_predict < TENDS_TO_ZER0, TENDS_TO_ZER0, self.y_predict)
            loss = -np.log(self.y_predict)
            loss = np.sum(loss * self.y_true)

        else:
            self.exp_terms = np.exp(self.y_predict)
            self.exp_terms_sum = np.sum(self.exp_terms, axis=1, keepdims=True)
            logits = self.y_predict * self.y_true
            loss = np.sum(np.log(self.exp_terms_sum) * self.y_true - logits)

        loss = loss / self.batches

        return loss

    def gradient(self):
        if not self.from_logits:
            grad = -1 / self.y_predict
            grad = grad * self.y_true
        else:
            softmax_out = self.exp_terms / self.exp_terms_sum
            grad = softmax_out - self.y_true

        grad = grad / self.batches

        return grad


class OneHotHinge(Loss):
    def __init__(self):
        super(OneHotHinge, self).__init__()

        self.y_predict = None
        self.y_true = None

        self.pos_mask = None
        self.loss_no_reduce = None

        self.batches = None

    def __call__(self, y_predict, y_true):
        self.y_predict = np.array(y_predict)
        self.y_true = np.array(y_true)

        self.batches = self.y_true.shape[0]

        neg = np.sum(self.y_predict * self.y_true, axis=1)
        self.pos_mask = self.y_predict * (1 - self.y_true)
        pos = np.max(self.pos_mask, axis=1)

        self.loss_no_reduce = np.maximum(pos - neg + 1, 0)

        loss = np.sum(self.loss_no_reduce)

        loss = loss / self.batches

        return loss

    def gradient(self):
        y_true_mask = -self.y_true
        y_predict_mask = (self.pos_mask == self.pos_mask.max(axis=1, keepdims=True))

        loss_mask = (self.loss_no_reduce > 0)

        grad = (y_true_mask + y_predict_mask) * np.expand_dims(loss_mask, axis=1)
        grad = grad / self.batches

        return grad


losses_dict = {
    "mse": MeanSquaredError,
    "binary_hinge": BinaryHinge,
    "bce": BinaryCrossEntropy,
    "binary_cross_entropy": BinaryCrossEntropy,
    "ce": OneHotCrossEntropy,
    "one_hot_cross_entropy": OneHotCrossEntropy,
    "hinge": OneHotHinge,
    "one_hot_hinge": OneHotHinge
}
