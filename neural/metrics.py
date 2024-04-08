import numpy as np


class Metric:
    def __init__(self):
        pass

    def __call__(self, y_predict, y_true):
        pass


class BinaryAccuracy(Metric):
    def __init__(self, name="binary_acc", threshold=0.5):
        super(BinaryAccuracy, self).__init__()
        self.name = name
        self.thresh = threshold

    def __call__(self, y_predict, y_true):
        y_predict = np.array(y_predict)
        y_true = np.array(y_true)

        y_predict = y_predict.reshape(-1)
        y_true = y_true.reshape(-1)

        m = y_predict.shape[0]

        y_predict = np.where(y_predict >= 0.5, 1, 0)

        acc_mask = np.where(y_predict == y_true, 1, 0)

        acc = np.sum(acc_mask) / m

        return acc


class OneHotAccuracy(Metric):
    def __init__(self, name="one_hot_acc"):
        super(OneHotAccuracy, self).__init__()
        self.name = name

    def __call__(self, y_predict, y_true):
        y_predict = np.array(y_predict)
        y_true = np.array(y_true)

        m = y_predict.shape[0]

        y_predict = np.argmax(y_predict, axis=-1)
        y_true = np.argmax(y_true, axis=-1)

        acc_mask = np.where(y_predict == y_true, 1, 0)

        acc = np.sum(acc_mask) / m

        return acc


class MeanAbsoluteError(Metric):
    def __init__(self, name="mae"):
        super(MeanAbsoluteError, self).__init__()
        self.name = name

    def __call__(self, y_predict, y_true):
        y_predict = np.array(y_predict)
        y_true = np.array(y_true)

        y_predict = y_predict.reshape(-1)
        y_true = y_true.reshape(-1)

        m = y_predict.shape[0]

        mae = (1 / m) * np.sum(np.abs(y_predict - y_true))

        return mae


class RootMeanSquaredError(Metric):
    def __init__(self, name="rmse"):
        super(RootMeanSquaredError, self).__init__()
        self.name = name

    def __call__(self, y_predict, y_true):
        y_predict = np.array(y_predict)
        y_true = np.array(y_true)

        y_predict = y_predict.reshape(-1)
        y_true = y_true.reshape(-1)

        m = y_predict.shape[0]

        rmse = np.sqrt((1 / m) * np.sum(np.abs(y_predict - y_true)))

        return rmse


metrics_dict = {
    "binary_accuracy": BinaryAccuracy,
    "binary_acc": BinaryAccuracy,
    "one_hot_accuracy": OneHotAccuracy,
    "one_hot_acc": OneHotAccuracy,
    "mae": MeanAbsoluteError,
    "rmse": RootMeanSquaredError
}
