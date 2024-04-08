import numpy as np
import neural.preprocessing
import neural.layers
import neural.losses
from neural.printer import Printer


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.loss = neural.losses.MeanSquaredError()
        self.optimizer = neural.optimizers.Adam()
        self.metrics = []

    def _get_metrics(self, y_hat, y):
        metrics_res = []
        for metric in self.metrics:
            cur_val = metric(y_hat, y)
            metrics_res.append((metric.name, cur_val))

        return metrics_res

    def _construct_metrics_history(self, history=True):
        if history:
            metrics_hist = {
                "loss": []
            }
        else:
            metrics_hist = {
                "loss": 0.0
            }

        for metric in self.metrics:
            name = metric.name
            if history:
                metrics_hist[name] = []
            else:
                metrics_hist[name] = 0.0

        return metrics_hist

    def set(self, loss="mse", optimizer="adam", metrics: list = None):
        if type(loss) == str:
            self.loss = neural.losses.losses_dict[loss]()
        else:
            self.loss = loss

        if type(optimizer) == str:
            self.optimizer = neural.optimizers.optimizers_dict[optimizer]()
        else:
            self.optimizer = optimizer

        if metrics is not None:
            self.metrics = []
            for metric in metrics:
                if type(metric) == str:
                    self.metrics.append(neural.metrics.metrics_dict[metric]())
                else:
                    self.metrics.append(metric)

    def get_num_grad(self, x, y, epsilon=1e-7):
        param_dict = {
            "dw": [],
            "db": []
        }

        for layer in self.layers:
            if layer.get_base() == neural.layers.StdLayer:
                w_shape = layer.kernel.shape
                b_shape = layer.bias.shape

                dw = np.zeros(shape=np.prod(w_shape))
                db = np.zeros(shape=(np.prod(b_shape)))

                for i in range(np.prod(w_shape)):
                    layer.kernel = layer.kernel.reshape(-1)
                    layer.kernel[i] += epsilon
                    layer.kernel = layer.kernel.reshape(w_shape)
                    y_hat = self.predict_batch(x)
                    loss_pos = self.loss(y_hat, y)

                    layer.kernel = layer.kernel.reshape(-1)
                    layer.kernel[i] -= epsilon
                    layer.kernel[i] -= epsilon
                    layer.kernel = layer.kernel.reshape(w_shape)
                    y_hat = self.predict_batch(x)
                    loss_neg = self.loss(y_hat, y)

                    layer.kernel = layer.kernel.reshape(-1)
                    layer.kernel[i] += epsilon
                    layer.kernel = layer.kernel.reshape(w_shape)

                    der = (loss_pos - loss_neg) / (2 * epsilon)
                    dw[i] = der

                for i in range(np.prod(b_shape)):
                    layer.bias = layer.bias.reshape(-1)
                    layer.bias[i] += epsilon
                    layer.bias = layer.bias.reshape(b_shape)
                    y_hat = self.predict_batch(x)
                    loss_pos = self.loss(y_hat, y)

                    layer.bias = layer.bias.reshape(-1)
                    layer.bias[i] -= epsilon
                    layer.bias[i] -= epsilon
                    layer.bias = layer.bias.reshape(b_shape)
                    y_hat = self.predict_batch(x)
                    loss_neg = self.loss(y_hat, y)

                    layer.bias = layer.bias.reshape(-1)
                    layer.bias[i] += epsilon
                    layer.bias = layer.bias.reshape(b_shape)

                    der = (loss_pos - loss_neg) / (2 * epsilon)
                    db[i] = der

                dw = dw.reshape(w_shape)
                db = db.reshape(b_shape)

                param_dict["dw"].append(dw)
                param_dict["db"].append(db)
            else:
                pass

        return param_dict

    def back_prop(self, x, y):
        params_dict = {
            "dw": [],
            "db": []
        }

        # Forward Propagation
        y_hat = x
        for layer in self.layers:
            y_hat = layer(y_hat, train=True)

        loss = self.loss(y_hat, y)

        # Back Propagation
        last_grad = self.loss.gradient()
        layers_reversed = self.layers[::-1]
        for layer in layers_reversed:

            if layer.get_base() == neural.layers.StdLayer:
                dw, db, last_grad = layer.gradient(last_grad)

                params_dict["dw"].append(dw)
                params_dict["db"].append(db)

            elif layer.get_base() == neural.layers.NoWeightsLayer:
                last_grad = layer.gradient(last_grad)
            else:
                raise RuntimeError(f"Layer Base {layer.get_base()} Not Following Standard")

        return params_dict

    def train_batch(self, x, y):
        params_dict = {
            "weights": [],
            "der": []
        }

        # Forward Propagation
        y_hat = x
        for layer in self.layers:
            y_hat = layer(y_hat, train=True)

        loss = self.loss(y_hat, y)

        # Back Propagation
        last_grad = self.loss.gradient()
        layers_reversed = self.layers[::-1]
        num_of_layers = len(layers_reversed)
        for layer_ind, layer in enumerate(layers_reversed):

            if layer.get_base() == neural.layers.StdLayer:
                if layer_ind < num_of_layers - 1:
                    dw, db, last_grad = layer.gradient(last_grad, last=False)
                else:
                    dw, db = layer.gradient(last_grad, last=True)

                w = layer.kernel
                b = layer.bias

                params_dict["weights"].append(w)
                params_dict["weights"].append(b)

                params_dict["der"].append(dw)
                params_dict["der"].append(db)

            elif layer.get_base() == neural.layers.NoWeightsLayer:
                if layer_ind < num_of_layers - 1:
                    last_grad = layer.gradient(last_grad)
                else:
                    pass  # Do nothing
            else:
                raise RuntimeError(f"Layer Base {layer.get_base()} Not Following Standard")

        weights = params_dict["weights"]
        der = params_dict["der"]

        self.optimizer.gradient_step(weights, der)

        return y_hat, loss

    def train(self, x, y, epochs=1, batch_size=32, verbose=True):
        metrics_hist = self._construct_metrics_history()

        printer = None
        if verbose:
            printer = Printer((x.shape[0] // batch_size) + 1, show_epochs=True, max_epoch=epochs)

        x = np.array(x)
        y = np.array(y)

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=-1)

        for epoch in range(epochs):
            mini_batch_gen = neural.preprocessing.mini_batch_generator(x, y, batch_size=batch_size, shuffle=True)

            metrics = None
            avg_metrics = np.zeros(shape=len(self.metrics))
            avg_name_metrics = []

            m_trained = 0
            avg_loss = 0

            for (x_cur, y_cur) in mini_batch_gen:
                y_hat, loss = self.train_batch(x_cur, y_cur)

                avg_loss = (avg_loss * m_trained + loss * x_cur.shape[0]) / (m_trained + x_cur.shape[0])

                metrics = self._get_metrics(y_hat, y_cur)
                for ind, (_, val) in enumerate(metrics):
                    avg_metrics[ind] = (avg_metrics[ind] * m_trained + val * x_cur.shape[0]) / (m_trained + x_cur.shape[0])

                avg_name_metrics = [(name, avg_metrics[ind]) for ind, (name, _) in enumerate(metrics)]

                m_trained += x_cur.shape[0]

                if verbose:
                    printer.update(loss=avg_loss, metrics=avg_name_metrics)

            if verbose:
                printer.finish(loss=avg_loss, metrics=avg_name_metrics)

            metrics_hist["loss"].append(avg_loss)

            if metrics is not None:
                metrics_hist_lst = list(metrics_hist.items())[1:]
                for (ind, (metric_name, _)) in enumerate(metrics_hist_lst):
                    metrics_hist[metric_name].append(avg_metrics[ind])

        return metrics_hist

    def predict_batch(self, inputs):
        y_hat = inputs

        for layer in self.layers:
            y_hat = layer(y_hat, train=False)

        return y_hat

    def predict(self, inputs, batch_size=32, verbose=True):
        inputs = np.array(inputs)
        data_gen = neural.preprocessing.data_generator(inputs, batch_size=batch_size)
        predictions = None
        printer = None

        if verbose:
            printer = Printer((inputs.shape[0] // batch_size) + 1)

        for data in data_gen:

            last_out = self.predict_batch(data)

            if predictions is None:
                predictions = last_out
            else:
                predictions = np.concatenate((predictions, last_out), axis=0)

            if verbose:
                printer.update()

        if verbose:
            printer.finish()

        return predictions

    def evaluate(self, x, y, batch_size=32, verbose=True):
        metrics_hist = self._construct_metrics_history(history=False)
        x = np.array(x)
        y = np.array(y)
        m = x.shape[0]
        data_gen = neural.preprocessing.mini_batch_generator(x, y, batch_size=batch_size, shuffle=False)
        printer = None

        if verbose:
            printer = Printer((x.shape[0] // batch_size) + 1)

        for data, labels in data_gen:
            cur_m = data.shape[0]

            y_hat = self.predict_batch(data)
            metrics_hist["loss"] += (self.loss(y_hat, labels) * cur_m)

            metrics = self._get_metrics(y_hat, labels)
            for (metric_name, metric_val) in metrics:
                metrics_hist[metric_name] += (metric_val * cur_m)

            if verbose:
                printer.update()

        if verbose:
            printer.finish()

        for metric_name in metrics_hist:
            metrics_hist[metric_name] /= m

        return metrics_hist
