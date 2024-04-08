class Printer:
    def __init__(self, total_counts, show_epochs=False, max_epoch=1):
        self.total_counts = total_counts
        self.show_epochs = show_epochs
        self.max_epoch = max_epoch
        self.cntr = 0

        self.cntr_str = f"{self.cntr}/{self.total_counts}: "
        self.main_str = "............................................................"
        self.main_str_len = len(self.main_str)
        self.epoch_num = 1

    def _update_cntr_str(self, finish=False):
        if finish:
            self.cntr_str = f"{self.total_counts}/{self.total_counts}: "
        else:
            cntr_len = len(str(self.cntr))
            counts_len = len(str(self.total_counts))

            cntr_str = " " * (counts_len - cntr_len)

            cntr_str += str(self.cntr)

            self.cntr_str = f"{cntr_str}/{self.total_counts}: "

    def _update_main_str(self, finish=False):
        if finish:
            self.main_str = "o" * self.main_str_len
        else:
            frac = self.cntr / self.total_counts
            done = int(frac * self.main_str_len)

            if done > self.main_str_len:
                self.main_str = "o" * self.main_str_len
            else:
                self.main_str = "o" * done + self.main_str[done:]

    def _get_print_str(self, loss=None, metrics: list or tuple = None):
        print_str = ""

        if self.show_epochs:
            max_epoch_len = len(str(self.max_epoch))
            epoch_num_len = len(str(self.epoch_num))

            wide_space = " " * (max_epoch_len - epoch_num_len)

            print_str += f"Epoch {self.epoch_num}/{self.max_epoch}: " + wide_space

        print_str += self.cntr_str + self.main_str

        if loss is not None:
            print_str += "\tloss: {:.5f}".format(loss)

        if metrics is not None:
            for (name, val) in metrics:
                print_str += f"\t{name}: " + "{:.5f}".format(val)

        return print_str

    def update(self, loss=None, metrics: list or tuple = None):
        self.cntr += 1
        self._update_cntr_str()
        self._update_main_str()

        print_str = self._get_print_str(loss, metrics)

        print("\r"+print_str, end="", flush=True)

    def finish(self, loss=None, metrics: list or tuple = None):
        self._update_cntr_str(finish=True)
        self._update_main_str(finish=True)

        print_str = self._get_print_str(loss, metrics)

        print("\r"+print_str, end="\n", flush=True)

        self.cntr = 0
        self.main_str = "." * self.main_str_len
        self.epoch_num += 1
