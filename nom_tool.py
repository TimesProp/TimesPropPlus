import numpy as np

class GlobalMinMaxScaler:
    def __init__(self):
        self.max_val = None

    def fit(self, array):
        array = np.copy(array)
        array[array < 0] = 0

        self.max_val = np.max(array)

        if self.max_val == 0:
            self.max_val = 1.0

        return self

    def transform(self, array):
        if self.max_val is None:
            raise ValueError("Must call fit before transform.")

        array = np.copy(array)
        array[array < 0] = 0

        return array / self.max_val

    def fit_transform(self, array):
        self.fit(array)
        return self.transform(array)

    def inverse_transform(self, array):
        if self.max_val is None:
            raise ValueError("Must call fit before inverse_transform.")

        return array * self.max_val