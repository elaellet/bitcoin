# Base class that other forecasters could inherit from.
class BaseForecaster:
    '''An abstract base class for all forecasting models.'''
    def __init__(self, train_ds, valid_ds, target_col):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.target_col = target_col
    
    def fit(self):
        raise NotImplementedError('The `fit` method must be implemented by a subclass.')

    def evaluate(self):
        raise NotImplementedError('The `evaluate` method must be implemented by a subclass.')
    
    def predict(self):
        raise NotImplementedError('The `predict` method must be implemented by a subclass.')