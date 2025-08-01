# Basee class that other forecasters could also inherit from.
class BaseForecaster:
    '''An abstract base class for all forecasting models.'''
    def __init__(self, df_train, df_valid, target_col):
        self.df_train = df_train
        self.df_valid = df_valid
        self.target_col = target_col
        self.model = None
        self.history = None
    
    def fit(self):
        raise NotImplementedError('The "fit" method must be implemented by a subclass.')

    def evaluate(self):
        raise NotImplementedError('The "evaluate" method must be implemented by a subclass.')
    
    def predict(self):
        raise NotImplementedError('The "predict" method must be implemented by a subclass.')