import pandas as pd

class MultiOutputModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return pd.DataFrame(predictions).T  

