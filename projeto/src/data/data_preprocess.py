import pandas as pd
from sklearn.pipeline import Pipeline

import structlog 

logger = structlog.get_logger()

class DataPreprocess:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe
        self.trained_pipe = None

    def train(self, dataframe):
        logger.info('Treinando o pipeline de preprocessamento...')
        self.trained_pipe = self.pipe.fit(dataframe)

    def transform(self, dataframe: pd.DataFrame):
        if self.trained_pipe is None:
            raise ValueError('O pipeline de preprocessamento não foi treinado...')
        logger.info('Transformando os dados')
        data_preprocessed = self.trained_pipe.transform(dataframe)
        return data_preprocessed
