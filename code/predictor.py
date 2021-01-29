class Predictor:
    """An interface for predictor behaviour."""

    def define_and_fit(self):
        """
        Interface for defining and fitting a predictor.
        Must be implemented for each new predictor you want to use.

        :param file_path: string of the file path to the data.
        :return: a dictionary of params with keys as specified in Dataset()

        """
        raise NotImplementedError

    def get_top2_predictions(self, x):
        """
        Interface for getting the top 2 predictions.
        Must be implemented for each new predictor you want to use.
        
        :param x: the input value for which you want the predictions.
        :returns four floats: 
            first predicted class, corresponding probability
            secondary class, corresponding probability
        """
        raise NotImplementedError