class Predictor:
    """An interface for all expected predictor behaviour."""

    def define_and_fit(self, model_file):
        """
        Interface for defining and fitting a predictor.
        Must be implemented for each new predictor you want to use.

        :param model_file: string of the file path where the model should be saved.
        :returns: the fitted model

        """
        raise NotImplementedError

    def set_predictions(self):
        """
        Interface for setting 3 class variables:
        train_pred: list of output values of the training data
        correct_pred: DataFrame of a subset of the training data that was classified correctly.
        incorrect_pred: DataFrame of a subset of the training data that was classified incorrectly.

        """
        raise NotImplementedError

    def get_prediction(self, x):
        """
        Interface for getting the prediction of the model on instance(s) x.
        
        :param x: Series/array of input value(s) for which you want the prediction
        :returns: the raw prediction(s) 
       
        """
        raise NotImplementedError

    def get_prediction_proba(self, x):
        """
        Only for classification models:
        Interface for getting the prediction prob of the model on instance(s) x.
        
        :param x: Series/array of input value(s) for which you want the prediction
        :returns: the raw prediction(s) 
       
        """
        raise NotImplementedError

    def get_second_prediction(self, x):
        """
        Interface for getting the secondary prediction of the model on the instance x.
        
        :param x: Series/array of input value(s) for which you want the prediction
        :returns: the secondary prediction as an int

        """
        raise NotImplementedError

    def get_data_corr_predicted_as(self, noc):
        """
        Interface for getting all training data correctly predicted as a certain NOC.
        
        :param noc: int representing the groun truth NOC value you want the data from.
        :returns: DataFrame of all training data correctly predicted as noc.

        """
        raise NotImplementedError