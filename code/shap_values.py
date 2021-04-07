import shap

from data import Dataset
from predictor import Predictor

class ShapGenerator:
    """A class for generating SHAP feature importance values."""

    def __init__(self, dataset, predictor, k):
        """Init method

        :param dataset: Dataset instance containing all data information.
        :param predictor: Predictor instance wrapping all predictor functionality.
        """

        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise ValueError("should provide data as Dataset instance")

        if isinstance(predictor, Predictor):
            self.predictor = predictor
        else:
            raise ValueError("should provide predictor as Predictor instance")
        
        self.explainer = self.generate_shap_explainer(k)

    def generate_shap_explainer(self, k):
        """Generate the SHAP explainer object

        :param k: int representing the number of clusters.

        Note that the choice of k influences the speed with which the computation is done quite
        dramatically. If no k is chosen, the entire training dataset will be used. SHAP docs often
        use about 100 samples.
        """

        # Decide which part of the training data to take for the explaination object.
        if k is not None:
            train_data_expl = shap.kmeans(self.dataset.train_data[self.dataset.feature_names], k)
        else:
            train_data_expl = self.dataset.train_data[self.dataset.feature_names]
        
        # Using KernelExplainer to stay model-agnostic. 
        # TODO: for classification, change get_prediction to get_prediction_proba
        return shap.KernelExplainer(self.predictor.get_prediction, train_data_expl)

    def get_shap_values(self, data_point):
        """Calculate the SHAP values for the current data point"""
        # shap_values = self.explainer.shap_values(data_point[self.dataset.feature_names])
        # f=shap.force_plot(self.explainer.expected_value, shap_values, data_point[self.dataset.feature_names], show=False)
        # shap.save_html("index.html", f)
        return self.explainer.shap_values(data_point[self.dataset.feature_names])

    def get_shap_explanation(self, data_point):
        """
        Create the SHAP explanation object required for new plots
        see: https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html
        You can still use shap.plots._waterfall.waterfall_legacy() without this object, but not
        shap.plots.waterfall(), shap.plots.bar(), or shap.plots._bar_bar_legacy(). 

        """
        
        shap_values = self.explainer.shap_values(data_point[self.dataset.feature_names])

        shap_object = shap.Explanation(values=shap_values,
                                       base_values=self.explainer.expected_value,
                                       feature_names = data_point.index,
                                       data = data_point[self.dataset.feature_names].values)
        return shap_object

############################ Save plot as html ############################

# shap.initjs()
# f=shap.force_plot(explainer.expected_value, shap_values, 
#                   data_point[dataset_merged.feature_names], show=False)
# shap.save_html("index.html", f)