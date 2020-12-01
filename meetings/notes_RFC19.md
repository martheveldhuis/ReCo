## RFC19 model

Research was done to create a ML model evaluated on 350 training profiles, 120 validation ("test") profiles, and 120 test ("hold-out") profiles.

### Feature engineering
First, per profile there were 278 features:  
25  samples features + (11 locus features * 23 autosomal loci)

Then, those were filtered down based on correlation with the outcome to 50:
> "features were selected that had the highest partial correlation with the NOC, controlling for the effect of previously selected features. This partial correlation is a measure of how much two variables are correlated while removing the effect of other variables."

### Machine learning
10 ML models were trained with increasing numbers of features, for which grid search was also performed to tune hyperparameters.


The best model was selected based on the accuracy of the validation set, which turned out to be a Random Forest classifier with 19 features.

> "Besides accuracy, precision and recall are measures of relevance and can be used to compare the performance of the various models. When generating precision-recall plots and comparing the ten models it becomes clear that **not one of the models outperformed all others as precision and recall differed per NOC**"

Interesting findings regarding features:
* Other markers that usually do not play at the foreground while doing MAC analysis showed up for certain cases.

### My main remarks

- *Can we improve the feature selection process?*
- *Can we improve the model selection process?*
- *Can we generate counterfactual explanations?*


Since the dataset is quite unbalanced, accuracy might not the best metric to determine the best model. For example, the most frequent class is where the NOC is 1 (twice as much data than for samples where the NOC is 5). However, we don't see any samples being classified as 1, while they should be 5, which would give misleading accuracy results. However, it might still be a good idea to analyze the scores **per class**, using precision and recall.

**Precision** provides an indication of inaccurate positive predictions (should be high if you want make sure that the samples you classify to this class are indeed from this class).  
**Recall** provides an indication of missed positive predictions (should be high if you want to make sure you get all the samples that belong to this class).

I think that in this case, both are equally important -> F1 score?

**F1 score** = (2 * Precision * Recall) / (Precision + Recall)


**Contrastive explanations** work well for humans, as we tend to ask "Why X and not Y?". In this case, I think contrastive explanations work particularly well since you get something like this:

>*Why did the model predict this sample to have 4 contributors instead of the
actual 5?*
- *Because the allele count at locus x has value y*
- *If the allele count at locus x was y higher, the prediction would have been 5*

I think this would be better than simple feature importance where you get something like:

> *How does the most make predictions?*
- *Feature x was most important, followed by feature y, and z*

Therefore techniques I am instantly thinking of are:
- Anchors "*This prediction will hold for this area of the feature space"*
- Individual SHAP values "*For this prediction, features x and y were most important as compared to the average feature use*"
- Various counterfactual explanation techniques (GRACE for NN with python, MOC for * with R)
