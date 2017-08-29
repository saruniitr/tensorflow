# Credit Card Fraud Analysis using Logistic Regression

This program processes Credit card transaction data and classifies whether a
given transaction is fradulent or normal using Logistic Regression.

## Dataset
The dataset is a standard dataset taken from internet. It is available on Kaggle and can also be downloaded from various Machine learning blogs etc.

source: https://www.kaggle.com/dalpozz/creditcardfraud
The datasets contains transactions made by credit cards in September 2013 by
european cardholders. This dataset presents transactions that occurred in two
days, where we have 492 frauds out of 284,807 transactions. The dataset is
highly unbalanced, the positive class (frauds) account for 0.172% of all
transactions.

It contains only numerical input variables which are the result of a PCA
transformation. Unfortunately, due to confidentiality issues, we cannot
provide the original features and more background information about the
data. Features V1, V2, ... V28 are the principal components obtained with
PCA, the only features which have not been transformed with PCA are 'Time' and
'Amount'. Feature 'Time' contains the seconds elapsed between each transaction
and the first transaction in the dataset. The feature 'Amount' is the
transaction Amount, this feature can be used for example-dependant cost-
senstivelearning. Feature 'Class' is the response variable and it takes value
1 in case of fraud and 0 otherwise.

Citation: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
Bontempi. Calibrating Probability with Undersampling for Unbalanced
Classification. In Symposium on Computational Intelligence and Data Mining
(CIDM), IEEE, 2015

## Implementation
This is implemented using Tensorflow.

The logistic regression hypothesis is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).

In the current implementation the features are first normalized and split into Training (70%) and Validation set (30%). At the moment the Training error/Cost is minimized without Regularization. We use Tensorflow's Gradient descent optimizer to minimize the cost. Once the cost is minimized we get optimal parameters (theta) which are used to make Predictions on Validation set.

The cost function used is,

![Cost function](https://github.com/saruniitr/tensorflow/blob/master/machine_learning_stanford/fraud_detection/log_reg_cost_func.png)

## Performance
This dataset is *higly skewed*, meaning the difference between positive and negative examples is huge (only 0.172% of examples are fradulent transactions). In such cases prediction accuracy is not a good performance indicator; We have to calculate *Precision, Recall and F1 Score* which are reasonable performance indicators. The training error and other metrics on a particular run are,

* Training Error/Cost with respect to No. of Iterations
![Training Error](https://github.com/saruniitr/tensorflow/blob/master/machine_learning_stanford/fraud_detection/cost_vs_iterations.png)

| Performance Indicator   | Value     |
|------------------------:|:---------:|
| Training Accuracy       | 99.92%    |
| Precision               | 0.8511    |
| Recall                  | 0.5882    |
| F1 Score                | 0.6957    |

Due to the skewed nature of the dataset, eventhough training accuracy is high the F1 Score is not very high. It is also to be noted that these values may vary a bit because with each run the data is shuffled before splitting them into training and validation sets.
