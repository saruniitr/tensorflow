# Anomaly Detection Exercise using Tensorflow

In this notebook we will implement Anomaly Detection and apply it to detect failing servers on a network. We first compute Multivariate Gaussian distribution on training set and validation set. With the Validation set labels and Validation probability distribution we determine best Epsilon (threshold) and F1 score. Once we calculate these values these are applied to training set to determine the Outliers (anomalies).

## Dataset

There are two sets of datasets examined.

Dataset1 is a simpler dataset with only two features which is helpful for data visualization. This is a test data.

* Visualization of data

![Anomaly detection data](https://github.com/saruniitr/tensorflow/blob/master/machine_learning_stanford/anomaly_detection/dataset1.png)

* Dataset with Anomalies marked

![Dataset with Anomalies marked](https://github.com/saruniitr/tensorflow/blob/master/machine_learning_stanford/anomaly_detection/dataset1_with_anomalies.png)

* Metrics
<br>No. of features: 2
<br>No. of Examples in Training set: 307
<br>No. of Examples in Cross Validation set: 307
<br>Best Epsilon: 8.990853e-05, F1 Score: 0.8750
<br>No. of Outliers: 6
<br>
Dataset2 is a more realistic dataset with more features. In this case we only determine the outliers as we cannot plot higher dimensions.

* Metrics 
<br>No. of features: 11
<br>No. of Examples in Training set: 1000
<br>No. of Examples in Cross Validation set: 100
<br>Best Epsilon: 1.377229e-18, F1 Score: 0.6154
<br>No. of Outliers: 117
