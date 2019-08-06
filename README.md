# predictive-maintenance

This example illustrates some machine learning approaches to predictive maintenance problems that are common in various businesses. This is a toy example to demonstrate some fundamental concepts of machine learning. The example highlights the importance of correct selection of data attributes before we attempt to build machine learning models. 


I demonstrate the utility of the following two important concepts that are ubiquitous in predictive maintenance problems:
1. **Curse of Dimensionality:** Selecting the useful data attributes is important especially when the raw data consists of multiple defining features (usually depicted as columns in a *csv* file). More number of features don't necessarily result in better models. On the contrary, we experience what is traditionally referred to as the *curse of dimensionality*. This simply means that as the number of features increase, we are abstracting the problem in a higher dimensional Euclidean space. This exponentially scales the amount of data required to provide coverage in order for the models to be meaningful.

I present a solution based on statistics and the [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) to identify a subset of features that are more informative than the entire given feature set. This vastly reduces the problem space and result in macine learning models which are more interpretable and have better performance. 

2. **Class Imbalance:** The problem of class imbalance stems from not having sufficient representative data points for a subset of classes in the given data. This is typical of problems related to predictive maintenance where the class of interest that we want to predict occurs infrequently. Hence, the data used for modelling can be skewed and could result in biased machine learning models. 

I present a solution based on the [Synthetic Minority Oversampling Technique (SMOTE)](https://pypi.org/project/imbalanced-learn/) algorithm. As the name suggests, the goal is to *oversample* from the empirical distribution of the minority class to generate a larger representative data set. This would minimize the bias in the training data and would give the machine learning models a wider set of patterns to *learn* from the minority class. A good reference for understanding the mechanics of SMOTE is illustrated [here](http://rikunert.com/SMOTE_explained).

