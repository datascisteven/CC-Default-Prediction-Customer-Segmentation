# Building a Model for Credit Card Default Prediction

**Project Contributor**:  Steven Yan

**Project Supervisors**: Fangfang Lee, Joshua (need last name)

<img src="images/credit_card.jpeg">

## Background and Business Problem:

A credit card issuer based in the United States has forayed into the Asian market and wants to gain a better understanding of the customer base andcredit car their credit card habits.

In predict the likelihood of default for its customers, as well as identify the key drivers that determine this likelihood.

This would inform the issuer’s decisions on who to give a credit card to and what credit limit to provide.

It would also help the issuer have a better understanding of their current and potential customers, which would inform their future strategy, including their planning of offering targeted credit products to their customers.

The goal behind using this model is to achieve two things:

* Bring more consistency to the loaning process and;
* Investigate what the key drivers are behind a potential defaulter

(need to edit and clean up)

## Data Collection:

The dataset considered in this analysis is the *“Default of Credit Card Clients”* dataset released under the public license of *Creative Commons* and available on the [Kaggle website](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset).

This dataset contains **30000 observations of 25 variables** from a bank (and also a cash and credit card issuer in Taiwan), where each observation corresponds to a particular credit card client. Among the total 30000 observations, 6636 observations (22.1%) are cardholders with default payment.

The 25 variables in this dataset comprises of:

- demographic variables (gender, education level, marriage status, and age)
- financial variables of 6-months worth of payment data from April 2005 to September 2005
  - amount of given credit
  - monthly repayment statuses
  - monthly amount of bill statements,
  - monthly amount of previous payments)


## Procedure:

- Data Cleaning:

  - Check for null values
  - Check for anomalous values and outliers
- EDA:

  - Univariate Analysis:  Categorical and Continuous Features, boxplots, bar graphs, distribution plots
  - Bivariate Analysis: pairplots, stacked bar plots,
- Train-Validate-Test Split:

  - Training Set: 70
  - Validation Set: 20
  - Testing or Holdout Set: 10
- Building a Vanilla Model:

  - Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes, Linear Discriminant Analysis, K-Nearest Neighbors, Adaboost, Gradient Boosting, XGBoost
- Feature Engineering:

  - Using domain knowledge to develop different features that may or may not impact the predictive ability of the model
- Feature Selection:

  - Random Forest Feature Importance
  - Decision Tree Feature Importance
  - XGBoost Feature Importance
  - Recursive Feature Elimination with Cross Validation
- Develop Baseline Models:

  - Logistic Regression
  - Random Forest Classifier
  - Adaboost Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
- Hyperparameter Tuning:

  - Using GridSearchCV
- Tuned Models:

  - Logistic Regression, Random Forest, Adaboost, Gradient Boosting, XGBoost Classifiers
- Class Imbalance Methods:

  - Ensemble:  BaggingClassifier, BalancedBaggingClassifier
  - Undersampling: Tomek Links, ENN
  - Oversampling: SMOTE, ADASYN
  - Hybridized: SMOTE-ENN, SMOTE-Tomek


## Exploratory Data Analysis:

<img src="images/baseline.png">

There is a clear class imbalance that will need to be addressed in creating the best predictive model for customers likely to default on their next payment.

<img src="images/pairplot2.png">

The difference in distributions as exhibited in the diagonal plots indicates that `behind1` - `behind6` has correlation with `default`.

<img src="images/pairplot1.png">

There is not the same clear difference in distribution between the different `gender`, `education`, `marriage`categories, which would indicate to me that they would have less of a correlation with `default`.




## Next Steps:

- Exploration and analysis of Class Imbalance Methods
- Use a different normalization technique (i.e. MinMaxScaler)
- Incorporate datasets from different countries
- Customer segmentation: implementation of unsupervised learning algorithms on datasets
- Try additional ensemble methods on dataset:  BrownBoost, Catboost, LightGBM
- Try unsupervised learning algorithms:   PCA, K-Means, Neural Networks


## Folder Structure:

```
├── /data                          (folder containing data files)
│    ├── /pickles                  <- pickles for transfering data between project workbooks
│    ├── /charts                   <- evaluation metrics charts
├── /images                        (folder containing generated visualizations)
│    ├── *.png                     <- code-generated visualizations for EDA
├── /workbooks                     (folder containing workbooks for project)
│    ├── EDA_Notebook.ipynb        <- data cleaning, EDA, feature engineering workbook
│    ├── Modeling_Notebook.ipynb   <- baseline model and feature selection workbook
│    ├── Modeling_2_Notebook.ipynb <- hyperparameter tuning and class imbalance workbook
│    └── Holdout_Notebook.ipynb    <- holdout set workbook
├── README.md                      <- top-level README for reviewers of this project
├── Final_CC_Default.ipynb.        <- final notebook summarizing the entire project
└── presentation.pdf               <- pdf of the project presentation

```

## Questions:

If you have questions about the project or, please feel free to connect with me at my email:

- Email: **<a href='mailto@stevenyan@uchicago.edu'>stevenyan@uchicago.edu</a>**

If you are interested in connecting for networking, please feel free to connect with me via LinkedIn:

- My Profile: **<a href='https://www.linkedin.com/in/examsherpa/'>Steven Yan</a>**


## References:

Default of Credit Card Clients Dataset on Kaggle: [https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)

UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. [PDF link](https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf)

Kovács, G. (2019). An empirical comparison and evaluation of minority oversampling techniques on a large number of imbalanced datasets. *Applied Soft Computing*, *83*, 105662. [PDF link](https://www.sciencedirect.com/science/article/pii/S1568494619304429)

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PloS one*, *10*(3), e0118432. [HTML link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
