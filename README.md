# Building a Model for Credit Card Default Prediction

**Project Contributor**:  Steven Yan

**Project Supervisors**: Fangfang Lee, Joshua...


<img src="images/credit_card.jpeg">

## Background and Business Problem:

A credit card issuer based in the United States has forayed into the Asian market and wants to gain a better understanding of the customer base andcredit car their credit card habits.

In predict the likelihood of default for its customers, as well as identify the key drivers that determine this likelihood.

This would inform the issuer’s decisions on who to give a credit card to and what credit limit to provide.

It would also help the issuer have a better understanding of their current and potential customers, which would inform their future strategy, including their planning of offering targeted credit products to their customers.

The goal behind using this model is to achieve two things:

* Bring more consistency to the loaning process and;
* Investigate what the key drivers are behind a potential defaulter


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


## Analysis:

There was not a significant difference in the vanilla model, model with all the engineered features, and model after using feature selection methods.  The initial models were selected for the highest accuracy and PR AUC score.

Some of the engineered features created seemed to have a stronger correlation than the original variables.  I have to check for collinearity as some of the variables would overlap in context.  I am surprised that the demographic features does not have a greater correlation with default.  It would seem useful for companies to be able to identify certain demographic groups that are more prone to defaulting.

The metric I used was the PR AUC score, but with an eye to increasing accuracy and PR AUC score, which is the scoring parameters I used in GridSearchCV for hyperparameter tuning. Hyperparameter tuning improved accuracy to 82% from a baseline of 77%, and the highest PR AUC score at around 54%. My initial analysis of implementing class imbalance methods is that it substantially increases the PR AUC score to almost 90%, but accuracy tops out at 82% on the validation set.



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

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. [link](https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf)
