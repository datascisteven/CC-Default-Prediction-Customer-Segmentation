# Building a Model for Credit Card Default Prediction

**Project Contributor**:  Steven Yan

**Project Supervisors**: Fangfang Lee, Joshua Szymanowski

<img src="images/credit_card.jpeg">

# Background and Business Problem:

A credit card issuer based in the United States has forayed into the Asian market and wants to gain a better understanding of the customer base and its credit card habits. Building a supervised machine learning model for predicting the likelihood of default, as well as identifying the key factors that determine that likelihood would inform the issuer’s decision-making process on whom to give a credit card to and what credit limit to provide.

Many statistical methods in the past have been used to develop models of risk prediction and with the evolution of AI and machine learning to forecast credit risk.  From the perspective of risk control, predicting the probability of defaulting is more meaningful, pertinent, and tangible for practitioners.

<<<<<<< HEAD
Default occurs when a credit card customer fails to pay a calculated minimum monthly amount, comprising of interest and some principal amount. High default has been a major problem in the credit card market and has been growing in recent years despite the strength of the U.S. economy. Clearly credit card default is a complex phenomenon involving many factors beyond the scope of the present research. The variables which we have examined here capture some key behaviors and provide the issuer a better understanding of current and potential customers, specificially which would inform their strategy in the new market.
=======
High default has been a major problem in the credit card market and has been growing in recent years despite the strength of the U.S. economy. Clearly credit card default is a complex phenomenon involving many factors beyond the scope of the present research. The variables which we have examined here capture some key behaviors and provide the issuer a better understanding of current and potential customers, specificially which would inform their strategy in the new market.
>>>>>>> 40cd93e831ebb862574eaa467e34948626d5ad00

The goal behind using this model is to achieve two things:  to bring more consistency to the loaning process and investigate what key factors are behind a potential defaulter.


# Data Sources:

The dataset considered in this analysis is the *“Default of Credit Card Clients”* dataset released under the public license of *Creative Commons* and available on the [Kaggle website](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset).

<<<<<<< HEAD
This dataset contains  1 response variable, 23 explanatory variables, and 30000 case data from a bank and credit card issuer in Taiwan, where each observation corresponds to a particular credit card client. Among the total 30000 observations, 6636 observations (22.1%) are cardholders with default payment.
=======
This dataset contains  1 response variable, 23 explanatory variables, and 30000 case data. observations of 25 variables** from a bank (and also a cash and credit card issuer in Taiwan), where each observation corresponds to a particular credit card client. Among the total 30000 observations, 6636 observations (22.1%) are cardholders with default payment.
>>>>>>> 40cd93e831ebb862574eaa467e34948626d5ad00

The 25 variables in this dataset comprises of:

- demographic variables (gender, education level, marriage status, and age)
- financial variables of 6-months worth of payment data from April 2005 to September 2005
  - amount of given credit
  - monthly repayment statuses
  - monthly amount of bill statements,
  - monthly amount of previous payments)


<<<<<<< HEAD
## Data Understanding:
=======



## Exploratory Data Analysis:
>>>>>>> 40cd93e831ebb862574eaa467e34948626d5ad00

<img src="images/baseline.png">

There is a clear class imbalance that will need to be addressed in creating the best predictive model for customers likely to default on their next payment.

<img src="images/pairplot2.png">

The difference in distributions as exhibited in the diagonal plots indicates that `behind1` - `behind6` has correlation with `default`.

<img src="images/pairplot1.png">

There is not the same clear difference in distribution between the different `gender`, `education`, `marriage`categories, which would indicate to me that they would have less of a correlation with `default`.




## Next Steps:

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
│    ├── Class_Imbalance.ipynb     <- workbook for exploring different class imbalance methods
│    ├── Modeling_Notebook.ipynb   <- baseline model and feature selection workbook
│    ├── Modeling_2_Notebook.ipynb <- hyperparameter tuning and class imbalance workbook
│    ├── Holdout_Notebook.ipynb    <- holdout set workbook
│    └── utils.py                  <- file containing self-generated functions
├── README.md                      <- top-level README for reviewers of this project
├── Final_CC_Default.ipynb         <- final notebook summarizing the entire project
├── presentation.pdf               <- pdf of the project presentation
└── utils.py                       <- file containing self-generated functions

```

## Questions:

If you have questions about the project or, please feel free to connect with me at my email:

- Email: **<a href='mailto@stevenyan@uchicago.edu'>stevenyan@uchicago.edu</a>**

If you are interested in connecting for networking, please feel free to connect with me via LinkedIn:

- My Profile: **<a href='https://www.linkedin.com/in/datascisteven/'>Steven Yan</a>**


## References:

Default of Credit Card Clients Dataset on Kaggle: [https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)

UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. [PDF link](https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf)

Kovács, G. (2019). An empirical comparison and evaluation of minority oversampling techniques on a large number of imbalanced datasets. *Applied Soft Computing*, *83*, 105662. [PDF link](https://www.sciencedirect.com/science/article/pii/S1568494619304429)

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PloS one*, *10*(3), e0118432. [HTML link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
