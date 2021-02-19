# Default of Credit Card Prediction

**Project Contributor**:  Steven Yan

<img src="images/credit_card.jpeg">

## Business Problem:

A credit card issuer based in the United States has forayed into the Asian market and wants to gain a better understanding of the customer base andcredit car their credit card habits.

In predict the likelihood of default for its customers, as well as identify the key drivers that determine this likelihood.

This would inform the issuer’s decisions on who to give a credit card to and what credit limit to provide.

It would also help the issuer have a better understanding of their current and potential customers, which would inform their future strategy, including their planning of offering targeted credit products to their customers.

The goal behind using this model is to achieve two things:

* Bring more consistency to the loaning process and;
* Investigate what the key drivers are behind a potential defaulter

## Business Proposal:

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

## Modeling Process:

- Train-Validate-Test Split: 70-20-10
- Vanilla Model: Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes, Linear Discriminant Analysis, K-Nearest Neighbors, Adaboost, Gradient Boosting, XGBoost
- Feature Engineering and Selection:  Random Forest, Decision Tree, Adaboost, RFECV
- Baseline Model:  Logistic Regression, Decision Tree, Random Forest, Adaboost, Gradient Boosting, XGBoost
- Hyperparameter Tuning:  Decision Tree, Random Forest, Adaboost, Gradient Boosting, XGBoost with GridSearchCV
- Class Imbalance: SMOTE, Tomek Links

## Analysis and Next Steps:


## Folder Structure:

```
├── /data
│    ├── *.pickle                  <- pickles for transfering data between workbooks
│    ├── *.csv                     <- initial and training-validation-testing datasets
├── /images                        (folder containing generated visualizations)
│    ├── *.png                     <- code-generated visualizations for EDA
├── /workbooks                     (folder containing workbooks for project)
│    ├── EDA_Data_Cleaning.ipynb   <- data cleaning, EDA, feature engineering workbook
│    ├── Initial_Modeling.ipynb    <- baseline model and feature selection workbook
│    └── More_Modeling.ipynb       <- hyperparameter tuning and class imbalance workbook
├── README.md                      <- top-level README for reviewers of this project

├── data                           <- dataset files
├── summary_presentation.pdf       <- a pdf of the project presentation
└── images                         <- both sourced externally and generated from code
```

## Questions:

Steven Yan—

- Email: stevenyan@uchicago.edu
- LinkedIn: <a href='https://www.linkedin.com/in/examsherpa/'>Steven Yan</a>

## Sources:
