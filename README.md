# Default of Credit Card Prediction

**Project Contributor**:  Steven Yan

<img src="images/credit_card.jpeg">

## Business Problem and Proposal:

A credit card issuer based in the United States has forayed into the Asian market and wants to gain a better understanding of the customer base andcredit car their credit card habits.

In developing a model for predicting the likelihood of default for the customer base, as well as identifying the key drivers that determine this likelihood, our team would be able to inform the issuer’s decisions on whom to give a credit card to and what credit limit to provide.

We also aim to provide our client a better understanding of its current and potential customer base, which would help inform its future strategy, which includes the offering of targeted credit products to its customers.

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
- Vanilla Model:  Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes, Linear Discriminant Analysis, K-Nearest Neighbors, Adaboost, Gradient Boosting, XGBoost
- Feature Engineering and Selection:  Random Forest, Decision Tree, and Adaboost Feature Importance, Recursive Feature Elimination
- Baseline Model:  Logistic Regression, Decision Tree, Random Forest, Adaboost, Gradient Boosting, XGBoost
- Hyperparameter Tuning:  Decision Tree, Random Forest, Adaboost, Gradient Boosting, and XGBoost Classifiers with GridSearchCV
- Class Imbalance: SMOTE, Tomek Links

## Analysis and Next Steps:

* Explore further into class imbalance methods
* Adjust threshold for optimal precision and recall in PR curve

## Folder Structure:

```
├── /data                          (folder containing data files)
│    ├── *.pickle                  <- pickles for transfering data between project workbooks
│    ├── *.csv                     <- initial spreadsheet and training-validation-testing datasets
├── /images                        (folder containing generated visualizations)
│    ├── *.png                     <- code-generated visualizations for EDA
├── /workbooks                     (folder containing workbooks for project)
│    ├── EDA_Data_Cleaning.ipynb   <- data cleaning, EDA, feature engineering workbook
│    ├── Initial_Modeling.ipynb    <- baseline model and feature selection workbook
│    └── More_Modeling.ipynb       <- hyperparameter tuning and class imbalance workbook
├── README.md                      <- top-level README for reviewers of this project
└── presentation.pdf               <- pdf of the project presentation

```

## Questions:

If you have questions about the project or, please feel free to connect with me at my email:

- Email: **<a href='mailto@stevenyan@uchicago.edu'>stevenyan@uchicago.edu</a>**

If you are interested in connecting for networking, please feel free to connect with me via LinkedIn:

- My Profile: **<a href='https://www.linkedin.com/in/examsherpa/'>Steven Yan</a>**

## Sources:
