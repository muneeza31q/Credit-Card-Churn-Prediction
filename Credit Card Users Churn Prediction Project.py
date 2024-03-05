#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# ## Background and Context
# The Thera bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged to every user irrespective of usage, while others are charged under specified circumstances.
# 
# Customers’ leaving credit cards services would lead bank to loss, so the bank wants to analyze the data of customers and identify the customers who will leave their credit card services and reason for same – so that bank could improve upon those areas
# 
# You as a Data scientist at Thera bank need to come up with a classification model that will help the bank improve its services so that customers do not renounce their credit cards
# 
# You need to identify the best possible model that will give the required performance
# 
# ## Objective
# 
# The objective is to explore and visualize the dataset, build a classification model to predict if the customer is going to churn or not, optimize the model using appropriate techniques, and generate a set of insights and recommendations that will help the bank.
# 
# ## Data Dictionary:
# 
# CLIENTNUM: Client number. Unique identifier for the customer holding the account
# 
# Attrition_Flag: Internal event (customer activity) variable - if the account is closed then "Attrited Customer" (1), else "Existing Customer" (0)
# 
# Customer_Age: Age in Years
# 
# Gender: Gender of the account holder
# 
# Dependent_count: Number of dependents
# 
# Education_Level:  Educational Qualification of the account holder - Graduate, High School, Unknown, Uneducated, College(refers to a college student), Post-Graduate, Doctorate.
# 
# Marital_Status: Marital Status of the account holder
# 
# Income_Category: Annual Income Category of the account holder
# 
# Card_Category: Type of Card
# 
# Months_on_book: Period of relationship with the bank
# 
# Total_Relationship_Count: Total no. of products held by the customer
# 
# Months_Inactive_12_mon: No. of months inactive in the last 12 months
# 
# Contacts_Count_12_mon: No. of Contacts between the customer and bank in the last 12 months
# 
# Credit_Limit: Credit Limit on the Credit Card
# 
# Total_Revolving_Bal: The balance that carries over from one month to the next is the revolving balance
# 
# Avg_Open_To_Buy: Open to Buy refers to the amount left on the credit card to use (Average of last 12 months)
# 
# Total_Trans_Amt: Total Transaction Amount (Last 12 months)
# 
# Total_Trans_Ct: Total Transaction Count (Last 12 months)
# 
# Total_Ct_Chng_Q4_Q1: Ratio of the total transaction count in 4th quarter and the total transaction count in 1st quarter
# 
# Total_Amt_Chng_Q4_Q1: Ratio of the total transaction amount in 4th quarter and the total transaction amount in 1st quarter
# 
# Avg_Utilization_Ratio: Represents how much of the available credit the customer spent
#  
# 
# ## Best Practices for Notebook : 
# 
# The notebook should be well-documented, with inline comments explaining the functionality of code and markdown cells containing comments on the observations and insights.
# The notebook should be run from start to finish sequentially before submission.
# It is preferable to remove all warnings and errors before submission.
#  
# 
# ## Submission Guidelines :
# 
# The submission should be: well commented Jupyter notebook [format - .HTML] - Please run the notebook sequentially before submitting.
# Any assignment found copied/ plagiarized with other groups will not be graded and awarded zero marks
# Please ensure timely submission as any submission post-deadline will not be accepted for evaluation
# Submission will not be evaluated if,
# it is submitted post-deadline, or,
# more than 1 files are submitted

# In[1]:


#pip install xgboost
get_ipython().system('pip install --upgrade scikit-learn')


# In[2]:


# To help with reading and manipulating data
import pandas as pd
import numpy as np

# To help with data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# To be used for missing value imputation
from sklearn.impute import SimpleImputer

# To help with model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier

# To get different metric scores, and split data
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    #plot_confusion_matrix
)

#from sklearn.metrics import plot_confusion_matrix
#from sklearn.utils.multiclass import plot_confusion_matrix

# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


# To be used for tuning the model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# To be used for creating pipelines and personalizing them
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# To define maximum number of columns to be displayed in a dataframe
pd.set_option("display.max_columns", None)

# To supress scientific notations for a dataframe
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# To supress warnings
import warnings

warnings.filterwarnings("ignore")

# This will help in making the Python code more structured automatically (good coding practice)
#%load_ext nb_black


# In[3]:


import sklearn
print(sklearn.__version__)


# ### Read the dataset

# In[4]:


df0 = pd.read_csv("BankChurners.csv")


# ### Data Overview

# In[5]:


# copying data to another varaible to avoid any changes to original data
df00 = df0.copy()


# In[6]:


df00.head()


# In[7]:


df00.tail()


# In[8]:


np.random.seed(1)
df00.sample(n=15)


# ###     Printing the information 

# In[9]:


df00.info()


# ####  Observations 
# * There are 10127 rows and 21 columns. 
# * All the columns have 10127 non-null values except Education_Level and Marital_Status column which indicates that there are null values in these columns and we need to treat these missing values.  

# ####  Let's check for duplicate values 

# In[10]:


df00.duplicated().sum()


# In[11]:


# create a new column to label Existing Customer= 1 and Attrited Customer=0
df00["Attrition_Flag"] = df00["Attrition_Flag"].apply(
    lambda x: 0 if x == "Existing Customer" else 1
)

# print the updated dataset
df00.head()


# In[12]:


df00.info()


# ####   Let's check for percentage of missing values 

# In[13]:


round(df00.isnull().sum() / df00.isnull().count() * 100, 2)


# ####  Observations 
# - Education_Level column has 15% missing values, and Marital_Status has 7.4% missing values. We will further explore to treat them. 

# Let's check the statistical summary for the data so that we can find an appropriate way to impute the missing values. 

# In[14]:


# let's view the statistical summary of the numerical columns in the data
df00.describe().T


# ####  Observations 
# 

# - The average customer age was 46 years old.
# - The range for Dependent_count is between 1-3.
# - The period of the relationship with the bank, on average, is 35.9 years.
# - The total number of products held by the customer is between 3-5 (by the 25%-75%).
# - At least 75% of customers were inactive for 3 months in the last 12 months.
# - At least 25% or more customers had 2 contacts between the customer and bank in the last 12 months.
# - The average credit limit was 8,632 dollars.
# - The average revolving balance was 1,163 dollars.
# - The average amount left on the credit card to use (Average of last 12 months) was 7,469 dollars.
# - Between 25%-75% of customers had 2,155.5-4,741 dollars of	total transaction amount (Last 12 months).
# 

# Let's explore the variables in depth to have a better understanding. 

# In[15]:


## Let's create a list of numerical and categorical columns
categorical_cols = [
    "Attrition_Flag",
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

numerical_cols = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]


# ###   Univariate EDA on Numerical variables  

# In[16]:


# creating histograms
df00[numerical_cols].hist(figsize=(14, 14))
plt.show()


# ####  Observations  
# * The **Customer Age** distribution is close to normal with majarity of the people's average age being 50. 
# * The distribution of **Dependent Count** is hard to tell, but the average is between 2-4.
# * **Months_on_Book** seems to have a normal distribution, and seems to be around 35 months of the relationship between the bank and the customers. 
# * **Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ration** tend to have a skewed distribution. On comparing the amount spend on each product, majority of the customers spend more amount on wines and meat products. 

# ###   Univariate EDA on Categorical variables  

# In[17]:


#categorical_cols = [    "Attrition_Flag",    "Gender",    "Education_Level",    "Marital_Status",    "Income_Category",    "Card_Category",]

fig, axes = plt.subplots(2, 3)
fig.suptitle("Count Plot for Categorical Features")
sns.set(rc={"figure.figsize": (15, 10)})

for i, col in enumerate(categorical_cols):
    row = i // 3
    col = i % 3
    sns.countplot(ax=axes[row, col], x=categorical_cols[i], data=df00)
    axes[row, col].set_title(categorical_cols[i])
    axes[row, col].tick_params(axis='x', labelrotation=45)

plt.show()


# ####  Observations 

# - Most of Thera Bank have existing customers with the following prominent factors: female, graduate, married, income less than 40K, and Blue card.
# 

# ###   Bivariate EDA  

# **We have analyzed different categorical and numerical variables.** 
# 
# **Let's check how different variables are related to each other.**

# ####   Correlation Plot 

# In[18]:


plt.figure(figsize=(15, 7))
sns.heatmap(df00.corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()


# ####  Observations 

# * **Attrition_Flag and Contacts_Count_12_mon** have the strongest positive correlation.
# * **Attrition_Flag and Total_Trans_Ct** have the strongest negative correlation.

# In[19]:


# function to plot stacked bar chart


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# In[20]:


stacked_barplot(df00, "Attrition_Flag", "Card_Category")


# * For both existing and attrited customers both had blue cards.

# In[21]:


stacked_barplot(df00, "Attrition_Flag", "Months_Inactive_12_mon")


# * It seems that customers have been inactive 2 or 3 months.

# **Let's check the relationship between response and Numerical variables**

# In[22]:


# Mean of numerical variables grouped by attrition
df00.groupby(["Attrition_Flag"])[numerical_cols].mean()


# ####  Observations 
# * People with high income are more likely to accept the offer in the campaign which is fair enough. 
# * Customers who spent more money on wines are more likely to take the offer.
# * Customers who are making more visits to the store and the website have high chance of taking the offer. 

# ###  Data Processing  

# - We can drop the column - `CLIENTNUM` as it is unique for each customer and will not add value to the model.

# In[23]:


# Dropping column - ID
df00.drop(columns=["CLIENTNUM"], inplace=True)


# **Creating a copy of data to build the model**

# In[24]:


df = df00.copy()


# **Separating target variable and other variables**

# In[25]:


X = df.drop(columns="Attrition_Flag")
X = pd.get_dummies(X)

Y = df["Attrition_Flag"]


# **Splitting the data into train/test**

# Note: We will split the data into train and test as data size is very small to create validation set as well. We will be using stratified sampling technique to ensure that relative class frequencies are approximately preserved in each train and validation fold.

# In[26]:


# Splitting data into training, validation and test set:
# first we split data into 2 parts, say temporary and test

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)

print(X_train.shape, X_test.shape)


# ### Missing-Value Treatment
# 
# * We will use median to impute missing values in Income column. We have decided to impute this variable with median as data is skewed. 

# In[27]:


imputer = SimpleImputer(strategy="median")
impute = imputer.fit(X_train)

X_train = impute.transform(X_train)
X_test = imputer.transform(X_test)


# ###  Building the model 

# ### Model evaluation criterion:
# 
# #### Model can make wrong predictions as:
# 1. Predicting a customer will buy the product and the customer doesn't buy - Loss of resources
# 2. Predicting a customer will not buy the product and the customer buys - Loss of opportunity
# 
# #### Which case is more important? 
# * Predicting that customer will not buy the product but he buys i.e. losing on a potential source of income for the company because that customer will not be targeted by the marketing team when he should be targeted.
# 
# #### How to reduce this loss i.e need to reduce False Negatives?
# * Company wants Recall to be maximized, greater the Recall lesser the chances of false negatives.

# **Let's start by building different models using KFold and cross_val_score and tune the best model using GridSearchCV and RandomizedSearchCV**
# 
# - `Stratified K-Folds cross-validation` provides dataset indices to split data into train/validation sets. Split dataset into k consecutive folds (without shuffling by default) keeping the distribution of both classes in each fold the same as the target variable. Each fold is then used once as validation while the k - 1 remaining folds form the training set.

# In[28]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("dtree", DecisionTreeClassifier(random_state=1)))
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("GBM", GradientBoostingClassifier(random_state=1)))
models.append(("Adaboost", AdaBoostClassifier(random_state=1)))
models.append(("Xgboost", XGBClassifier(random_state=1, eval_metric="logloss")))

results = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models


# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance:" "\n")

for name, model in models:
    scoring = "recall"
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scoring, cv=kfold
    )
    results.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean() * 100))

print("\n" "Training Performance:" "\n")

for name, model in models:
    model.fit(X_train, y_train)
    scores = recall_score(y_train, model.predict(X_train)) * 100
    print("{}: {}".format(name, scores))


# ### Model Building for Oversampled Data

# In[29]:


#pip install --upgrade scikit-learn


# In[30]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Load your data and split it into train and test sets

# Define oversampling method
oversample = RandomOverSampler(sampling_strategy='minority')

# Apply oversampling to your data
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

# Define the models
models = [('Decision Tree', DecisionTreeClassifier()),
          ('Bagging', BaggingClassifier()),
          ('Random Forest', RandomForestClassifier()),
          ('AdaBoost', AdaBoostClassifier()),
          ('Gradient Boosting', GradientBoostingClassifier()),
          ('XGBoost', XGBClassifier())]

# Fit each model and evaluate its performance on test set
for name, model in models:
    # Fit the model on oversampled data
    model.fit(X_train_over, y_train_over)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Evaluate performance using classification report and confusion matrix
    print(f'{name} Results:')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# ### Model Building for Undersampled Data

# In[31]:


import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Convert categorical columns to binary columns
df = pd.get_dummies(df)

# Split the data into training and testing sets
X = df.drop(columns=["Attrition_Flag"])
y = df["Attrition_Flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Perform undersampling on the training set to balance the classes
rus = RandomUnderSampler(random_state=1)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Define the 6 models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=1),
    "Bagging": BaggingClassifier(random_state=1),
    "Random Forest": RandomForestClassifier(random_state=1),
    "AdaBoost": AdaBoostClassifier(random_state=1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1),
    "XGBoost": xgb.XGBClassifier(random_state=1)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"{name}: Accuracy = {acc:.2f}, Precision = {prec:.2f}, Recall = {rec:.2f}, F1 Score = {f1:.2f}")



# In[32]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results)
ax.set_xticklabels(names)

plt.show()


# - We can see that XGBoost is giving the highest cross-validated recall followed by GBM and Adaboost
# - We will tune - Adaboost, GBM, XGBoost and see if the performance improves. 

# ###  Hyperparameter Tuning 

# * We will tune Adaboost, GBM, and xgboost models using GridSearchCV and RandomizedSearchCV. We will also compare the performance and time taken by these two methods - grid search and randomized search.

# **First, let's create two functions to calculate different metrics and confusion matrix so that we don't have to use the same code repeatedly for each model.**

# The following model performance classification function will return the accuracy, recall, precision and F1 score. 

# In[33]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1,},
        index=[0],
    )

    return df_perf


# The following function will return the confusion matrix for a model.

# In[34]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### AdaBoosting

# Let's tune Adaboost model using **GridSearch**
# 
# We define certain hyperparameters in the grid and GridSearchCV build model using every possible combination of the hyperparameters defined in the grid and it returns the best combination of the hyperparameters. Grid Search is usually computationally expensive. 

# **Grid Search**

# In[35]:


get_ipython().run_cell_magic('time', '', '\n# defining model\nmodel = AdaBoostClassifier(random_state=1)\n\n# Parameter grid to pass in GridSearchCV\n\nparam_grid = {\n    "n_estimators": np.arange(10, 110, 10),\n    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],\n    "base_estimator": [\n        DecisionTreeClassifier(max_depth=1, random_state=1),\n        DecisionTreeClassifier(max_depth=2, random_state=1),\n        DecisionTreeClassifier(max_depth=3, random_state=1),\n    ],\n}\n\n# Type of scoring used to compare parameter combinations\nscorer = metrics.make_scorer(metrics.recall_score)\n\n# Calling GridSearchCV\ngrid_cv = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs = -1)\n\n# Fitting parameters in GridSearchCV\ngrid_cv.fit(X_train, y_train)\n\nprint(\n    "Best Parameters:{} \\nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)\n)\n')


# In[36]:


# building model with best parameters
adb_tuned1 = AdaBoostClassifier(
    n_estimators=70,
    learning_rate=1,
    random_state=1,
    base_estimator=DecisionTreeClassifier(max_depth=2, random_state=1),
)

# Fit the model on training data
adb_tuned1.fit(X_train, y_train)


# **Checking model performance**

# In[37]:


# Calculating different metrics on train set
Adaboost_grid_train = model_performance_classification_sklearn(
    adb_tuned1, X_train, y_train
)
print("Training performance:")
print(Adaboost_grid_train)

print("*************************************")


# creating confusion matrix
confusion_matrix_sklearn(adb_tuned1, X_train, y_train)


# ####  Observations 
# - On comparing the CV score and the training score, model is overfitting. 
# - The validation recall is still less than 50% i.e. the model is not good at identifying potential customers who would take the offer.

# **Randomized Search** 

# In[38]:


# defining model
model = AdaBoostClassifier(random_state=1)

# Parameter grid to pass in GridSearchCV

param_grid = {
    "n_estimators": np.arange(10, 110, 10),
    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],
    "base_estimator": [
        DecisionTreeClassifier(max_depth=1, random_state=1),
        DecisionTreeClassifier(max_depth=2, random_state=1),
        DecisionTreeClassifier(max_depth=3, random_state=1),
    ],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_jobs=-1,
    n_iter=50,
    scoring=scorer,
    cv=5,
    random_state=1,
)

# Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train, y_train)

print(
    "Best parameters are {} with CV score={}:".format(
        randomized_cv.best_params_, randomized_cv.best_score_
    )
)


# In[39]:


# building model with best parameters
adb_tuned2 = AdaBoostClassifier(
    n_estimators=90,
    learning_rate=1,
    random_state=1,
    base_estimator=DecisionTreeClassifier(max_depth=2, random_state=1),
)

# Fit the model on training data
adb_tuned2.fit(X_train, y_train)


# **Checking model performance**

# In[40]:


# Calculating different metrics on train set
Adaboost_random_train = model_performance_classification_sklearn(
    adb_tuned2, X_train, y_train
)
print("Training performance:")
print(Adaboost_random_train)


print("*************************************")

# creating confusion matrix
confusion_matrix_sklearn(adb_tuned2, X_train, y_train)


# ####  Observations 
# - Grid search took a significantly longer time than random search. This difference would further increase as the number of parameters increases. 
# - The results from both grid and random search are similar

# ###   Gradient Boosting 

# Let's tune GBM model using **GridSearch**
# 
# We define certain hyperparameters in the grid and GridSearchCV build model using every possible combination of the hyperparameters defined in the grid and it returns the best combination of the hyperparameters. Grid Search is usually computationally expensive. 

# **Grid Search**

# In[41]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np

# defining model
model = GradientBoostingClassifier(random_state=1)

# Parameter grid to pass in GridSearchCV
param_grid = {
    "n_estimators": np.arange(10, 110, 10),
    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling GridSearchCV
grid_cv = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1)

# Fitting parameters in GridSearchCV
grid_cv.fit(X_train, y_train)

print(
    "Best Parameters:{} \nScore: {}".format(grid_cv.best_params_, grid_cv.best_score_)
)


# In[42]:


# building model with best parameters
gbm_tuned1 = GradientBoostingClassifier(
    n_estimators=70,
    learning_rate=1,
    random_state=1,
    max_depth=2,
)

# Fit the model on training data
gbm_tuned1.fit(X_train, y_train)


# **Checking model performance**

# In[43]:


# evaluate the model on the training set
train_score = gbm_tuned1.score(X_train, y_train)

# evaluate the model on the testing set
test_score = gbm_tuned1.score(X_test, y_test)

print("Train score:", train_score)
print("Test score:", test_score)

#if train_score > test_score:
#    print("The model is overfitting.")
#elif train_score < test_score:
#    print("The model is underfitting.")
#else:
#    print("The model is just right.")


# In[44]:


# Calculating different metrics on train set
gbm_grid_train = model_performance_classification_sklearn(gbm_tuned1, X_train, y_train)
print("Training performance:")
print(gbm_grid_train)

print("*************************************")


# creating confusion matrix
confusion_matrix_sklearn(gbm_tuned1, X_train, y_train)


# ####  Observations 
# - On comparing the CV score and the training score, model seems to be balanced.

# **Randomized Search** 

# In[45]:


# defining model
model = GradientBoostingClassifier(random_state=1)

# Parameter grid to pass in GridSearchCV
param_grid = {
    "n_estimators": np.arange(10, 110, 10),
    "learning_rate": [0.1, 0.01, 0.2, 0.05, 1],
    "max_depth": [1, 2, 3],
    "max_leaf_nodes": [None, 5, 10],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, n_iter=50, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# In[46]:


# building model with best parameters
gbm_tuned2 = GradientBoostingClassifier(
    n_estimators=90,
    learning_rate=1,
    random_state=1,
    max_depth=2,
    max_leaf_nodes=None,
)

# Fit the model on training data
gbm_tuned2.fit(X_train, y_train)


# **Checking model performance**

# In[47]:


# evaluate the model on the training set
train_score = gbm_tuned2.score(X_train, y_train)

# evaluate the model on the testing set
test_score = gbm_tuned2.score(X_test, y_test)

print("Train score:", train_score)

print("Test score:", test_score)

# if train_score > test_score:
#    print("The model is overfitting")
# elif train_score < test_score:
#    print("The model is underfitting.")
# else:
#    print("The model is just right.")


# - Both scores seem to be close, making the model balanced.

# In[48]:


# Calculating different metrics on train set
gbm_random_train = model_performance_classification_sklearn(
    gbm_tuned2, X_train, y_train
)
print("Training performance:")
print(gbm_random_train)


print("*************************************")

# creating confusion matrix
confusion_matrix_sklearn(gbm_tuned2, X_train, y_train)


# ####  Observations 
# - On comparing the CV score and the training score, model seems to be balanced as the accuracy, recall, and precision are almost within the same range. F1 score is high also.

# ##   XGBoost 

# **Grid Search**

# In[49]:


# defining model
model = XGBClassifier(random_state=1, eval_metric="logloss")

# Parameter grid to pass in GridSearchCV
param_grid = {
    "n_estimators": [10, 30, 50],
    "scale_pos_weight": [0, 1],
    "subsample": [0.5, 0.9],
    "learning_rate": [0.1, 0.2],
    "gamma": [0, 1],
    "colsample_bytree": [0.5, 0.9],
    "colsample_bylevel": [0.5, 0.9],
}


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling GridSearchCV
grid_cv = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring=scorer, cv=3, n_jobs=-1, verbose=2
)

# Fitting parameters in GridSearchCV
grid_cv.fit(X_train, y_train)


print(
    "Best parameters are {} with CV score={}:".format(
        grid_cv.best_params_, grid_cv.best_score_
    )
)


# In[50]:


# building model with best parameters
xgb_tuned1 = XGBClassifier(
    random_state=1,
    n_estimators=50,
    scale_pos_weight=10,
    subsample=0.9,
    learning_rate=0.1,
    gamma=0,
    eval_metric="logloss",
    reg_lambda=5,
    max_depth=1,
)

# Fit the model on training data
xgb_tuned1.fit(X_train, y_train)


# **Checking model performance**

# In[51]:


# Calculating different metrics on train set
xgboost_grid_train = model_performance_classification_sklearn(
    xgb_tuned1, X_train, y_train
)
print("Training performance:")
print(xgboost_grid_train)

print("*************************************")


# creating confusion matrix
confusion_matrix_sklearn(xgb_tuned1, X_train, y_train)


# ####  Observations 
# - The validation recall has increased by >54% as compared to the result from cross-validation with default parameters.
# - The model has very low precision score (however low precision shouldn't affect us much here)

# **Randomized Search** 

# In[52]:


# defining model
model = XGBClassifier(random_state=1, eval_metric="logloss")

# Parameter grid to pass in RandomizedSearchCV
param_grid = {
    "n_estimators": np.arange(50, 150, 50),
    "scale_pos_weight": [2, 5, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.05],
    "gamma": [0, 1, 3, 5],
    "subsample": [0.8, 0.9, 1],
    "max_depth": np.arange(1, 5, 1),
    "reg_lambda": [5, 10],
}

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Calling RandomizedSearchCV
xgb_tuned2 = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=50,
    scoring=scorer,
    cv=5,
    random_state=1,
    n_jobs=-1,
)

# Fitting parameters in RandomizedSearchCV
xgb_tuned2.fit(X_train, y_train)

print(
    "Best parameters are {} with CV score={}:".format(
        xgb_tuned2.best_params_, xgb_tuned2.best_score_
    )
)


# In[53]:


# building model with best parameters
xgb_tuned2 = XGBClassifier(
    random_state=1,
    n_estimators=50,
    scale_pos_weight=10,
    gamma=1,
    subsample=1,
    learning_rate=0.1,
    eval_metric="logloss",
    max_depth=1,
    reg_lambda=10,
)
# Fit the model on training data
xgb_tuned2.fit(X_train, y_train)


# **Checking model performance**

# In[54]:


# Calculating different metrics on train set
xgboost_random_train = model_performance_classification_sklearn(
    xgb_tuned2, X_train, y_train
)
print("Training performance:")
print(xgboost_random_train)

print("*************************************")


# creating confusion matrix
confusion_matrix_sklearn(xgb_tuned2, X_train, y_train)


# ####  Observations
# - The parameters obtained from both grid search and random search are approximately same
# - The performance of both the models is also very similar
# - Tuning with grid search took a significantly longer time

# ###  Comparing all models 

# In[55]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        Adaboost_grid_train.T,
        Adaboost_random_train.T,
        gbm_grid_train.T,
        gbm_random_train.T,
        xgboost_grid_train.T,
        xgboost_random_train.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "AdaBoosting Tuned with Grid search",
    "AdaBoosting Tuned with Random search",
    "Gradient Boosting Tuned with Grid search",
    "Gradient Boosting Tuned with Random search",
    "Xgboost Tuned with Grid search",
    "Xgboost Tuned with Random Search",
]
print("Training performance comparison:")
models_train_comp_df


# ####  Observations 
# - On comparing CV scores and the training score, AdaBoosting Tuned with random search and gradient boosting model tuned using random search is giving the better results. 
# 
# - We will go ahead with AdaBoosting tuned.
# 
# - Let's check the model's performance on test set and then see the feature importance from the tuned adaboost model.

# ###  Performance on the test set

# In[56]:


# Calculating different metrics on the test set
adaboost_grid_test = model_performance_classification_sklearn(
    adb_tuned2, X_test, y_test
)
print("Test performance:")
adaboost_grid_test


# - It looks like they are all betwee 0.89 and 0.97. It seems to be balanced.

# ### Feature Importance Using Sklearn 

# In[57]:


feature_names = X.columns
importances = adb_tuned2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 12))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# ####  Observations 
# - Total transaction amount is the most important feature, followed by total transaction count and Ratio of the total transaction count in 4th quarter and the total transaction count in 1st quarter.

# **Let's try new package "SHAP" to explore the contribution of features in making the prediction**

# ###   SHAP (SHapley Additive exPlanations) 
# 
# * SHAP stands for SHapley Additive exPlanations.
# * SHAP makes the sophisticated machine learning models easy to understand.
# * We need to be sure of what our model is actually predicting, for that we have to take a closer look at each variable and SHAP open up the possibilities to explore which variables were intensely used by the model to make predictions. 
# * It is a great tool to that tell us how each feature in the model has contributing to the predictions. 
# * It helps in visualizing the relationships in the model
# 

# #### Installing shap

# In[58]:


# Install library using
# In jupyter notebook
get_ipython().system('pip install shap')

# or
# In anaconda command prompt
# conda install -c conda-forge shap - in conda prompt


# #### Import the package

# In[59]:


import shap


# In[60]:


## Initialize the package
shap.initjs()


# #### Calculating the shap values

# In[61]:


explainer = shap.TreeExplainer(xgb_tuned1)
shap_values = explainer.shap_values(X)


# ###   Shap summary plot 
# - The Y-axis indicates the variable names, arranged in order of importance from top to bottom.
# - The X-axis indicates the Shap value.
# - For every variable, the dot represents an observation
# - The color represents the value of the feature from low to high.
# - The farther away from the values from the central line(SHAP = 0), the more impact that variable will have on predictions

# In[62]:


# Make plot.
shap.summary_plot(shap_values, X)


# #### <span style="color:blue">  Observations </span>
# * Total_Trans_Ct, Total_Revolving_Bal, and Total_Trans_Amt are the top three important features that contributes in the prediction of target. 
# * Total_Trans_Ct and Total_Revolving_Bal have high negative impact on the response as higher the values of Total_Trans_Ct and Total_Revolving_Bal, lower the chances of not renouncing.
# * Strangely, Total_Trans_Ct and Total_Revolving_Bal have positive impact on Attrtion_Flag i.e lower the value of Total_Trans_Ct and Total_Revolving_Bal, lower the chances of not renouncing. 

# #### Let's look at one specific observation to get deeper insights

# ###  Force Plot 
# - Force plot can be informative in understanding the contribution of each variable in the prediction of a given observation. 
# - The output f(x) is the score predicted by the model for a given observation. 
# - Higher scores lead the model to make the predictions closer to 1 and low scores make the predictions closer to 0
# - Features in red color influence positively i.e make the predictions closer to 1, whereas blue color influence negatively. 
# - Base value is the mean prediction value by the model. 
# - Features that had more impact on the score are located closer to the dividing boundary between red and blue. 
# - The impact of each feature is represented by the size of the bar. 
# 
# 

# In[63]:


### Exploring an individual observation
shap.force_plot(explainer.expected_value, shap_values[1, :], X.iloc[1, :])


# #####   Observations 
# - The model predicts 0.07 score for the given observation (the index value(1) defined in force_plot())
# - Total_Trans_Ct has a positive impact on the prediction i.e influences the model to predict the score close to 1 
# - Total_Revolving_Bal, Months_Inactive_12_mon, Total_Amt_Chng_Q4_Q1, Total_Ct_Chng_Q4_Q1 have a negative impact on the prediction i.e influences the model to predict the score close to 0. 

# ###  Pipelines for productionizing the model 
# 
# - Pipeline is a means of automating the machine learning workflow by enabling data to be transformed and correlated into a model that can then be analyzed to achieve outputs. This type of ML pipeline makes the process of inputting data into the ML model fully automated. 
# 
# - Now, we have a final model. let's use pipelines to put the model into production

# ###   Column Transformer 
# * We know that we can use pipelines to standardize the model building, but the steps in a pipeline are applied to each and every variable - how can we personalize the pipeline to perform different processing on different columns
# * Column transformer allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space. This is useful for heterogeneous or columnar data, to combine several feature extraction mechanisms or transformations into a single transformer.

# We will create 2 different pipelines, one for numerical columns and one for categorical columns. For numerical columns, we will do missing value imputation as pre-processing. For categorical columns, we will do one hot encoding and missing value imputation as pre-processing
# We are doing missing value imputation for the whole data, so that if there is any missing value in the data in future that can be taken care of.

# In[64]:


# creating a list of numerical variables
numerical_features = [
     "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# creating a transformer for numerical variables, which will apply simple imputer on the numerical variables
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


# creating a list of categorical variables
categorical_features = ["Attrition_Flag",
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",]

# creating a transformer for categorical variables, which will first apply simple imputer and 
#then do one hot encoding for categorical variables
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# handle_unknown = "ignore", allows model to handle any unknown category in the test data

# combining categorical transformer and numerical transformer using a column transformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)
# remainder = "passthrough" has been used, it will allow variables that are present in original data 
# but not in "numerical_columns" and "categorical_columns" to pass through the column transformer without any changes


# In[65]:


# Separating target variable and other variables
#df1=df00.copy()
#X = df.drop("Attrition_Flag", axis=1)
#y = df["Attrition_Flag"]


# - Now we already know the best model we need to process with, so we don't need to divide data into 3 sets - train, validation and test

# In[66]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=1, stratify=Y
)
print(X_train.shape, X_test.shape)


# In[67]:


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("BankChurners.csv")

# Include the Attrition_Flag column
df = df[['Attrition_Flag', 'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]


# Encode the categorical features
categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

# Scale the numerical features
scaler = StandardScaler()
num_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
num_transformer = Pipeline(steps=[
    ('scaler', scaler)
])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', num_transformer, num_cols)
    ], remainder='passthrough')

# Split the data into training and testing sets
X = df.drop(['Attrition_Flag'], axis=1)
y = df['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

model = Pipeline(
    steps=[
        ("pre", preprocessor),
        (
            "XGB",
            XGBClassifier(
                random_state=1,
                n_estimators=50,
                scale_pos_weight=10,
                subsample=0.8,
                learning_rate=0.01,
                gamma=0,
                eval_metric="logloss",
                reg_lambda=5,
                max_depth=1,
            ),
        ),
    ]
)
# Fit the model on training data
#model.fit(X_train, y_train)
model.fit(X, y)


# ### Business Recommendations

# - Thera Bank should target customers who transactioned the amount of around 4,404 dollars in the last 12 months. This should be looked into as the average annual income for most customers are less than 40k dollars.
# - We observed in our analysis that total transaction amount and count are srong negative factors for customers. The bank should provide reward system for depositing "x" amount of money in their account.
# - The relationship between the bank and its customers is 35 months or almost 3 years on average. At least 75% of customers were inactive for 3 months in the last 12 months. At least 25% or more customers had 2 contacts between the customer and bank in the last 12 months. To increase the relationship, the bank could provide more options in their customer service, such as options for lower late fees if the customer is late for paying. 
# - The best test recall is ~90% and the test precision is 89% at the same time. This means that the model is good at identifying non-defaulter, therefore, the bank has more chances of not losing its customers since it was identified at the right time.
