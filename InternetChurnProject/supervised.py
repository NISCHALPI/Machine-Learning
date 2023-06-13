#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project: Churn Prediction for an Internet and Telephone Provider
# 
# This Jupyter notebook presents a comprehensive machine learning project focused on designing an accurate churn prediction model for an internet and telephone provider.
# 
# ## Project Objectives
# 
# - Develop a machine learning model to predict customer churn, helping the provider identify customers at risk of leaving.
# - Conduct exploratory data analysis (EDA) to gain insights into the churn data and understand key patterns and trends.
# - Perform feature engineering to create meaningful and predictive features from the available dataset.
# - Evaluate multiple models using appropriate evaluation metrics to identify the most effective algorithm for churn prediction.
# - Fine-tune the chosen model through hyperparameter tuning to optimize its performance.
# - Deploy the finalized model in a production environment for real-time churn prediction.
# 
# ## Project Phases
# 
# 1. **Exploratory Data Analysis (EDA):** Explore and visualize the churn data to gain a deep understanding of its characteristics, uncover important relationships, and identify potential challenges or biases.
# 
# 2. **Feature Engineering:** Transform and enhance the dataset by creating new features, encoding categorical variables, handling missing values, and scaling numerical features. This step aims to maximize the predictive power of the model.
# 
# 3. **Model Evaluation and Selection:** Train and evaluate different machine learning models using appropriate metrics such as accuracy, precision, recall, and F1-score. Compare their performance to select the most suitable model for churn prediction.
# 
# 4. **Hyperparameter Tuning:** Fine-tune the chosen model by optimizing its hyperparameters using techniques like grid search or randomized search. This step aims to further improve the model's predictive performance.
# 
# 5. **Model Deployment:** Integrate the finalized model into the provider's infrastructure for real-time predictions. This involves creating an API or web service that enables the provider to make accurate churn predictions on new customer data.
# 
# ## Conclusion
# 
# By the end of this project, we will have developed a robust churn prediction model that empowers the internet and telephone provider to proactively address customer churn. The insights gained from the project will contribute to improved customer retention and ultimately drive business growth.
# 
# Let's embark on this exciting machine learning journey and build an exceptional churn prediction model!
# 

# ## SECTION : Exploratory Data Analysis

# In[1]:


# Necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns  


# In[2]:


# Read the churn data 
df = pd.read_csv(filepath_or_buffer='Telco-Customer-Churn.csv')


# In[3]:


# Basic info
df.info()


# In[4]:


df.head()


# Clearly no data is missing and customer ID is irrelevent. There are lot of of categorical features.

# In[5]:


#Drop customer id
df = df.drop(columns='customerID')


# In[6]:


# Let's see if our feature is balance or not
sns.countplot(data=df, x='Churn')


# So, the churn count is lower. The dataset is unblanced over the class. Might need to set weights to balance in ML algorithm 

# In[7]:


# Let's see the relationship between monthly charge and churn rate
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')


# Ohh there it is. Who want to pay more money? Clearly, there is good relationship between monthly charge and churn rate. Let's see if Internet Service have a impact 

# In[8]:


sns.countplot(data=df, x='InternetService', hue='Churn');


# Churn rate is clearly high for Fiber optics Internet Service. How is the TechSupport there ?

# In[9]:


sns.catplot(data=df, x='TechSupport', col='InternetService', kind='count');


# Clearly, there are more subscriber and comparably less tech support for fiber optics. Need to fund the tech support for this department

# Let's see the relationship between contarct and churn rates.

# In[10]:


sns.countplot(data=df, x='Contract', hue='Churn');


# Clearly, churn rates are higher in month-to-month cotract. Marketing should focus on long term contract.

# In[11]:


sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True);


# Clearly, we are losing new customer more than old customers.

# In[12]:


sns.countplot(data=df, x='Partner', hue='Churn');


# Churn rate is higher in non patners customers.

# Is security as reason for churn rates? Let's see

# In[13]:


sns.countplot(data=df, x='OnlineSecurity', hue='Churn');


# Clearly, security also has some relationship with churn rate.

# In[14]:


df.columns


# Should we push for phone service for customer retentation ? 

# In[15]:


sns.countplot(data=df, x='MultipleLines', hue='Churn')


# No clear indication! Probably not the reson for churn rate

# # Section: Feature Engineering

# Let's one hot encode these categorical data!

# In[16]:


df.info()


# Let's seprate feature and lables.

# In[17]:


df = pd.get_dummies(df, drop_first=True)


# In[18]:


# Correlation with churn rates, there are signals! 
plt.figure(figsize=(12,8))
correlation = df.corr()['Churn_Yes'].sort_values(ascending=False)[1:]
sns.barplot(x=correlation.index, y=correlation)
plt.xticks(rotation=90);


# In[19]:


# Features and Lables 
X = df.drop(columns='Churn_Yes')
y = df['Churn_Yes']


# In[20]:


X.head()


# In[21]:


y.head()


# Let's do some train test splits and feature scaling with the data.

# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# # Section: Model Selection and Evaluation 

# ## Potential Classifiers
# 1. **Logistic Regression** 
# 2. **KNearest Neighbours**
# 3. **Support Vector Machines**
# 4. **Random Forest Classifiers**
# 5. **Boosted Classifiers**
# 6. **Bagged Classifiers**
# 
# Might need to do a grid search for hyperparameter tuning. Let's build some pipelines for potential candidates.

# In[117]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

# Let's start with default parameters to see baseline performance 

pipe_log = make_pipeline(StandardScaler(), LogisticRegression( class_weight='balanced', random_state=101))
pipe_rf = make_pipeline(StandardScaler(), RandomForestClassifier(class_weight='balanced', random_state=101))
pipe_erf = make_pipeline(StandardScaler(), ExtraTreesClassifier(class_weight='balanced', random_state=101))
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
pipe_svc = make_pipeline(StandardScaler(), SVC(class_weight='balanced', random_state=101, probability=True))
pipe_gbc  = make_pipeline(StandardScaler() ,GradientBoostingClassifier(random_state=101))
pipe_ada = make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=101))
# List of pipes 
pipes = [pipe_log, pipe_rf, pipe_erf ,pipe_knn, pipe_svc, pipe_gbc, pipe_ada]


# In[118]:


# Let's see the cross val score of these model on the traning set
from sklearn.model_selection import cross_val_score

scores = []
for model in pipes:
    scores.append(cross_val_score(model , X_train , y_train,n_jobs=-1, cv=10 , scoring='accuracy').mean())


# Let's see the baseline crossvalidation performace without hyperparameter tuning!

# In[119]:


sns.barplot(x=[list(pipe.named_steps.keys())[1] for pipe in pipes], y=scores)
plt.xticks(rotation = 90);
scores


# They all have similar performance but trees seems to have a edge over others. 

# # Section: Hyperparameter Tuning  

# Let's do a grid search for best hyperparameter for RandomForest,GradientBoosted Trees,SVC and KNN!

# In[120]:


from sklearn.model_selection import GridSearchCV

hyper_parms_rf = {'randomforestclassifier__max_depth' : [None , *np.arange(3,10)],'randomforestclassifier__n_estimators' : np.arange(64,128) ,'randomforestclassifier__criterion': ['gini', 'entropy'], 'randomforestclassifier__max_features' :['sqrt', 'log2']}
hyper_parms_gbc = {'gradientboostingclassifier__learning_rate': np.linspace(0.1, 1, 10) ,'gradientboostingclassifier__n_estimators' : np.arange(64,128)}
hyper_parms_svc = {'svc__kernel': ['rbf', 'linear', 'poly'], 'svc__gamma': ['scale', 'auto'], 'svc__degree': [2,3], 'svc__C': np.logspace(-4,4,11)}
hyper_parm_knn = {'kneighborsclassifier__n_neighbors': np.arange(3,21)}


# In[121]:


grid_rf = GridSearchCV(estimator=pipe_rf, param_grid=hyper_parms_rf, cv=5, n_jobs=-1, scoring=['accuracy', 'f1'], verbose=2, refit='accuracy', pre_dispatch='n_jobs')
grid_gbc = GridSearchCV(estimator=pipe_gbc, param_grid=hyper_parms_gbc, cv=5, n_jobs=-1, scoring=['accuracy', 'f1'], verbose=2, refit='accuracy', pre_dispatch='n_jobs')
grid_svc = GridSearchCV(estimator=pipe_svc, param_grid=hyper_parms_svc,  scoring=['accuracy', 'f1'], verbose=2, refit='accuracy', pre_dispatch='n_jobs', cv=5, n_jobs=-1)
grid_knn = GridSearchCV(estimator=pipe_knn, param_grid=hyper_parm_knn,  scoring=['accuracy', 'f1'], verbose=2, refit='accuracy', pre_dispatch='n_jobs', cv=5, n_jobs=-1)


# In[122]:


# Caching Model 
import os
from joblib import dump, load  
def check_and_load(grid: GridSearchCV = None, filename: str = None) -> GridSearchCV:
    if filename in os.listdir('./learners/'):
        grid = load(filename=f'./learners/{filename}')
        return grid
    else:
        grid.fit(X_train, y_train)
        dump(grid, filename=f'./learners/{filename}')
    return grid


# In[ ]:


grid_knn = check_and_load(grid_knn, filename='knn.joblib')
grid_rf = check_and_load(grid_rf, filename='rf.joblib')
grid_svc = check_and_load(grid_svc, filename='svc.joblib')
grid_gbc = check_and_load(grid_gbc, filename='gbc.joblib')

regularized_estimators = [grid_rf,grid_gbc,grid_rf,grid_knn]
estimators_name = [list(estimators.best_estimator_.named_steps.keys())[1] for estimators in regularized_estimators]


# The grid are trained and optimized. Let's see the performance on the test set of the individual estimators.

# In[53]:


from sklearn.metrics import accuracy_score
accuracy_list = []

for estimators in regularized_estimators:
    accuracy_list.append(accuracy_score(y_test, estimators.predict(X_test)))
    print(f'{list(estimators.best_estimator_.named_steps.keys())[1]} has a accuracy of {np.round(accuracy_list[-1], decimals=3)}')


# In[56]:


sns.barplot(x= estimators_name, y = accuracy_list)
plt.xticks(rotation=90);


# We see that all of them perform similary in the test data. Since these are different algorathims, we can use voting meta learner to combine all of these classifies to get a consistent predictions.

# # SECTION: Deployment of the model

# Let's create a meta learner and deploy the model.

# In[106]:


meta_estimators_list = [pipe_log, grid_rf.best_estimator_, pipe_erf ,grid_knn.best_estimator_, grid_svc.best_estimator_, grid_gbc.best_estimator_, pipe_ada]
meta_estimators_names = [list(pipe.named_steps.keys())[1] for pipe in meta_estimators_list]


# In[107]:


from sklearn.ensemble import VotingClassifier

meta_estimator = VotingClassifier(list(zip(meta_estimators_names, meta_estimators_list)), voting='hard', n_jobs=-1)

meta_estimator.fit(X_train, y_train)


# We have trained our meta learner and now we can check the accuracy and retrain and deploy!

# In[115]:


from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print(classification_report(y_test, meta_estimator.predict(X_test)))
ConfusionMatrixDisplay(confusion_matrix=None).from_estimator(meta_estimator, X_test, y_test)


# Okay, let's retrain and deploy 

# In[109]:


meta_estimator.fit(X, y)


# In[123]:


from joblib import dump
dump(value=meta_estimator, filename='final_model')


# In[ ]:




