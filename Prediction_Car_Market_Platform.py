# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <p style="font-size: 36px"><strong>Case Study - Car Market Platform Predictions</strong></p>

# by *Julian Dörr*

# # Load modules 

from util import plot_describe, ClassifierModule, RegressorModule
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import datetime
import csv
from skorch import NeuralNetClassifier, NeuralNetRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, make_scorer, fbeta_score, mean_absolute_error, mean_poisson_deviance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from IPython.display import Markdown as md

# In this notebook two vanilla feed forward neural networks are trained using PyTorch. If you want to replicate the code, please specify on which compute device you want to conduct the training. If set to 'cuda' GPU acceleration is used. I have a NVIDIA GPU available on my local machine, so device defaults to 'cuda'.

device='cuda' # change to 'cpu' if no GPU is available

# # Inspect and clean data

# Read data.

df = pd.read_csv(
    './data/AS24_Case_Study_Data.csv', 
    sep=';', 
    parse_dates=['created_date', 'deleted_date'], 
    dayfirst=True,
    date_format='%d.%m.%y'
)

# Are there any missings?

df.isnull().sum(axis=0)

df.loc[df.ctr.isnull() | df.search_views.isnull() | df.detail_views.isnull()]

# There are missing values in `search_views`, `detail_views` and consequently `ctr` for a very small fraction (< 0.1%) of observations of which all belong to the 'Basic' `product_tier` group. Without further information on the decision process which cars are viewed more/less often implementing a reasonable imputation strategy does not add much value (consulting the process-owning business unit would be required first). Moreover, the fraction of missing values is negligible which is why we simply drop the underlying observations (implicit assumption: missing values occur at random).

df = df.loc[~(df.ctr.isnull() | df.search_views.isnull() | df.detail_views.isnull())]

# What are the data types?

df.dtypes

# Why is `ctr` read as string? `ctr` is defined as click through rate and calculated as the quotient of detail_views over search_views. The resulting data type should be a float.

df.ctr.value_counts(dropna=False).sort_index(ascending=False)

# Values such as '9.677.419.354.838.700' seem to be a data quality issue. Recalculate `ctr` according to its definition: <br>
# $ctr = \frac{detail\_views}{search\_views}$

df['ctr'] = df.detail_views/df.search_views

# Let's visually inspect the data to spot potential outliers or other irregularities.

plot_describe(df, columns=df.drop(columns=['article_id', 'created_date', 'deleted_date']).columns, 
              log_dict={
                  'make_name': True,
                  'first_registration_year': True, 
                  'search_views': True, 
                  'detail_views': True,
                  'ctr': True})

# Findings:
# - `product_tier` is highly imbalanced. 'Plus' and 'Premium' cars are strongly underrepresented. This will make training of a classification model that is capable to reliably classify the minority groups challenging.
# - `first_registration_year` shows one value > 2100 which seems to be a data quality issue. Car registrations in the future make no sense.
# - `stock_days` shows at least one value < 0 which seems another data quality issue. Time cannot be negative.
# - `search_views`, `detail_views` and consequently `ctr` are strongly skewed to the right with some exceptionally high values.

# Drop observation with `first_registration_year` in the future. Again one could think of a imputation strategy but since only one observation is missing, dropping this observation is the more pragmatic/efficient solution.

df.loc[df.first_registration_year > datetime.date.today().year]

df = df.loc[~(df.first_registration_year > datetime.date.today().year)]

# Inspect negative values in `stock_days`.

df.loc[df.stock_days<0].sort_values('stock_days').head(3)

# It seems that `stock_days` does not strictly follow its definition as the time in days between the creation of the listing and the deletion of the listing. Recalculate `stock_days` according to its definition: <br>
# $stock\_days = deleted\_date - created\_date$

df['stock_days'] = (df.deleted_date - df.created_date).dt.days

# The data seems now to be sufficiently clean. So we can proceed with some feature engineering.

# # Feature Engineering

# Let's analyze correlations between the numeric features first in order to understand how the variables relate to one another.

sns.heatmap(df.drop(columns=['article_id', 'product_tier',  'make_name', 'created_date', 'deleted_date']).corr(), vmin=-1, vmax=1, annot=True, cmap=sns.diverging_palette(246, 246, s=100, l=50, as_cmap=True))

# Findings:
# - Correlations between features are low in most cases
# - Third strongest correlation between `stock_days` and `search_views`/`detail_views` which means that the longer a car is listed the more often it is shown/clicked.
# - Second strongest (but still modest) correlation between `price` and `first_registration_year` indicating that newer vehicles are typically pricier (ignoring that old timers may be even more pricier). From this observation it makes sense to calculate the vehicle's age at the time it was listed on the online platform which is a better feature than `first_registration_year` because it captures the actual age of the car by additionally controlling when the car has been listed which can be at different points in time. This feature is defined as time between the vehicle's first registration and its listing date in years: $vehicle\_age = created\_date - first\_registration\_year$
# - Strongest correlation between `search_views` and `detail_views` which is to be expected. The more often a product is shown to the customer, the more often this product is clicked by customers. For the second modelling task this observation requires special attention.

# Let's calculate the vehicle's age which is more suited as feature than the product's first registration year. Note that for some observations year of first registration is greater than year of listing which results in a negative vehicle age. I tackle this data quality issue by replacing a negative vehicle age with zero (reflecting a new car) which seems to be a reasonable strategy.

df['vehicle_age'] = (df.created_date.dt.year - df.first_registration_year).apply(lambda x: max(x, 0)) 

# # Modelling

# ## Classification model 

# **Is it possible to predict the product tier from the information given in the other columns?**

# To tackle this question we aim at developing a classification model that allows to predict how likely it is that a product belongs to any of the three product tier categories based on the other product characteristics (features). From a business perspective this seems to be mainly a process automation task, i.e. is it possible to auto-populate the `product_tier` field, once the user provided information on other product characteristics which entail a strong signal on the product's tier.

# Let us get a better feeling how the target variable `product_tier` is distributed across the different features.

plot_describe(df, columns=df.drop(columns=['article_id', 'created_date', 'deleted_date']).columns, 
              log_dict= {}, hue='product_tier')

# Findings:
# - 'Premium' products show a relatively high concentration in ZIP codes starting with 2.
# - In general, from a pure visual and bivariate inspection it does not look like the features show a high discriminatory power in distinguishing the three product tier classes because the feature distributions across the three classes look quite similar.
# - Given the many different make categories it cannot be clearly judged whether the car make entails a strong signal in discriminating between the three product categories.

# To obtain a clearer picture on answering the initial question it requires a multivariate classification model that is developed in the following.

# This classifier shall serve the purpose of indicating the probability of a product to belong to any of the three product tier classes. The major challenge will be to train a model that can handle the imbalanced data set. To tackle this issue we try oversampling the minority groups in the training data. For this purpose, we use a Synthetic Minority Over-Sampling Technique ([SMOTE](https://arxiv.org/abs/1106.1813)) approach. We also try an undersampling technique using [Tomek Links](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4309452) which helps to remove noisy or hard to classify observations in the majority group that do not help the underlying model to find a suitable discrimination boundary. Moreover, we acknowledge that in a use case where the `product_tier` field shall be autopopulated *at the time when the product is listed on the platform*, features that are only generated over the product's listing time will not be available at inference. Therefore, we drop the features `search_views`, `detail_views`, `stock_days` and `ctr` when training the model.

# +
# Features
X = df.drop(columns=['article_id', 'product_tier', 'created_date', 'deleted_date', 'first_registration_year'])
# One-hot encoding of make_name
X = pd.get_dummies(X, columns=['make_name'], drop_first=True, dtype=np.float32) # car make 'AC' serves as reference group
X = X.astype(np.float32)                                                        # create common dtype to ease matrix manipulation

# Target
y = df['product_tier']
# -

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

# Drop features that are likely not available at inference time
X_train_reduced = X_train.drop(columns=['search_views', 'detail_views', 'stock_days', 'ctr']) # these features are generated over the product's listing period
X_test_reduced = X_test.drop(columns=['search_views', 'detail_views', 'stock_days', 'ctr'])   # these features are generated over the product's listing period

# +
# Oversampling of minority groups via SMOTE
N_majority = y_train.value_counts()['Basic'] # number of observations in the majority group 'Basic'
ratio = 5                                    # for every fifth observation in the majority group there will be one observation in both minority groups

# On original feature set 
X_train_oversampled, y_train_oversampled = SMOTE(
    sampling_strategy={'Plus': round(N_majority/ratio), 'Premium': round(N_majority/ratio)},
    random_state=333).fit_resample(X_train, y_train)

# On reduced feature set
X_train_reduced_oversampled, y_train_reduced_oversampled = SMOTE(
    sampling_strategy={'Plus': round(N_majority/ratio), 'Premium': round(N_majority/ratio)},
    random_state=333).fit_resample(X_train_reduced, y_train)

# +
# Undersampling of majority group via Tomek Links

# On original feature set 
X_train_undersampled, y_train_undersampled = TomekLinks(
    sampling_strategy='majority'                                   # undersample only the majority class
    ).fit_resample(X_train, y_train)

# On reduced feature set
X_train_reduced_undersampled, y_train_reduced_undersampled = TomekLinks(
    sampling_strategy='majority'                                  # undersample only the majority class
    ).fit_resample(X_train_reduced, y_train)
# -

# The target variable in the original training data is highly imbalanced with the following distribution across the `product_tier` categories:

pd.DataFrame(round(y_train.value_counts()/len(y_train), 3)).rename(columns={'count': 'Distribution'})

# By oversampling the minority groups 'Plus' and 'Premium' we can create training data that is more balanced in its target:

pd.DataFrame(round(y_train_oversampled.value_counts()/len(y_train_oversampled), 3)).rename(columns={'count': 'Distribution'})

# By removing observations in the majority class which are closest to the observations from the 'Plus' and 'Premium' class undersampling aims at making it easier for the model to find a suitable discrimination boundary. With this undersampling technique the distribution across the `product_tier` categories looks as follows:

pd.DataFrame(round(y_train_undersampled.value_counts()/len(y_train_undersampled), 3)).rename(columns={'count': 'Distribution'})

# In the following, we train commonly used ML/DL classification models for the purpose of testing whether the product tier can be auto-populated via a predictive model which reliably indicates to which tier a product belongs based on other product features that are typically available at inference. Our setup is as follows:
# - for the sake of comparability we train the classifier on the <br>
#   (1) original training data (without resampling) <br>
#   (2) oversampled training data <br>
#   (3) undersampled training data.
# - since it is not entirely clear which features are available at inference time for the underlying business case, we also train the classifier on the <br>
#   (1) original training data (incl. features likely not available at inference) <br>
#   (2) reduced training data (excl. features likely not available at inference) <br>
#   and we do so for each resampling strategy mentioned above.
# - we aim at approximating the best set of hyperparameters on validation sets using cross-validation in order to prevent overfitting.
# - we focus on optimizing the average F1-Score. In other words, our goal is to create a classification model that performs well in identifying the maximum number of 'Plus', 'Premium' and 'Basic' vehicles which indeed belong to the 'Plus'/'Premium'/'Basic' class (Recall), while also ensuring that those vehicles classified as 'Plus', 'Premium' or 'Basic' do not actually belong to another product tier class (Precision).
# - for scale sensitive models (e.g. Support Vector Machine, Neural Network) we scale the features by subtracting their mean and dividing by their standard deviation.

# Define a feed-forward neural network (2 layers) for the regression task based on PyTorch (see util.py)
NNClassifier = NeuralNetClassifier(
    module=ClassifierModule,
    max_epochs=20,
    lr=0.1,
    device=device,       # I have a GeForce RTX 3050 GPU on my local machine which I use for training the neural network
    train_split=False,   # disable the skorch-internal train-validation split as GridSearchCV conducts the train-validation splits
    verbose=0,           # Verbosity is steered via GridSearchCV
)

# +
# Define scoring metric
fscore_tier = make_scorer(
    score_func=fbeta_score, 
    beta=1,                      # equal weight on Precision and Recall
    #labels=['Plus', 'Premium'], # focus on prediction quality of minority groups only
    average='macro'              # calculate metrics for each label, and find their unweighted mean. 
                                 # This means that a high F-Score for the majority group but a low F-Score for 
                                 # the minority groups leads to an overall low F-Score. We aim at a model 
                                 # that performs well in distinguishing between the three categories and 
                                 # not at one that is biased towards the majority class.
    )       

# Define the pipeline for each model
pipelines = [
    ('logistic_regression', Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(                               
            max_iter=600,
            multi_class='multinomial',                              # handles multiclass problem inherently with multi_class='multinomial'
            solver='lbfgs'))                 
    ])),
    ('random_forest', Pipeline([
        ('model', RandomForestClassifier(random_state=333))         # handles multiclass problem inherently
    ])),
    ('gradient_boosting', Pipeline([
        ('model', GradientBoostingClassifier(random_state=333))     # handles multiclass problem as one-vs-rest: K trees (for K classes) are built at each of the M iterations
    ])),
    ('support_vector_machine', Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(random_state=333))                            # handles multiclass problem as one-vs-one
    ])),
    ('neural_network', Pipeline([
        ('scaler', StandardScaler()),
        ('model', NNClassifier)
    ]))
]

# Define the hyperparameter grid for each model
hyperparameters = {
    'logistic_regression': { 
        'model__penalty': [None, 'l2'],
    },
    'random_forest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 2, 5],
        'model__min_samples_leaf': [1, 2, 5]
    },
    'gradient_boosting': {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0,1, 0.5],
        'model__max_depth': [2, 3, 4]
    },
    'support_vector_machine': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['rbf']
    },
    'neural_network': {
        'model__lr': [0.05, 0.1],
        'model__module__num_units': [10, 20],
        'model__module__dropout': [0, 0.5],
    },
}
# -

# Train the models given the different hyperparameter constellations and store the average F1-Score on the validation folds for later comparison.

now = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
with open(f'df_model_classification_{now}.csv', 'w') as csvFile:
    fieldnames = ['MODEL', 'SAMPLING', 'PARAMS', 'MEAN_SCORE']
    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    writer.writeheader()
    for model_name, pipeline in tqdm([pipelines[4]]):
        if model_name=='neural_network':
            n_jobs=None # in case of neural network do not use parallelization but use GPU for acceleration
        else:
            n_jobs=-1   # distribute training jobs across all available cores
        for strategy, (X, y) in [('No resampling (all features)', (X_train, y_train)), 
                                 ('No resampling (inference features)', (X_train_reduced, y_train)), 
                                 ('Oversampling (all features)', (X_train_oversampled, y_train_oversampled)),
                                 ('Oversampling (inference features)', (X_train_reduced_oversampled, y_train_reduced_oversampled)),
                                 ('Undersampling (all features)', (X_train_undersampled, y_train_undersampled)),
                                 ('Undersampling (inference features)', (X_train_reduced_undersampled, y_train_reduced_undersampled)),
                                 ]:
            
            grid_search = GridSearchCV(pipeline, hyperparameters.get(model_name), cv=3, scoring=fscore_tier, verbose=0, n_jobs=n_jobs)
            
            if model_name=='neural_network':
                X = X.to_numpy()
                # Handle the case where only the reduced features are used by padding X with zeros
                # Check the second dimension of X
                second_dim = X.shape[1]
                # Check if the second dimension is smaller than original feature space
                if second_dim < 97:
                    # Calculate the number of zeros to add
                    num_zeros = 97 - second_dim
                    # Pad X with zeros along the second dimension
                    X = np.pad(X, ((0, 0), (0, num_zeros)), mode='constant')
                    
                # Obtain numeric representation of target variable
                y = pd.factorize(y)[0] # 'Basic': 0, 'Premium': 1, 'Plus': 2
                
            grid_search.fit(X, y)
            
            score = grid_search.cv_results_['mean_test_score']
            params = grid_search.cv_results_['params']
            
            for i in range(score.shape[0]):
                writer.writerow({
                'MODEL': model_name, 
                'SAMPLING': strategy,
                'PARAMS': params[i], 
                'MEAN_SCORE': score[i]    
                })
                csvFile.flush()

df_model = pd.read_csv('df_model_classification_29-04-24_175731.csv', sep=",")

# For each model look at the best performing specification model specification (hyperparameter set).

df_model.sort_values('MEAN_SCORE', ascending=False).groupby(['MODEL', 'SAMPLING']).head(1).sort_values(['MODEL', 'MEAN_SCORE'], ascending=False).style

md(f"""The best performing model is a {df_model.sort_values('MEAN_SCORE', ascending=False).iloc[0].MODEL} with an average F1-Score of {round(df_model.sort_values('MEAN_SCORE', ascending=False).iloc[0].MEAN_SCORE*100,2)}% on the validation folds. Little surprisingly, oversampling strongly helps the model to learn from the minority groups since for all models (except the neural network) the performance is best when being trained on the oversampled training data. Moreover, and also expected, the model performs better when being exposed to the entire feature set. The crucial question is how well the best performing model performs when being confronted with the real-world distribution of the target variable. For this purpose, we test the best model's performance on the test set. We do so by using the entire feature set assuming that if the model does not generalize well on all features it will also not generalize on the reduced inference features.""")

# +
# Define the best performing model
pipeline = Pipeline(steps=[
    ('best_model', RandomForestClassifier(max_depth=None, min_samples_leaf=1, n_estimators=300, n_jobs=-1, random_state=333))
])

# Train on the entire oversampled training data
pipeline.fit(X_train_oversampled, y_train_oversampled)

# Make predictions for the test observations
y_pred = pipeline.predict(X_test)

# Return performance metrics
print(classification_report(y_true=y_test, y_pred=y_pred))
# -

# We can see that the best performing model does not generalize well:
# - from the 105 'Plus'|482 'Premium' products in the test data only 3|220 have been detected by the classifier (Recall = 3/105 = 0.03 | 220/482 = 0.46)
# - for 10 vehicles in the test data the classifier has predicted that they are 'Plus' products. Only 3 of them is indeed a 'Plus' article (Precision = 3/10 = 0.30)

# **This means that when the classifier is confronted with the real distribution of the target variable it does not help in reliably differentiating between 'Basic', 'Premium' and 'Plus' articles.**

# These results come with many caveats:
# - zero knowledge about the actual business case, i.e. no exchange with business unit, no information on the meaning/logic of different product tier classes
# - limited knowledge about the features and about which features are available at inference from a process perspective 
# - results derived under time constraints
#
# In a real-world scenario I would undergo the following next steps:
# - get in close exchange with business unit trying to better understand the actual use case and the meaning/logic of the target variable `product_tier`
# - try to understand whether a differentiation between 'Basic' vs. 'non-Basic' (i.e. 'Plus' *or* 'Premium') is helpful from a business perspective - initial results suggest that training such a binary classification model will be a more successful endeavor 
# - gather additional observations and/or features if possible
# - spend more time on understanding/engineering features
# - implement further approaches to handle imbalanced data

# ## Regression Model

# **Is it possible to predict detail views from the information given in the other columns?**

# To tackle this question we aim at developing a regression model that allows to predict how often a product's detail views are clicked by users based on other product characteristics and possibly information on underlying advertising campaigns. In the correlation heatmap above we have seen that `search_views` and `detail_views` are strongly positively correlated, i.e. the more often a product is shown to users the more often users click on the product. This is little surprising and means that `search_views` would be a strong predictor for `detail_views` if a model was trained on all other columns including `search_views`. From a business perspective it seems to be a more relevant use case to predict the number of times a product is clicked *normalized* by the number of times it is suggested to the customer, i.e. the click through rate `ctr`. Predicting click through rates instead of the number of detail views can be interesting for several reasons:
# - Optimizing advertising campaigns: By identifying which ads or ad appearances are more likely to generate higher click-through rates, businesses can allocate their *fixed* advertising budget more effectively. This helps in maximizing the effectiveness of advertising efforts.
# - Measuring campaign performance: Predicting click-through rates allows businesses to measure the performance of their marketing campaigns. By comparing predicted click-through rates with actual click-through rates, it is possible again to evaluate the *effectiveness* of a campaign.
# - Improving user engagement: Pairing product information with user demographics, predicting click-through rates may provide insights into user engagement with products or services. By predicting click-through rates, businesses can identify product but also customer characteristics that influence user behavior and tailor their offerings to better meet user needs and user engagement.

# Let us get a better feeling how the target variable `detail_views` is distributed across the different features. For this purpose we categorize the target variable.

df['detail_views_cat'] = pd.cut(df['detail_views'], bins=[0, 50, 100, 20000], include_lowest=True).astype(str)

plot_describe(df, df.drop(columns=['article_id', 'created_date', 'deleted_date', 'detail_views']).columns, log_dict={'search_views': True},
              hue='detail_views_cat')

# Findings:
# - As expected `search_views` but also `stock_days` seem to entail strong discriminatory power for predicting the number of times a product is clicked by the customer. Generally speaking, the longer the product is listed on the platform and the more heavily it is promoted, the more often it will be clicked.
#

# Let's create the same visualization for `ctr`.

df['ctr_cat'] = pd.cut(df['ctr'], bins=[0, 0.04, 0.07, 2], include_lowest=True).astype(str)

plot_describe(df, df.drop(columns=['article_id', 'created_date', 'deleted_date', 'ctr', 'detail_views_cat', 'detail_views', 'search_views']).columns, 
              log_dict={
                  'search_views': True,
                  'detail_views': True
              },
              hue='ctr_cat', ncols=2)

# Findings:
# - from a pure visual and bivariate inspection it does not seem that the features show a high discriminatory power in distinguishing product's with different click through rate classes because the feature distributions across the three classes look quite similar.

# In the following, we train commonly used ML/DL regression models for the purpose of testing whether the number of times users view the details of product can be predicted. Our setup is as follows:
# - since it could also be (and possibly is even more) interesting to predict click-through rates we train two models <br>
#   (1) one to predict `detail_views` (exclude `ctr` from feature set as it is calculated from `detail_views`)<br>
#   (2) the other one to predict `ctr` (exclude `search_views` and `detail_views` from feature set since they are used to calculate `ctr`)<br>
# - we aim at approximating the best set of hyperparameters on validation sets using cross-validation in order to prevent overfitting.
# - since the target variable is a count variable that follows a Poisson distribution, we prefer models that can be tuned on minimizing Poisson deviance. Poisson deviance specifically compares the observed counts with the predicted counts and is thus most suited for this use case.
# - we evaluate the model performance on the mean Poisson deviance and the mean absolute error (MAE). To assess how well the models perform we compare MAE against the MAE of a naive mean prediction.
# - for scale sensitive models we scale the features by subtracting their mean and dividing by their standard deviation.

# +
# Features
X = df.drop(columns=['article_id', 'detail_views', 'created_date', 'deleted_date', 'first_registration_year', 'ctr', 'detail_views_cat', 'ctr_cat'])
# One-hot encoding of make_name and product_tier
X = pd.get_dummies(X, columns=['product_tier'], drop_first=True, dtype=np.float32) # car make 'AC' serves as reference group
X = pd.get_dummies(X, columns=['make_name'], drop_first=True, dtype=np.float32)    # product tier 'Basic' serves as reference group
X = X.astype(np.float32)                                                           # create common dtype to ease matrix manipulation

# Target
y = df['detail_views']
# -

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

# +
# Drop features that are not available for ctr prediction
X_train_ctr = X_train.drop(columns=['search_views'])
X_test_ctr = X_test.drop(columns=['search_views'])

# ctr Target
y_train_ctr = df['ctr'].loc[y_train.index]
y_test_ctr = df['ctr'].loc[y_test.index]
# -

# Define a feed-forward neural network (2 layers) for the regression task based on PyTorch (see util.py)
NNRegressor = NeuralNetRegressor(
    RegressorModule,
    max_epochs=20,
    lr=0.1,
    device=device,
    train_split=False,   # disable the skorch-internal train-validation split as GridSearchCV conducts the train-validation splits
    verbose=0,           # Verbosity is steered via GridSearchCV
)

# +
score_mae = make_scorer(mean_absolute_error, greater_is_better=False)
score_deviance = make_scorer(mean_poisson_deviance, greater_is_better=False)
scoring = {'MAE': score_mae, 'Poisson_deviance': score_deviance}

# Define the pipeline for each model
pipelines = [
        ('poisson_regression', Pipeline([
            ('model', PoissonRegressor(                              
                max_iter=600,
                solver='newton-cholesky'))                 
             ])),
    ('random_forest', Pipeline([
        ('model', RandomForestRegressor(random_state=333))
    ])),
        ('gradient_boosting', Pipeline([
            ('scaler', StandardScaler()),
            ('model', HistGradientBoostingRegressor(
                loss='poisson',                         # Poisson deviance as loss function
                random_state=333))     
        ])),
        ('neural_network', Pipeline([
            ('scaler', StandardScaler()),
            ('model', NNRegressor)
        ])),
    ]

# Define the hyperparameter grid for each model
hyperparameters = {
    'poisson_regression': { 
        'model__alpha': [0.1, 1, 10],
        'model__warm_start': [True, False]
    },
    'random_forest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 2, 5],
        'model__min_samples_leaf': [1, 2, 5]
    },
    'gradient_boosting': {
        'model__max_iter': [100, 200, 300],
        'model__learning_rate': [0.01, 0,1, 0.5],
        'model__max_depth': [None, 2, 4]
    },
    'neural_network': {
        'model__lr': [0.05, 0.1],
        'model__module__num_units': [10, 20, 30]
    },
}

# + jupyter={"outputs_hidden": true}
now = datetime.datetime.now().strftime("%d-%m-%y_%H%M%S")
with open(f'df_model_prediction_{now}.csv', 'w') as csvFile:
    fieldnames = ['MODEL', 'TARGET', 'PARAMS', 'MEAN_SCORE']
    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    writer.writeheader()
    for model_name, pipeline in tqdm(pipelines):
        if model_name=='neural_network':
            n_jobs=None
        else:
            n_jobs=-1
        for strategy, (X, y) in [('detail_views', (X_train, y_train)), 
                                 ('ctr', (X_train_ctr, y_train_ctr)), 
                                 ]:
            grid_search = GridSearchCV(pipeline, hyperparameters.get(model_name), cv=3, scoring=scoring, refit='Poisson_deviance', verbose=2, n_jobs=n_jobs)
            if model_name=='neural_network':
                X = X.to_numpy()      
                # Handle the case where where ctr is predicted with one feature less
                # Check the second dimension of X
                second_dim = X.shape[1]
                # Check if the second dimension is smaller than original feature space
                if second_dim < 97:
                    # Calculate the number of zeros to add
                    num_zeros = 97 - second_dim
                    # Pad X with zeros along the second dimension
                    X = np.pad(X, ((0, 0), (0, num_zeros)), mode='constant')
                y = y.astype(np.float32).to_numpy()   
            grid_search.fit(X, y)
            score = grid_search.cv_results_['mean_test_MAE']*-1
            params = grid_search.cv_results_['params']
            for i in range(score.shape[0]):
                writer.writerow({
                'MODEL': model_name, 
                'TARGET': strategy,
                'PARAMS': params[i], 
                'MEAN_SCORE': score[i]    
                })
                csvFile.flush()
# -

# For each model look at the best performing specification for both target variables.

df_model = pd.read_csv('df_model_prediction_30-04-24_230255.csv')

df_model.sort_values('MEAN_SCORE', ascending=True).groupby(['MODEL', 'TARGET']).head(1).sort_values(['MODEL', 'MEAN_SCORE'], ascending=False).style

md(f"""The best performing model is a {df_model.loc[df_model.TARGET=='detail_views'].sort_values('MEAN_SCORE', ascending=True).iloc[0].MODEL}/{df_model.loc[df_model.TARGET=='ctr'].sort_values('MEAN_SCORE', ascending=True).iloc[0].MODEL} with an MAE of {round(df_model.loc[df_model.TARGET=='detail_views'].sort_values('MEAN_SCORE', ascending=True).iloc[0].MEAN_SCORE)}/{round(df_model.loc[df_model.TARGET=='ctr'].sort_values('MEAN_SCORE', ascending=True).iloc[0].MEAN_SCORE*100, 3)} views/percentage points on the validation folds for the task of predicting the number of user clicks on an article/the click through rate. Let's see how well the model performs on the hold-out test data.""")

# +
# Define the best performing model for detail_views prediction
pipeline = Pipeline(steps=[
    ('best_model', RandomForestRegressor(max_depth=None, min_samples_leaf=5, n_estimators=300, n_jobs=-1, random_state=333))
])

# Train on the entire detail_views feature set
pipeline.fit(X_train, y_train)

# Make predictions for the test observations
y_pred = pipeline.predict(X_test)

# Return performance metrics
md(f"""The best performing model makes an out-of-sample prediction error of {round(mean_absolute_error(y_test, y_pred))} views. This means that on average the model under-/overestimates the number of detail views by 37 views.""")

# +
# Define the best performing model for ctr prediction
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('best_model', HistGradientBoostingRegressor(learning_rate= 0.5, max_depth=2, max_iter=100, loss='poisson', random_state=333))
])

# Train on the entire ctr feature set
pipeline.fit(X_train_ctr, y_train_ctr)

# Make predictions for the test observations
y_pred = pipeline.predict(X_test_ctr)

# Return performance metrics
md(f"""The best performing model's out-of-sample prediction error is {round(mean_absolute_error(y_test_ctr, y_pred), 3)}. This means that on average the model under-/overestimates the article's click through rate by 2.3 percentage points.""")
# -

# Both models seem to generalize well. In order to judge how well the models perform, i.e. how severe their error is, we compare the models' error against the error of a simple mean predictor.

# Error of naive predictor for detail_views
md(f"""Prediction error mean predictor: {round(mean_absolute_error(y_test, np.repeat(y_train.mean(), len(y_test))))} views""")

# Error of naive predictor for ctr
md(f"""Prediction error mean predictor: {round(mean_absolute_error(y_test_ctr, np.repeat(y_train_ctr.mean(), len(y_test_ctr)))*100,3)} percentage points""")

# **This means that the prediction of the number of detail views works decently enough based on information about the product and possibly based on information on marketing campaigns that drive search views. It is recommended to investigate this endeavor further. Prediction of click-through rates does not look promising based on the available features.**

# These results come with many caveats:
# - no knowledge about the actual business case, i.e. no exchange with business unit, no information on the underlying process/logic how products are prioritized in searches (is there any selection in place? any campaign in the training data?)
# - Prediction error of the current model is still large - further investigation/fine-tuning of model required for a production ready model
# - results derived under time constraints
#
# In a real-world scenario I would undergo the following next steps:
# - get in close exchange with business unit trying to better understand the actual use case as to why prediction of `detail_views` is interesting/helpful
# - gather additional observations and/or features if possible
# - spend more time on understanding/engineering features
# - experiment with further algorithms/conduct further fine-tuning to improve prediction performance
