#imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

#example dataset
dataset = pd.DataFrame({
    'end_customer': ['Cliente A', 'Cliente B'],
    'region': ['Estado 1', 'Estado 2'],
    'winner': ['Empresa 1', 'Empresa 2'],
    'status_opty': ['Lost', 'Won'],
    'industry': ['Chemicals', 'Paper'],
    'success_chance': [0.5, 0.7],
    'execution_chance': [0.9, 0.6]
})

#z-score outlier replacement
def replace_outliers_zscore(dataset, imputation_columns, method='median'):
    for col in imputation_columns:
        z_scores = stats.zscore(dataset[col])
        dataset[col + '_zscore'] = np.where(np.abs(z_scores) > 3, dataset[col].median(), dataset[col])

    return dataset

#end customer frequency
customer_frequency = dataset['end_customer'].value_counts()
dataset['customer_frequency'] = dataset['end_customer'].map(customer_frequency)

#encode categorical variables
le = LabelEncoder()
dataset['country_encoded'] = le.fit_transform(dataset['region'])
dataset['winner_encoded'] = le.fit_transform(dataset['winner'])
dataset['status_encoded'] = le.fit_transform(dataset['status_opty'])
dataset['industry_encoded'] = le.fit_transform(dataset['industry'])
dataset['customer_encoded'] = le.fit_transform(dataset['end_customer'])
dataset['execution_encoded'] = le.fit_transform(dataset['execution_chance'])


#data interaction
dataset['chance_status_interaction'] = dataset['success_chance'] * dataset['status_encoded']
dataset['chance_country_interaction'] = dataset['success_chance'] * dataset['country_encoded']
dataset['chance_winner_interaction'] = dataset['success_chance'] * dataset['winner_encoded']
dataset['chance_customer_interaction'] = dataset['success_chance'] * dataset['customer_encoded']
dataset['chance_frequency_interaction'] = dataset['success_chance'] * dataset['customer_frequency']

imputation_columns = ['chance_frequency_interaction','chance_customer_interaction', 'chance_winner_interaction', 'chance_country_interaction',
                      'chance_status_interaction', 'customer_encoded', 'status_encoded', 'industry_encoded', 'execution_encoded',
                      'customer_frequency', 'success_chance', 'winner_encoded', 'country_encoded']

#knn
imputer = KNNImputer(n_neighbors=5)
dataset_knn = pd.DataFrame(imputer.fit_transform(dataset[imputation_columns]), 
                           columns=[col + '_knn' for col in imputation_columns])

#imputed dataset + original one
dataset = pd.concat([dataset, dataset_knn], axis=1)

#call zscore function
dataset = replace_outliers_zscore(dataset, imputation_columns, method='median')

#split features
X = dataset[['chance_frequency_interaction','chance_customer_interaction', 'chance_winner_interaction', 'chance_country_interaction',
             'chance_status_interaction', 'customer_encoded', 'status_encoded', 'industry_encoded', 'execution_encoded',
              'customer_frequency', 'success_chance', 'winner_encoded', 'country_encoded']]

#define target variable
y = (
    (dataset['success_chance'] >= 0.5) &
    (dataset['status_encoded'] != 7.0)
    ).astype(float)

#scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#regularization to avoid overfitting
model = LogisticRegression(C=0.5, penalty='l2', solver='saga', class_weight='balanced')

#train the model
model.fit(X_train, y_train)

#cross-validation to check performance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
average_cv_accuracy = cv_scores.mean()

y_pred = model.predict(X_test)

#calculate precision, recall, F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

#score columns
dataset['Cross_Validation_Accuracy'] = average_cv_accuracy
dataset['Precision'] = precision
dataset['Recall'] = recall
dataset['F1_Score'] = f1

#confusion matrix values
dataset['True_Positive'] = conf_matrix[1, 1]
dataset['True_Negative'] = conf_matrix[0, 0]
dataset['False_Positive'] = conf_matrix[0, 1]
dataset['False_Negative'] = conf_matrix[1, 0]

#predictions
dataset['Sales Opportunity'] = model.predict(X)

#convert results into human-readable labels
dataset['Sales Opportunity'] = dataset['Sales Opportunity'].replace({1: 'Potential', 0: 'Unlikely'})

#remove previously added columns
cols_to_drop = [col for col in dataset.columns if col.endswith('zscore') or col.endswith('knn')]

#drop extra columns from the dataset;
dataset.drop(columns=cols_to_drop, inplace=True)

#output
dataset