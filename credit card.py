#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#importing dataset
df = pd.read_csv("C:\\Users\\Jenit\\Desktop\\creditcard.csv")
print(df.head(5))

df = df.drop("Time", axis=1)

#preprocessing package
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

#standard scaling
df['std_Amount'] = scaler.fit_transform(df['Amount'].values.reshape (-1,1))

#removing Amount
df = df.drop("Amount", axis=1)

sns.countplot(x="Class", data=df)
plt.show()

import imblearn
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy=0.5)

cols = df.columns.tolist()
cols = [c for c in cols if c not in ["Class"]]
target = "Class"

#define X and Y
X = df[cols]
Y = df[target]

# Including a PCA conversion of the X variables; as there are some high correlation between some of the varibles
# In that way, co-linearity can be avoided

from sklearn.decomposition import PCA
pca=PCA(n_components=15, svd_solver='full')
pca = pca.fit_transform(X)

J = pd.DataFrame(pca)

#undersample
X_under, Y_under = undersample.fit_resample(X, Y)
X_pca, Y_pca = undersample.fit_resample(J, Y)

from pandas import DataFrame
test = pd.DataFrame(Y_under, columns = ['Class'])

#visualizing undersampling results
fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
sns.countplot(x="Class", data=df, ax=axs[0])
sns.countplot(x="Class", data=test, ax=axs[1])

fig.suptitle("Class repartition before and after undersampling")
a1=fig.axes[0]
a1.set_title("Before")
a2=fig.axes[1]
a2.set_title("After")
plt.show()

#splitting dataset into train data and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=1)

# Let's create a train_test_split for the data with PCA:

X_trainP, X_testP, y_trainP, y_testP = train_test_split(X_pca, Y_pca, test_size=0.2, random_state=1)

#importing packages for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# algorithm 1 : logistic regression

#train the model
model1 = LogisticRegression(random_state=2)
logit = model1.fit(X_trainP, y_trainP)

#predictions
y_pred_logit = model1.predict(X_testP)

#scores
print("Accuracy For Logistic Regression : ", metrics.accuracy_score(y_testP, y_pred_logit))
print("Precision For Logistic Regression : ", metrics.precision_score(y_testP, y_pred_logit))
print("Recall For Logistic Regression : ", metrics.recall_score(y_testP, y_pred_logit))
print("F1 Score For Logistic Regression : ", metrics.f1_score(y_testP, y_pred_logit))

#print CM
matrix_logit = confusion_matrix(y_test, y_pred_logit)
cm_logit = pd.DataFrame(matrix_logit, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_logit, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix For Logistic Regression")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()

# algorithm 2 : support vector machine

#train the model
model2 = SVC(probability=True, random_state=2)
svm = model2.fit(X_trainP, y_trainP)

#predictions
y_pred_svm = model2.predict(X_testP)

#scores
print("Accuracy For Support Vector Machine : ", metrics.accuracy_score(y_testP, y_pred_svm))
print("Precision For Support Vector Machine : ", metrics.precision_score(y_testP, y_pred_svm))
print("Recall For Support Vector Machine : ", metrics.recall_score(y_testP, y_pred_svm))
print("F1 Score For Support Vector Machine : ", metrics.f1_score(y_testP, y_pred_svm))

#CM matrix
matrix_svm = confusion_matrix(y_testP, y_pred_svm)
cm_svm = pd.DataFrame(matrix_svm, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_svm, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix For Support Vector Machine")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

# algorithm 3 : random forest

#train the model
model3 = RandomForestClassifier(random_state=2)
rf = model3.fit(X_trainP, y_trainP)

#predictions
y_pred_rf = model3.predict(X_testP)

#scores
print("Accuracy For Random Forest : ", metrics.accuracy_score(y_testP, y_pred_rf))
print("Precision For Random Forest : ", metrics.precision_score(y_testP, y_pred_rf))
print("Recall For Random Forest : ", metrics.recall_score(y_testP, y_pred_rf))
print("F1 Score For Random Forest : ", metrics.f1_score(y_testP, y_pred_rf))

#CM matrix
matrix_rf = confusion_matrix(y_testP, y_pred_rf)
cm_rf = pd.DataFrame(matrix_rf, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_rf, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix For Random Forest"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

model_results = [metrics.accuracy_score(y_testP, y_pred_logit)*100, metrics.accuracy_score(y_testP, y_pred_svm)*100, metrics.accuracy_score(y_testP, y_pred_rf)*100]
model_names = ['Logistic Regression', 'Support Vector Machine', 'Random Forest']

# Bar Plot for model evaluation
plt.bar(model_names, model_results)
plt.xlabel('Machine Learning Algorithms')
plt.ylabel("Accuracy (in %)")
plt.title('Model Evaluation')
plt.show()
