import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.head())
print(train_data.info())

print(test_data.head())
print(test_data.info())


# KOD: https://www.kaggle.com/tososomaru/mobile-price-classification-pca#Features-extraction-(PCA)
#Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report, confusion_matrix


def testing_model(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    conv_matrix = confusion_matrix(y_test, y_predict)
    sns.heatmap(conv_matrix, cmap='coolwarm', annot=True, fmt='.0f')
    print('                 Confusion matrix \n')


    print('                 Classification report \n')
    print(classification_report(y_test, y_predict))


X_train = train_data.iloc[:, :-1]
Y_train = train_data['price_range']

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.3, random_state=10)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
norm_x_train = scaler.transform(x_train)
norm_x_test = scaler.transform(x_test)

tree = DecisionTreeClassifier(random_state=10)
testing_model(norm_x_train, norm_x_test, y_train, y_test, tree)


# Features extraction (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(X_train)
pca_features = pca.transform(X_train)

x_train, x_test, y_train, y_test = train_test_split(pca_features, Y_train, test_size = 0.3, random_state=10)

scaler = preprocessing.StandardScaler()
scaler.fit(pca_features)
norm_x_train = scaler.transform(x_train)
norm_x_test = scaler.transform(x_test)


# KOD: https://www.kaggle.com/tanyildizderya/mobile-price-classification#KNN
# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)

print(knn.score(x_test,y_test))

pred = knn.predict(x_test)

error_rate = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=5)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


plt.show()