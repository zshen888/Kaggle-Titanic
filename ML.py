import pandas as pd
import numpy as np

# Load Data
import os
os.getcwd()
os.chdir('./Titanic - ML from Disaster/data')

train = pd.read_csv('clean_train.csv')
test = pd.read_csv('clean_test.csv')

from sklearn.model_selection import StratifiedKFold
skfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

train["Survived"] = train["Survived"].astype(int)
y_train = train["Survived"]
x_train = train.drop(labels=["Survived"], axis=1)



from sklearn.model_selection import GridSearchCV


# 1 AdaBoost
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
param_grid = [
    {
        'n_estimators': list(np.arange(50, 200, 50)),
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],
        'random_state': [42]
             }
]
ada_clf = AdaBoostClassifier()
grid_search = GridSearchCV(ada_clf, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)
grid_search.best_params_
grid_search.best_score_



# 2 Random Forest
param_grid = [
    {
        'n_estimators': list(np.arange(50, 500, 50)),
        'max_features': ['auto','sqrt'],
        'max_depth': list(np.arange(2, 10, 2)),
        'bootstrap': [False],
        'random_state': [42],
        'n_jobs': [-1]
     },
    {
        'n_estimators': list(np.arange(50, 500, 50)),
        'max_features': ['auto','sqrt'],
        'max_depth': list(np.arange(2, 12, 2)),
        'bootstrap': [True],
        'oob_score': [True],
        'random_state': [42],
        'n_jobs': [-1]
    }
]
rf_clf = RandomForestClassifier()
grid_search = GridSearchCV(rf_clf, param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train, y_train)
grid_search.best_params_
grid_search.best_score_



# 3 Logistic Regression
from sklearn.linear_model import SGDClassifier
param_grid = [
    {
        'loss': ['log'],
        'penalty': ['elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'l1_ratio': list(np.arange(0, 1, 0.05)),    # 0 corresponds to L2, 1 to L1
        'random_state': [42],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
        'eta0': [0.001, 0.01],
        'tol': [1e-3],
        'n_jobs': [-1]
     }
]
log_clf = SGDClassifier()
grid_search = GridSearchCV(log_clf, param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train, y_train)
grid_search.best_params_
grid_search.best_score_


# 4 SVM
from sklearn.svm import SVC
param_grid5 = [
    {
        'C': list(np.arange(0.05, 1, 0.05)),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [1, 2, 3],
        'probability': [True],
        'random_state': [42]
     }
]
svm_clf = SVC()
grid_search = GridSearchCV(svm_clf, param_grid5, cv=10, scoring='accuracy')
grid_search.fit(x_train, y_train)
grid_search.best_params_
grid_search.best_score_



# 5 KNN
param_grid6 = [
    {
        'n_neighbors': list(np.arange(5, 30, 5)),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        'leaf_size': list(np.arange(10, 100, 10)),
        'p': [1,2]
     }
]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid6, cv=10, scoring='accuracy')


#############################################################################################

ada_clf = AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
rf_clf = RandomForestClassifier(bootstrap=False, max_depth=8, max_features='auto', n_estimators=400, n_jobs=-1, random_state=42)
log_clf = SGDClassifier(alpha=0.001, eta0=0.001, l1_ratio=0.6, learning_rate='optimal', loss='log',n_jobs=-1, penalty='elasticnet',random_state=42,tol=0.001)
svm_clf = SVC(C=0.4, degree=1, kernel='rbf', probability=True, random_state=42)
knn_clf = KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, n_neighbors=25, p=2, weights='uniform')



from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf),
                ('ada', ada_clf),
                ('ext', ext_clf),
                ('lr', lr_clf),
                ('svm', svm_clf),
                ('knn', knn_clf),
                ('sgd', sgd_clf)
         ],
    voting='soft'
)

voting_clf.fit(x_train, y_train)


from sklearn.metrics import accuracy_score
for clf in (rf_clf, ada_clf, ext_clf, lr_clf, svm_clf, knn_clf, sgd_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



y_pred = voting_clf.predict(test_df.drop(['PassengerId'], axis=1).copy())


##########################################################################################

Submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred})
Submission.to_csv("Submission.csv", index=False)












from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)



from sklearn.model_selection import cross_val_score, cross_val_predict
cross_val_score(sgd_clf, x_train, y_train, cv=10, scoring='accuracy')
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=10)


# Precision, Recall, F1 Score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print(confusion_matrix(y_train, y_train_pred))
print('Precision', precision_score(y_train, y_train_pred))
print('Recall', recall_score(y_train, y_train_pred))
print('F1 Score', f1_score(y_train, y_train_pred))


# Precision Recall VS. Threshold
y_score = cross_val_predict(sgd_clf, x_train, y_train, cv=10, method='decision_function')

from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_train, y_score)

def prec_rec_vs_thres(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], 'b--', label='Precision')
    plt.plot(threshold, recall[:-1], 'g-', label='Recall')

    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])

prec_rec_vs_thres(precision, recall, threshold)
plt.show()


# Precision VS. Recall
def prec_vs_rec(precision, recall):
    plt.plot(recall, precision, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
prec_vs_rec(precision, recall)
plt.show()

# ROC
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train, y_score)

def roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
roc_curve(fpr, tpr)
plt.show()

roc_auc_score(y_train, y_score)
