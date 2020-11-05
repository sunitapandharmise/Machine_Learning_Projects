
# ----------------------- Sunita Pandharmise  ----------------------------------- #

# ----------------------- Import Libraries ----------------------------------------------- #

# import linear algebra and data manipulation libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import standard visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')

# import pre-processing libraries
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# import machine learning libraries
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# import split and metrics libraries
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

# tools for hyper parameters search
from sklearn.model_selection import GridSearchCV

# Feature Selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# ----------------------- Pre-processing function ----------------------------------------------- #

def performPreprocessing(df):

    # Step 1
    # check if data column has any outliers
    # sns.boxplot(data =pd.DataFrame(df))
    # plt.show()

    # Step 2
    # check if data column has any missing values
    # print(df.isna().sum())

    # Step 3
    # Handling Categorical Data
    # list of columns which categorical
    colname = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']
    le = {}
    for x in colname:
        le[x] = preprocessing.LabelEncoder()  # it will create labels for each column
    for x in colname:
        df[x] = le[x].fit_transform(df.__getattr__(x))  # the labels created will get assigned it to our actual Dataset

    # Step 4
    # Scaling Data
    X = df.iloc[:, :-1]     # creating feature train data
    Y = df.iloc[:, -1]      # creating class label data
    # to get the uniformity using standard scalar on feature train data
    scaler = StandardScaler()
    scaler.fit_transform(X)
    return X, Y

# ----------------------- Model function ----------------------------------------------- #

def runModels(X_train, Y_train):

    # XGBoost Model
    xgb = XGBClassifier()
    xgb_scores = cross_val_score(xgb, X_train, Y_train, cv=4) # cross-fold validation
    xgb_mean = xgb_scores.mean()

    # SVC
    svc_clf = SVC()
    svc_scores = cross_val_score(svc_clf, X_train, Y_train, cv=4) # cross-fold validation
    svc_mean = svc_scores.mean()

    # KNearestNeighbors
    knn_clf = KNeighborsClassifier()
    knn_scores = cross_val_score(knn_clf, X_train, Y_train, cv=4) # cross-fold validation
    knn_mean = knn_scores.mean()

    # Decision Tree
    tree_clf = tree.DecisionTreeClassifier()
    tree_scores = cross_val_score(tree_clf, X_train, Y_train, cv=4) # cross-fold validation
    tree_mean = tree_scores.mean()

    # Gradient Boosting Classifier
    grad_clf = GradientBoostingClassifier()
    grad_scores = cross_val_score(grad_clf, X_train, Y_train, cv=4) # cross-fold validation
    grad_mean = grad_scores.mean()

    # Random Forest Classifier
    rand_clf = RandomForestClassifier(n_estimators=10)
    rand_scores = cross_val_score(rand_clf, X_train, Y_train, cv=4) # cross-fold validation
    rand_mean = rand_scores.mean()

    # NeuralNet Classifier
    neural_clf = MLPClassifier(alpha=1)
    neural_scores = cross_val_score(neural_clf, X_train, Y_train, cv=4) # cross-fold validation
    neural_mean = neural_scores.mean()

    # Naives Bayes
    nav_clf = GaussianNB()
    nav_scores = cross_val_score(nav_clf, X_train, Y_train, cv=4) # cross-fold validation
    nav_mean = neural_scores.mean()

    # Create dictionary to store each model accuracy
    dict_model = {'Classifiers': ['XGBoost', 'SVC', 'KNN', 'Decision Tree', 'Gradient Boosting Classifier',
                          'Random Forest Classifier', 'Neural Classifier', 'Naives Bayes'],
          'Crossval Mean Scores': [xgb_mean, svc_mean, knn_mean, tree_mean, grad_mean, rand_mean, neural_mean,
                                   nav_mean]}

    # Create dataframe from using above dictionary
    result_df = pd.DataFrame(data=dict_model)
    # sort the models in ascending order based on their accuracy
    result_df = result_df.sort_values(by='Crossval Mean Scores', ascending=False)
    # return the resulted dataframe
    return result_df

# ----------------------- Classification report function ----------------------------------------------- #

def report(X_train, Y_train):
    # XGBoost Model
    xgb = XGBClassifier()
    y_pred = cross_val_predict(xgb, X_train, Y_train, cv=4)
    print('XGBClassifier:\n',classification_report(Y_train, y_pred))

    # SVC
    svc_clf = SVC()
    y_pred = cross_val_predict(svc_clf, X_train, Y_train, cv=4)
    print('SVC:\n',classification_report(Y_train, y_pred))

    # KNearestNeighbors
    knn_clf = KNeighborsClassifier()
    y_pred = cross_val_predict(knn_clf, X_train, Y_train, cv=4)
    print('KNeighborsClassifier:\n',classification_report(Y_train, y_pred))

    # Decision Tree
    tree_clf = tree.DecisionTreeClassifier()
    y_pred = cross_val_predict(tree_clf, X_train, Y_train, cv=4)
    print('DecisionTreeClassifier:\n',classification_report(Y_train, y_pred))

    # Gradient Boosting Classifier
    grad_clf = GradientBoostingClassifier()
    y_pred = cross_val_predict(grad_clf, X_train, Y_train, cv=4)
    print('GradientBoostingClassifier:\n',classification_report(Y_train, y_pred))

    # Random Forest Classifier
    rand_clf = RandomForestClassifier(n_estimators=10)
    y_pred = cross_val_predict(rand_clf, X_train, Y_train, cv=4)
    print('RandomForestClassifier:\n', classification_report(Y_train, y_pred))

    # NeuralNet Classifier
    neural_clf = MLPClassifier(alpha=1)
    y_pred = cross_val_predict(neural_clf, X_train, Y_train, cv=4)
    print('MLPClassifier:\n', classification_report(Y_train, y_pred))

    # Naives Bayes
    nav_clf = GaussianNB()
    y_pred = cross_val_predict(nav_clf, X_train, Y_train, cv=4)
    print('GaussianNB:\n', classification_report(Y_train, y_pred))


# ----------------------- Feature Selection function for Pre-processing ----------------------------------------------- #

def featureSelection(X, Y):
    # Build a forest and compute the feature importance
    forest = RandomForestClassifier(n_estimators=250, random_state=0)
    forest.fit(X, Y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for index in range(X.shape[1]):
        print("%d. Importance of feature %d (%f)" % (index + 1, indices[index], importances[indices[index]]))

# ----------------------- Hyper-Parameter function ----------------------------------------------- #

def hyperParameter(X, Y):
    # taking the best performing models from above and applying hyper parameter optimization on it

    # 1. Gradient Boosting Classifier
    hyper_var_gbc = [{'n_estimators': list(range(1, 20)), 'max_depth': [8], 'min_samples_split': [10]}]
    gsearch = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=hyper_var_gbc, n_jobs=4, cv=10)
    gsearch.fit(X, Y)
    print("Best parameteres for Gradient Boosting Classifier are ", gsearch.best_params_, "with a best score of ",
          gsearch.best_score_)

    # 2. XGBoost Classifier
    hyper_var_xgb = [{'n_estimators': list(range(1, 20)), 'max_depth': [8], 'min_child_weight': [1]}]
    xgsearch = GridSearchCV(estimator=XGBClassifier(), param_grid=hyper_var_xgb, n_jobs=4, cv=10)
    xgsearch.fit(X, Y)
    print("Best parameteres for XGBoost are ", xgsearch.best_params_, "with a best score of ", xgsearch.best_score_)

    # 3. Random Forest Classifier
    hyper_var_random = [{'n_estimators': list(range(1, 20)), 'max_depth': [8], 'min_samples_split': [10]}]
    randsearch = GridSearchCV(estimator=RandomForestClassifier(), param_grid=hyper_var_random, n_jobs=4, cv=10)
    randsearch.fit(X, Y)
    print("Best parameteres for Random Forest Classifier are ", randsearch.best_params_, "with a best score of ",
          randsearch.best_score_)

# ----------------------- Main function ----------------------------------------------- #

def main():
    # read the data using pandas
    bank_data = pd.read_csv("E:/Study/AI_Sem1/ML/bank.csv", delimiter=",")

    # Run pre-processing on data frame
    feature_train, class_label = performPreprocessing(bank_data)

    # ----------------- create baseline models ------------------------- #

    # split the data into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(feature_train, class_label, test_size=0.2, random_state=11)

    print('-------------------------------- Baseline Models before Feature Selection Pre-processing --------------------------------------')

    # call run model function without feature selection pre-processing
    model_df = runModels(X_train, Y_train)
    print(model_df)

    # call metrics function for one model without feature selection pre-processing
    report(X_train, Y_train)

    # ----------------- create models after applying feature selection technique in pre-processing ------------------------- #

    # call feature selection function before spliting the data
    featureSelection(feature_train, class_label)

    # remove features which has lower importance ranking ie, marital, default and loan columns
    feature_train1 = feature_train.drop(['marital', 'default', 'loan'], axis=1)

    # split the data into test and train
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(feature_train1, class_label, test_size=0.2, random_state=10)

    print('--------------------------------Models After Feature Selection --------------------------------------')
    # call run model function with feature selection pre-processing
    model_df_feature = runModels(X_train1, Y_train1)
    print(model_df_feature)

    # call metrics function for one model after feature selection
    report(X_train1, Y_train1)

    print('-------------------------------- Hyper Parameter optimization on top 3 models ----------------------------------------')

    hyperParameter(X_train1, Y_train1)

    print('-------------------------------- Research Topic - Feature Selection -----------------------------------')

    # Research - Feature Selection

    print('-------------------------------- 1. Recursive Feature Elimination -------------------------------------')

    # 1. Recursive Feature Elimination
    # using Logistic Regression model to get the score of each feature
    model = LogisticRegression()
    # create the RFE model and select 10 attributes
    rfe = RFE(model, 10)
    rfe = rfe.fit(X_train, Y_train)
    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)

    # plot the ranking
    plt.bar(range(len(rfe.ranking_)), rfe.ranking_)
    plt.show()

    # based on these ranking remove the columns
    feature_train2 = feature_train.drop(['age', 'job', 'balance', 'day', 'duration', 'pdays'], axis=1)
    # split the data into test and train
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(feature_train2, class_label, test_size=0.2, random_state=10)

    # call run model function for Recursive Feature Elimination
    model_df_RFE = runModels(X_train2, Y_train2)
    print(model_df_RFE)

    print('-------------- Hyper Parameter optimization on top 3 models for Recursive Feature Elimination technique ------------------')

    hyperParameter(X_train2, Y_train2)

    print('-------------------------------- 2. Feature Importance -------------------------------------')

    # 2. Feature Importance
    # fit an Extra Trees model to the data
    model = ExtraTreesClassifier()
    model.fit(X_train, Y_train)
    # display the relative importance of each feature
    print('Score values of each fetaure: ', model.feature_importances_)

    # plot the scores
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()

    # based on these score remove the columns
    feature_train3 = feature_train.drop(['marital', 'education', 'default', 'housing', 'loan', 'duration', 'pdays', 'previous'], axis=1)
    # split the data into test and train
    X_train3, X_test3, Y_train3, Y_test3 = train_test_split(feature_train3, class_label, test_size=0.2, random_state=10)

    # call run model function for Feature Importance technique
    model_df_Feature_Importance = runModels(X_train3, Y_train3)
    print(model_df_Feature_Importance)

    print('------------------- Hyper Parameter optimization on top 3 models for Feature Importance technique ----------------------')

    hyperParameter(X_train3, Y_train3)

    print('-------------------------------- 3. Sequential Feature Selector -------------------------------------')

    sfs1 = SFS(KNeighborsClassifier(), k_features=10, forward=True, floating=False, verbose=2, scoring='accuracy', cv=0)
    sfs1 = sfs1.fit(X_train, Y_train)
    print('Indices of the 10 best features: ',sfs1.k_feature_idx_)


    # based on these score remove the columns
    feature_train4 = feature_train.drop(['age', 'job', 'education', 'balance', 'day', 'campaign'], axis=1)
    # split the data into test and train
    X_train4, X_test4, Y_train4, Y_test4 = train_test_split(feature_train4, class_label, test_size=0.2, random_state=10)

    # call run model function for Sequential Feature Selector
    model_df_SFS = runModels(X_train4, Y_train4)
    print(model_df_SFS)

    print('------------------- Hyper Parameter optimization on top 3 models for Sequential Feature Selector ----------------------')

    hyperParameter(X_train4, Y_train4)


# call the main function
main()


