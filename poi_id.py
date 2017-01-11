#!/usr/bin/python

#standard modules used throughout
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np
import warnings
import pickle
#plots display in notebook
%matplotlib inline
#import math(scipy stats), time and dictionary modules
from collections import defaultdict
from scipy.stats import randint
from scipy import stats
from time import time
#import course provided functions
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
#import sklearn classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#import additional sklearn modules 
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit,\
                                    RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
#supresses warnings about cross_validation being depreciated in favour of
#model_selection, introduced version 0.18 removed 0.20
warnings.simplefilter(action = "ignore", category = FutureWarning)

#load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)
	
#total number of data points
def total_people(dataset):
    people = len(dataset)
    print "Total Number of People in the Dataset:", people
    return people
    
people = total_people(enron_data)

#allocation across classes (POI/non-POI)
def total_classes(dataset):
    poi = 0
    people = 0
    for person in dataset:
        people += 1
        if dataset[person]["poi"]==1:
            poi += 1
    poi_pct = 100*float(poi)/people
    print "Number of POIs in the Dataset:",poi,"(%0.2f%%)"%poi_pct
    print "Number of Non-POIs in the Dataset:",people-poi,\
                                                "(%0.2f%%)"%(100-poi_pct)
    return poi

poi = total_classes(enron_data)

#number of features used
def total_features(dataset):
    features = []
    for key, value in dataset.iteritems() :
        for item in value:
            if item not in features:
                features.append(item)
    print "Number of Features Available:", len(features)
    return features
    
features = total_features(enron_data)
print "\nList of Features:", features

#indentifying features with missing values
missing_features = defaultdict(list)
mf_percent = defaultdict(list)
mf_poi = []
for feature in features:
    missing_features[feature] = [0,0]
    mf_percent[feature] = [0,0]
    
for person in enron_data:
    for feature in features:
        if enron_data[person][feature] == 'NaN':
            missing_features[feature][0] += 1
            if enron_data[person]["poi"] == 1:
                missing_features[feature][1] += 1
                
for feature in missing_features:
    x = (100*missing_features[feature][0] / float(people))
    mf_percent[feature][0] = "%0.1f" % x
    y = (100*missing_features[feature][1] / float(poi))
    mf_percent[feature][1] = "%0.1f" % y
    mf_poi.append(y)
    
print "Missing Features [Total, POI]:\n", missing_features
print "\nMissing Features as a Percentile [Total %, POI %]:\n", mf_percent

plt.hist(mf_poi)
plt.title("Features for POIs")
plt.xlabel("% Missing Values")
plt.ylabel("Frequency")
plt.show()

#removing features
removed_features = []
for feature in mf_percent:
    if float(mf_percent[feature][1]) > 70:
        removed_features.append(feature)
for person in enron_data:
    for feature in removed_features:
        enron_data[person].pop(feature, 0)
print "Removed Features:\n", removed_features

#indentifying persons with missing values
names = []
zeros = []
pois = []
for person in enron_data:
    zero = 0
    pois.append(enron_data[person]['poi'])
    for key in enron_data[person]:
        value = enron_data[person][key]
        if value == (0 or 'NaN'):
            zero += 1
    names.append(person)
    zeros.append(zero)
#seperating zeros with POI allocation
poi_true = []
poi_false = []
for i in range(0, len(zeros)):
    if pois[i] == True:
        poi_true.append(zeros[i])
    else:
        poi_false.append(zeros[i])
#histogram of persons by number of NaN values, seperated by POI allocation
binwidth = 1
bins= range(min(poi_false), max(poi_false) + binwidth, binwidth)
plt.hist(poi_true, bins, alpha=0.5, label='POI')
plt.hist(poi_false, bins, alpha=0.5, label='Non POI')
plt.legend(loc='upper right')
plt.xticks(np.arange(min(poi_false), max(poi_false)+1, 1.0))
plt.title('Histogram of Persons by Number of \'NaN\' values')
plt.xlabel("No. Missing Values")
plt.ylabel("No. Persons")
plt.show()

removed_names = []
for i in range(0, len(names)):
    if zeros[i] >= 11: 
        enron_data.pop(names[i], 0)
        removed_names.append(names[i])

print "No. Datapoints Removed:", len(removed_names), "\n\n", removed_names

def scatter(dataset, features):
    data = featureFormat(dataset, features)
    for pair in data:
        x = pair[0]
        y = pair[1]
        plt.scatter(x,y)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()    
	
#plotting salary vs bonus
features = ['total_payments', 'expenses']
scatter(enron_data, features)

#searching for the outlier
def find_largest_value(dataset, feature_name):
    outlier_name = ""
    largest_value = 0
    for key, value in enron_data.iteritems():
        inside_dict = value
        for k, v in inside_dict.iteritems():
            if (k == feature_name) & (v != 'NaN'):
                if v > largest_value:
                    largest_value = v
                    outlier_name = key
    print outlier_name, largest_value
	
find_largest_value(enron_data, 'total_payments')

enron_data.pop("TOTAL", 0);

def scatter_hist(dataset, features):
    data = featureFormat(enron_data, features)
    df = pd.DataFrame(data, columns=[features[0], features[1]])
    sns.jointplot(x=features[0], y=features[1], data=df)
	
scatter_hist(enron_data, features)

find_largest_value(enron_data, 'total_payments')

enron_data.pop("LAY KENNETH L", 0)
scatter_hist(enron_data, features)

#no. features remaining
features=total_features(enron_data)
#total number of data points
people=total_people(enron_data)
#allocation across classes (POI/non-POI)
poi=total_classes(enron_data)

#original financial features
f_features = ['salary','deferral_payments','total_payments','loan_advances',
              'bonus','restricted_stock_deferred','deferred_income', 
              'total_stock_value','expenses','exercised_stock_options','other', 
              'long_term_incentive','restricted_stock','director_fees']
#original email features
e_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
              'from_this_person_to_poi', 'shared_receipt_with_poi']

#removing features from original list if in removed_features
f_features = [feat for feat in f_features if feat not in removed_features]
e_features = [feat for feat in e_features if feat not in removed_features]

enron_df = pd.DataFrame.from_dict(enron_data, "index")

financial_df = enron_df[f_features]
financial_df = financial_df.replace("NaN", 0)

email_df = enron_df[e_features]
email_df = email_df.replace("NaN", 0)

financial_df["bonus/salary"] = financial_df["bonus"] / financial_df["salary"]
financial_df["stock/salary"] = financial_df["total_stock_value"] /\
                                                    financial_df["salary"]
financial_df = financial_df.replace(["inf","-inf", "NaN"], 0)

features = ["bonus/salary","stock/salary"]
sns.jointplot(x=features[0], y=features[1], data=financial_df[features]);

outlier = financial_df["stock/salary"].idxmax()
print "Stock/Salary outlier:", outlier
print enron_data[outlier]

enron_df = enron_df.drop(outlier)
financial_df = financial_df.drop(outlier)
email_df = email_df.drop(outlier)

sns.jointplot(x=features[0], y=features[1], data=financial_df[features]);

email_df["rec_poi/total_rec"] = email_df["from_poi_to_this_person"] /\
                                                    email_df["to_messages"]
email_df["sent_poi/total_sent"] = email_df["from_this_person_to_poi"] /\
                                                    email_df["from_messages"]
email_df = email_df.replace(["inf","-inf", "NaN"], 0)

features = ["rec_poi/total_rec","sent_poi/total_sent"]
sns.jointplot(x=features[0], y=features[1], data=email_df[features]);

hybrid_df = pd.DataFrame()
hybrid_df["payments/rec_poi_ratio"] = financial_df["total_payments"] /\
                                        email_df["rec_poi/total_rec"]
hybrid_df["payments/sent_poi_ratio"] = financial_df["total_payments"] /\
                                        email_df["sent_poi/total_sent"]
hybrid_df = hybrid_df.replace(["inf","-inf", "NaN"], 0)

features = ["payments/rec_poi_ratio","payments/sent_poi_ratio"]
sns.jointplot(x=features[0], y=features[1], data=hybrid_df[features]);

#calculates pearson correlation (r)
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    
def pair_grid(dataframe, filename):
    g = sns.PairGrid(dataframe)
    g.map_offdiag(plt.scatter)
    g.map_diag(sns.distplot, kde=False)
    g.map_lower(corrfunc)
    g.savefig(filename)
	
pair_grid(financial_df, "pairgrid_ff.png")

features = ["total_stock_value", "exercised_stock_options"]
pair_grid(financial_df[features], "pairgrid_stocks.png")

pca = PCA(n_components=1)
stocks = pca.fit_transform(financial_df[features])
stocks = stocks-min(stocks)
financial_df["pca_stocks"] = stocks
#plot new pca stock feature with former stock features
features = ["total_stock_value", "exercised_stock_options", "pca_stocks"]
pair_grid(financial_df[features], "pairgrid_stocks_pca.png")
#drop former stock features
features = ["total_stock_value", "exercised_stock_options"]
financial_df = financial_df.drop(features, 1)

def add_one(array):
    new_array = [x+1 for x in array]
    return new_array
	
def log10(array):
    #adding 1 to the feature data to prevent log10(0)=undefined
    new_array = [x+1 for x in array]
    new_array = np.log10(new_array)
    return new_array
	
financial_df = financial_df.apply(abs).apply(log10)

def boolean_to_string(boolean):
    if boolean == False:
        return "No"
    elif boolean == True:
        return "Yes"
    else:
        return boolean
		
financial_df["poi"] = enron_df["poi"]
#boolean to string is needed so that pairplot won't include poi as a feature
financial_df["poi"] = financial_df["poi"].apply(boolean_to_string)
features = ["total_payments", "bonus/salary", "pca_stocks", "poi"]
sns.pairplot(financial_df[features], hue="poi");

#pair_grid(email_df, "pairgrid_ef_all.png")
features = ["to_messages", "from_poi_to_this_person"]
pair_grid(email_df[features], "pairgrid_ef.png")

#log10 transformation of email features
email_df = email_df.apply(log10)
email_df["poi"] = enron_df["poi"]
#boolean to string is needed so that pairplot won't include poi as a feature
email_df["poi"] = email_df["poi"].apply(boolean_to_string)

features = ["to_messages", "rec_poi/total_rec", 
            "shared_receipt_with_poi", "poi"]
sns.pairplot(email_df[features], hue="poi");

#log10 transformation of hybrid features
hybrid_df = hybrid_df.apply(log10)
hybrid_df["poi"] = enron_df["poi"]
#boolean to string is needed so that pairplot won't include poi as a feature
hybrid_df["poi"] = hybrid_df["poi"].apply(boolean_to_string)

sns.pairplot(hybrid_df, hue="poi");

#dropping poi strings used for 'hue' in the preceding graphs
financial_df = financial_df.drop("poi", 1)
email_df = email_df.drop("poi", 1)
hybrid_df = hybrid_df.drop("poi", 1)

#merging the dataframes together
features_df = financial_df.join(email_df).join(hybrid_df).join(enron_df["poi"])
feature_list = features_df.columns.tolist()
#move poi to front of list for tester.py
feature_list.insert(0, feature_list.pop(feature_list.index('poi')))

#creating a new dictionary dataset
dataset = features_df.to_dict(orient='index')

#no. features remaining
features=total_features(dataset)
#total number of data points
people=total_people(dataset)
#allocation across classes (POI/non-POI)
poi=total_classes(dataset)

pipe = Pipeline(steps=[
        ('scale1', MinMaxScaler()),
        ('pca', PCA()),
        ('scale2', MinMaxScaler()),
        ('skb', SelectKBest())
    ])
param = {
        'pca__n_components':randint(8,19),
        'skb__k':randint(1,9)
    }
	
names = []
pipes = []
params = []

names.append('Gaussian Naive Bayes')
pipes.append(('gnb', GaussianNB()))
params.append({
    })

names.append('Support Vector Machine')
pipes.append(('svc', SVC()))
params.append({
        'svc__kernel':('poly', 'rbf', 'sigmoid'), 
        'svc__C':randint(1,151), 
        'svc__gamma':randint(1,21)
    })

names.append('Decision Tree Classifier')
pipes.append(('dtc', DecisionTreeClassifier()))
params.append({
        'dtc__criterion':('gini', 'entropy'), 
        'dtc__splitter':('best', 'random'),
        'dtc__min_samples_split':randint(2,26),
        'dtc__max_depth':randint(4,51),
        'dtc__max_leaf_nodes':randint(4,31)
    })

names.append('K Neighbors Classifier')
pipes.append(('knc', KNeighborsClassifier()))
params.append({
        'knc__n_neighbors':randint(1, 31), 
        'knc__p':randint(1,5),
        'knc__weights':('uniform', 'distance')
    })

names.append('Random Forest Classifier')
pipes.append(('rfc', RandomForestClassifier()))
params.append({
        'rfc__n_estimators':randint(2,11), 
        'rfc__criterion':('gini','entropy')
    })

names.append('AdaBoost Classifier')
pipes.append(('abc', AdaBoostClassifier()))
params.append({
        'abc__n_estimators':randint(2,11), 
        'abc__learning_rate':randint(1,6)
    })
	
pipelines = []
parameters = []

for i in range(0, len(pipes)):
    pipelines.append(Pipeline(steps=[x for x in pipe.steps]))
    pipelines[i].steps.append(pipes[i])
    parameters.append(param.copy())
    parameters[i].update(params[i])
    
classifiers = []
classifiers.extend((names, pipelines, parameters))

def search_test_clf(n_iter, n_splits, clfs, dataset, feat_list, fp=False):
    
    best_score = 0
    data = featureFormat(dataset, feat_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=42, 
                                          test_size=0.3)
    start = time()
    for i in range(0,len(clfs[0])):
        print clfs[0][i], "------------------------------------------"
        t0 = time() 
        warnings.filterwarnings('ignore')
        rs = RandomizedSearchCV(clfs[1][i], param_distributions=clfs[2][i],
                                n_iter=n_iter, scoring='f1', cv=cv,
                                random_state=42)
        rs.fit(features, labels)
        search_time = round(time()-t0, 3)
        t0 = time() 
        test_classifier(rs.best_estimator_, dataset, feat_list, 1000, fp)
        print "Search/Fit Time:", search_time, "s", "   Test Time:",\
                                                round(time()-t0, 3), "s", "\n"
            
        if rs.best_score_ > best_score:
            best_estimator = rs.best_estimator_
            best_score = rs.best_score_
            clf_name = clfs[0][i]
    
    print "\nTotal Run Time:", round(time()-start, 3), "s"
    print "Best Classifier:", clf_name
    return best_estimator
	
test_classifier(GaussianNB(), dataset, feature_list)

estimator = search_test_clf(25, 25, classifiers, dataset, feature_list)

dump_classifier_and_data(estimator, dataset, feature_list)

abstract_df = pd.DataFrame()
abstract_df["log-rec_poi_ratio(payments)"] = financial_df["total_payments"]/\
                                        email_df["rec_poi/total_rec"]
abstract_df["log-sent_poi_ratio(payments)"] = financial_df["total_payments"]/\
                                        email_df["sent_poi/total_sent"]
abstract_df["log-rec_poi_ratio(stock-salary)"] = financial_df["stock/salary"]/\
                                        email_df["rec_poi/total_rec"]
abstract_df["log-sent_poi_ratio(stock-salary)"] = financial_df["stock/salary"]/\
                                        email_df["sent_poi/total_sent"]
abstract_df["log-rec_poi_ratio(bonus-salary)"] = financial_df["bonus/salary"]/\
                                        email_df["rec_poi/total_rec"]
abstract_df["log-sent_poi_ratio(bonus-salary)"] = financial_df["bonus/salary"]/\
                                        email_df["sent_poi/total_sent"]
abstract_df = abstract_df.replace(["inf","-inf", "NaN"], 0)


abstract_df["poi"] = enron_df["poi"]
#boolean to string is needed so that pairplot won't include poi as a feature
abstract_df["poi"] = abstract_df["poi"].apply(boolean_to_string)

features = ["log-sent_poi_ratio(payments)", "log-rec_poi_ratio(bonus-salary)",
           "log-sent_poi_ratio(stock-salary)", "poi"]
g = sns.pairplot(abstract_df[features], hue="poi")
g.savefig("pair_abstract.png")

abstract_df = abstract_df.drop("poi", 1)

features_df = features_df.join(abstract_df)
feature_list = features_df.columns.tolist()
feature_list.insert(0, feature_list.pop(feature_list.index('poi')))
dataset = features_df.to_dict(orient='index')

search_test_clf(25, 25, classifiers, dataset, feature_list);

def remove_classifier(target, clfs):
    for i in range(0,len(clfs[0])):
        if clfs[0][i] == target:
            clfs[0].remove(clfs[0][i])
            clfs[1].remove(clfs[1][i])
            clfs[2].remove(clfs[2][i])
            break
			
remove_classifier("Gaussian Naive Bayes", classifiers)
remove_classifier("Support Vector Machine", classifiers)
remove_classifier("Decision Tree Classifier", classifiers)
remove_classifier("Random Forest Classifier", classifiers)
remove_classifier("AdaBoost Classifier", classifiers)

estimator = search_test_clf(200, 25, classifiers, dataset, feature_list)

dump_classifier_and_data(estimator, dataset, feature_list)