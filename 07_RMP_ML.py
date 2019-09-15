#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

src = r'/Users/liumingchun/【1】科研+实验室/3-科研项目/RateMyProfessor/bigdata/ratemyprofessor.csv'
data = pd.read_csv(src,nrows=10000, usecols=['star_rating', 'comments'])
pd.set_option('display.max_rows', 100, 'display.max_columns', 1000, "display.max_colwidth", 1000, 'display.width',1000)
data = data.rename(columns={'star_rating' : 'star_rating', 'comments' : 'processed_comments'})

sns.distplot(data['star_rating']) # 正态分布
# plt.show()

data.loc[(data['star_rating'] <= 2) & (data['star_rating'] >= 1), 'star_rating'] = 1
data.loc[(data['star_rating'] <= 3) & (data['star_rating'] > 2), 'star_rating'] = 2
data.loc[(data['star_rating'] <= 4) & (data['star_rating'] > 3), 'star_rating'] = 3
data.loc[(data['star_rating'] <= 5) & (data['star_rating'] > 4), 'star_rating'] = 4

sns.distplot(data['star_rating'])
# plt.show()

data['star_rating'] = data['star_rating'].astype(int)
data.isnull().sum()
data = data.dropna()
data.isnull().sum()
# print(data)


import nltk
import wordcloud as wcloud
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import wordpunct_tokenize


words = data['processed_comments']
words_str = words.str.cat(sep = ' ')
print(words_str[:20])
list_of_words = [i.lower() for i in wordpunct_tokenize(words_str) if i.lower() not in stop and i.isalpha()]
print("Number of words : ",len(list_of_words))

wordfreqdist = nltk.FreqDist(list_of_words)
print("Number of unique words : ",len(wordfreqdist))

mostcommon = wordfreqdist.most_common(30)
most_common_df = pd.DataFrame(mostcommon, columns=['word', 'count'])
print(most_common_df.head())



# # N-gram Analysis
most_common = most_common_df[0:10]
least_common_df = pd.DataFrame((wordfreqdist.most_common()[-10:]), columns=['word', 'count'])
print('most_common',most_common)
print('least_common',least_common_df)
sns.barplot(x = 'count', y = 'word', data = most_common_df[:10])
plt.show()
sns.barplot(x = 'count', y = 'word', data = least_common_df)
plt.show()

total_text = ' '.join(data['processed_comments'])
total_text = total_text.lower()
total_text = nltk.word_tokenize(total_text)
total_text = [word for word in total_text if word not in stop and len(word)>=3 ]
lemmatizer = nltk.stem.WordNetLemmatizer()
total_text = [lemmatizer.lemmatize(w, 'v') for w in total_text]

trigrams = nltk.ngrams(total_text, 3)
text = nltk.Text(trigrams)
fDist = nltk.FreqDist(text)


tri_grams_dict = {}
for gram in fDist.most_common(10)[0:10]:
    key = ' '.join(gram[0])
    tri_grams_dict[key] = gram[1]

tri_grams_df = pd.DataFrame(list(tri_grams_dict.items()), columns = ['trigram', 'freq'])
tri_grams_df.head(10)
tri_grams_dict_least = {}

for gram in fDist.most_common()[-10:]:
    key = ' '.join(gram[0])
    tri_grams_dict_least[key] = gram[1]

tri_grams_dict_least_df = pd.DataFrame(list(tri_grams_dict_least.items()), columns = ['trigram', 'freq'])


tri_grams_dict_least_df.head(10)
sns.barplot(x='freq', y = 'trigram', data=tri_grams_df)
plt.xticks(rotation=90)
sns.barplot(x='freq', y = 'trigram', data=tri_grams_dict_least_df)
plt.xticks(rotation=90)



# # Topic Modelling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# tf-idf
tf_idf_vect = TfidfVectorizer(stop_words=stop)
tf_idf = tf_idf_vect.fit_transform(data['processed_comments'])

# tf
tf_vect = CountVectorizer(stop_words=stop)
tf = tf_vect.fit_transform(data['processed_comments'])

from sklearn.cluster import MiniBatchKMeans
clusters = 10
kmeans_model = MiniBatchKMeans(n_clusters = clusters, init = 'k-means++', n_init = 1, init_size = 1000, batch_size = 1000, verbose = False, max_iter=1000)
kmodel = kmeans_model.fit(tf_idf)
kmodel_clusters = kmodel.predict(tf_idf)
centroids = kmodel.cluster_centers_.argsort()[:,::-1]
values = tf_idf_vect.get_feature_names()

for i in range(clusters):
    list1 = []
    print('Cluster %d:' % i)
    for j in centroids[i, :10]:
        list1.append(values[j])
    print(list1)
    print()


# # Reviews Classification
classified_df = data
classified_df['ratings_labeled'] = [1 if ratings >= 3 else 0 for ratings in classified_df['star_rating']]
classified_df.head()
classified_df['ratings_labeled'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(classified_df['processed_comments'],
                                                    classified_df['ratings_labeled'],
                                                    random_state=0)
c_vect = CountVectorizer(min_df=5).fit(X_train)
X_train = c_vect.transform(X_train)
X_test = c_vect.transform(X_test)


# ### Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_l1 = LogisticRegression(penalty='l1').fit(X_train, y_train)
log_y_pred = logistic_l1.predict(X_test)
print('Accuracy of Logistic with Lasso on test set: {:.2f}'.format(logistic_l1.score(X_test, y_test)))


# ### Confusion Matrix
from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(y_test, log_y_pred, labels=[0, 1])
ax= plt.subplot()
sns.heatmap(cfm, annot=cfm, ax = ax, cmap = 'PiYG', fmt='g')
plt.show()

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative', 'Positive'])


from sklearn.metrics import classification_report
print('聚类',classification_report(y_test, log_y_pred))

# ### Naive-Bayes
from sklearn.naive_bayes import MultinomialNB
multi_nb = MultinomialNB(alpha=.01).fit(X_train, y_train)
mnb_y_pred = multi_nb.predict(X_test)
print('Accuracy of Multinomial NB on test set: {:.2f}'.format(multi_nb.score(X_test, y_test)))

# ### RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion = 'entropy').fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
print('Accuracy of Random Forests on test set: {:.2f}'.format(rf.score(X_test, y_test)))

# ### Support Vector Classification
from sklearn.svm import LinearSVC
svc = LinearSVC(C=0.01).fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print('Accuracy of Random Forests on test set: {:.2f}'.format(svc.score(X_test, y_test)))

# # Handle Class Imbalance
import imblearn
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12, ratio = 'minority')
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# # Model Optimization
from sklearn.model_selection import GridSearchCV
parameters={'penalty' : ['l1', 'l2'], 'C' : [0.001, 0.01, 0.1, 1, 10]}
clf_log=LogisticRegression()
clf=GridSearchCV(clf_log,parameters)
clf.fit(X_train_res,y_train_res)
clf.best_params_


# ### Logistic Regression
logistic_l2 = LogisticRegression(penalty='l2', C=0.1).fit(X_train_res, y_train_res)
log_y_pred = logistic_l2.predict(X_test)
print('Accuracy of Logistic with l2 penalty on test set: {:.2f}'.format(logistic_l2.score(X_test, y_test)))

# ### Support Vector Classification
from sklearn.svm import LinearSVC
svc = LinearSVC(C = 0.005).fit(X_train_res, y_train_res)
svc_y_pred = svc.predict(X_test)
print('Accuracy of LinearSVC on test set: {:.2f}'.format(svc.score(X_test, y_test)))


# ### Stochastic Gradient Classifier
import warnings; warnings.simplefilter('ignore')
from sklearn.linear_model import SGDClassifier
sgd_l2 = SGDClassifier(penalty='l2', alpha=0.001, loss= 'log').fit(X_train_res, y_train_res)
sgd_y_pred = sgd_l2.predict(X_test)
print('Accuracy of Stocastic Gradient Descent Classifier with l2 penalty on test set: {:.2f}'.format(sgd_l2.score(X_test, y_test)))


# ### Confusion Matrix
cfm = confusion_matrix(y_test, sgd_y_pred, labels=[0, 1])
ax= plt.subplot()
sns.heatmap(cfm, annot=cfm, ax = ax, cmap = 'PiYG', fmt='g')
plt.show()

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Negative', 'Positive']); ax.yaxis.set_ticklabels(['Negative', 'Positive'])
print(classification_report(y_test, sgd_y_pred))
print(classification_report(y_test, svc_y_pred))