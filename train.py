# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/19 19:37
@Author  : qijunhui
@File    : train.py
"""
import os
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from sklearn.naive_bayes import MultinomialNB  # 多项式分布贝叶斯
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.svm import SVC  # 支持向量机
from sklearn import metrics  # 评估模型
from utils import read_csv, read_pkl, save_pkl
from config import DATA_PATH, MODEL_PATH, TRAIN_DATA_RATIO

RETRAIN = True  # 是否重新训练


def load_data():
    datasets = read_csv(os.path.join(DATA_PATH, "datasets.tsv"), filter_title=True, delimiter="\t")
    X = [text for text, target in datasets]
    y = [int(target) for text, target in datasets]
    return X, y


def get_vectorizer(X_train, feature="count", train=False):
    if not train:
        vectorizer = read_pkl(os.path.join(MODEL_PATH, "vectorizer.pkl"))
        if vectorizer:
            return vectorizer
        else:
            print("开始训练vectorizer...")
    if feature == "count":
        vectorizer = CountVectorizer(binary=True)  # 基于词频的文本向量
    elif feature == "tfidf":
        vectorizer = TfidfVectorizer(binary=True)  # 基于tfidf的文本向量
    else:
        raise Exception("请选择正确的feature")
    vectorizer.fit(X_train)
    save_pkl(os.path.join(MODEL_PATH, "vectorizer.pkl"), vectorizer)
    return vectorizer


def get_mnb_model(X_train, y_train, train=False):
    if not train:
        mnb = read_pkl(os.path.join(MODEL_PATH, "mnb_model.pkl"))
        if mnb:
            return mnb
        else:
            print("开始训练MNB...")
    mnb = MultinomialNB(alpha=1, fit_prior=True)  # alpha用于平滑化，默认为1即可；fit_prior：是否学习类的先验概率，默认是True
    mnb.fit(X_train, y_train)
    save_pkl(os.path.join(MODEL_PATH, "mnb_model.pkl"), mnb)
    return mnb


def get_lr_model(X_train, y_train, train=False):
    if not train:
        lr = read_pkl(os.path.join(MODEL_PATH, "lr_model.pkl"))
        if lr:
            return lr
        else:
            print("开始训练LR...")
    lr = LogisticRegression(max_iter=1000, n_jobs=4)
    lr.fit(X_train, y_train)
    save_pkl(os.path.join(MODEL_PATH, "lr_model.pkl"), lr)
    return lr


def get_svm_model(X_train, y_train, train=False):
    if not train:
        svm = read_pkl(os.path.join(MODEL_PATH, "svm_model.pkl"))
        if svm:
            return svm
        else:
            print("开始训练SVM...")
    svm = SVC()
    svm.fit(X_train, y_train)
    save_pkl(os.path.join(MODEL_PATH, "svm_model.pkl"), svm)
    return svm


def assess(y_test, y_pred):
    print("accuracy on test data:\t", metrics.accuracy_score(y_test, y_pred))
    print("precision on test data:\t", metrics.precision_score(y_test, y_pred))  # 在计算精准或者召回的时候是以[1]为准计算
    print("recall on test data:\t", metrics.recall_score(y_test, y_pred))  # 在计算精准或者召回的时候是以[1]为准计算
    print("f1 on test data:\t\t", metrics.f1_score(y_test, y_pred))
    print(f"confusion matrix:\n{metrics.confusion_matrix(y_test, y_pred, labels=[1, 0])}")  # [1]正例  [0]负例


X, y = load_data()  # 加载数据
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_DATA_RATIO, random_state=0)
vectorizer = get_vectorizer(X_train, train=RETRAIN)  # 提取特征
X_train, y_train = vectorizer.transform(X_train), y_train  # 特征转换
print("训练数据中的样本个数: ", X_train.shape, "测试数据中的样本个数: ", [len(X_test)])
mnb_model = get_mnb_model(X_train, y_train, train=RETRAIN)
lr_model = get_lr_model(X_train, y_train, train=RETRAIN)
svm_model = get_svm_model(X_train, y_train, train=RETRAIN)

# 测试集验证效果
X_test, y_test = vectorizer.transform(X_test), y_test
models = {"mnb_model": mnb_model, "lr_model": lr_model, "svm_model": svm_model}
for name, model in models.items():
    print(f"{'=*= ' * 10}[{name}] {'=*= ' * 10}")
    y_pred = model.predict(X_test)
    assess(y_test, y_pred)
    print(f"{'=*= ' * 10}[end_model] {'=*= ' * 10}\n")
