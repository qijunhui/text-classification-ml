# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/19 21:23
@Author  : qijunhui
@File    : test.py
"""
import os
from utils import read_pkl
from config import MODEL_PATH

VECTORIZE = read_pkl(os.path.join(MODEL_PATH, "vectorizer.pkl"))
MODELS = {
    "mnb": read_pkl(os.path.join(MODEL_PATH, "mnb_model.pkl")),
    "lr": read_pkl(os.path.join(MODEL_PATH, "lr_model.pkl")),
    "svm": read_pkl(os.path.join(MODEL_PATH, "svm_model.pkl")),
}


def predict(text, model):
    X = VECTORIZE.transform([text])
    y = MODELS[model].predict(X)[0]
    return y


if __name__ == "__main__":
    text = "this is a stunning film , a one of a kind tour de force"
    print("[MNB] 预测结果:", predict(text, model="mnb"))
    print("[LR] 预测结果:", predict(text, model="lr"))
    print("[SVM] 预测结果:", predict(text, model="svm"))
