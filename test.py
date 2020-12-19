# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/19 21:23
@Author  : qijunhui
@File    : test.py
"""
import os
from utils import read_pkl
from config import MODEL_PATH

vectorizer = read_pkl(os.path.join(MODEL_PATH, "vectorizer.pkl"))
models = {
    "mnb_model": read_pkl(os.path.join(MODEL_PATH, "mnb_model.pkl")),
    "lr_model": read_pkl(os.path.join(MODEL_PATH, "lr_model.pkl")),
    "svm_model": read_pkl(os.path.join(MODEL_PATH, "svm_model.pkl")),
}

X = vectorizer.transform(["this is a stunning film , a one of a kind tour de force"])
for name, model in models.items():
    y = model.predict(X)
    print(f"[{name}] 预测结果:", y)
