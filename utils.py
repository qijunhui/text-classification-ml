# -*- coding: utf-8 -*-
"""
@Time    : 2020/12/19 19:37
@Author  : qijunhui
@File    : utils.py
"""
import os, pickle, csv, re, jieba


def read_pkl(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, "rb") as fr:
            data = pickle.load(fr, encoding="utf-8")
        if isinstance(data, list) or isinstance(data, dict) or isinstance(data, tuple):
            print(f"{filepath} [{len(data)}] 已加载...")
        else:
            print(f"{filepath} 已加载...")
    else:
        print(f"{filepath} 文件不存在...")
    return data


def save_pkl(filepath, data):
    with open(filepath, "wb") as fw:
        pickle.dump(data, fw)
    if isinstance(data, list) or isinstance(data, dict) or isinstance(data, tuple):
        print(f"{filepath} [{len(data)}] 文件已存储...")
    else:
        print(f"{filepath} 文件已存储...")


def read_csv(filepath, filter_title=False, delimiter=","):
    data = []
    if os.path.exists(filepath):  # 如果目标文件存在:
        with open(filepath, "r", encoding="utf-8") as fr:
            data = csv.reader(fr, delimiter=delimiter)  # 逐行读取csv文件 迭代器变量
            if filter_title:
                next(data)  # 过滤首行
            data = list(data)
        print(f"{filepath} [{len(data)}] 已加载... ")
    else:
        print(f"{filepath} 文件不存在...")
    return data


def tokenizer(text, language="en"):
    if language == "en":
        text = re.sub(r"([,.!?])", r" \1 ", text)
        text = re.sub(" {2,}", " ", text)
    elif language == "zh":
        text = " ".join(jieba.cut(text))
    else:
        raise Exception("请选择正确的language")
    return text.strip()
