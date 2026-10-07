from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from time import perf_counter

root = Path(__file__).resolve().parent
data_dir = root / "data" / "NYTprocessed"

train = pd.read_csv(data_dir / "train.csv")
val = pd.read_csv(data_dir / "val.csv")
test = pd.read_csv(data_dir / "test.csv")

def tokenize_with_nltk(text):
    tokens = word_tokenize(text, preserve_line=True)
    return [
        token for token in tokens
        if any(char.isalnum() for char in token)
    ]

# 只根据训练集建立词表，并把训练文本转换为 0/1 向量
vectorizer = CountVectorizer(
    tokenizer=tokenize_with_nltk,
    token_pattern=None,
    lowercase=True,
    binary=True,
)
X_train = vectorizer.fit_transform(train["text"])

# 验证集和测试集沿用训练集的词表
X_val = vectorizer.transform(val["text"])
X_test = vectorizer.transform(test["text"])

print("词表大小：", len(vectorizer.vocabulary_))
print("训练集矩阵形状：", X_train.shape)

for solver in ("lbfgs", "saga"):
    model = LogisticRegression(
        solver=solver,
        max_iter=1000,
        random_state=42,
    )

    start = perf_counter()
    model.fit(X_train, train["label"])
    fit_seconds = perf_counter() - start

    val_pred = model.predict(X_val)

    start = perf_counter()
    test_pred = model.predict(X_test)
    predict_seconds = perf_counter() - start

    print(f"\nSolver: {solver}")
    print(f"训练耗时: {fit_seconds * 1000:.2f} 毫秒")
    print(f"测试集预测耗时: {predict_seconds * 1000:.2f} 毫秒")
    print(f"实际迭代次数: {model.n_iter_}")
    print("Validation Accuracy:", accuracy_score(val["label"], val_pred))
    print("Validation Macro-F1:", f1_score(val["label"], val_pred, average="macro"))
    print("Test Accuracy:", accuracy_score(test["label"], test_pred))
    print("Test Macro-F1:", f1_score(test["label"], test_pred, average="macro"))

'''
词表大小： 123944
训练集矩阵形状： (204800, 123944)

Solver: lbfgs
训练耗时: 30345.17 毫秒
测试集预测耗时: 5.06 毫秒
实际迭代次数: [219]
Validation Accuracy: 0.89046875
Validation Macro-F1: 0.890402655771555
Test Accuracy: 0.8873046875
Test Macro-F1: 0.8872279693474925

Solver: saga
训练耗时: 27476.04 毫秒
测试集预测耗时: 5.17 毫秒
实际迭代次数: [125]
Validation Accuracy: 0.8903125
Validation Macro-F1: 0.8902406123573597
Test Accuracy: 0.8878515625
Test Macro-F1: 0.8877723434186537
'''