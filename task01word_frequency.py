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

# 只根据训练集建立词表，并把训练文本转换为记录频率的向量
vectorizer = CountVectorizer(
    tokenizer=tokenize_with_nltk,
    token_pattern=None,
    lowercase=True,
    binary=False,
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
"newton-cholesky":
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 7.15 TiB for an array with shape (991560, 991560) and data type float64

词表大小： 123944
训练集矩阵形状： (204800, 123944)

Solver: lbfgs
训练耗时: 42741.35 毫秒
测试集预测耗时: 5.17 毫秒
实际迭代次数: [313]
Validation Accuracy: 0.8887109375
Validation Macro-F1: 0.88858709315708
Test Accuracy: 0.8865625
Test Macro-F1: 0.8865045427430736

Solver: saga
训练耗时: 80984.26 毫秒
测试集预测耗时: 5.22 毫秒
实际迭代次数: [369]
Validation Accuracy: 0.8892578125
Validation Macro-F1: 0.8891379210784262
Test Accuracy: 0.8869921875
Test Macro-F1: 0.8869327030027339
'''