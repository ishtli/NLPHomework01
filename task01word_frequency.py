from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

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

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
)
model.fit(X_train, train["label"])

val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(val["label"], val_pred))
print("Validation Macro-F1:", f1_score(val["label"], val_pred, average="macro"))

test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(test["label"], test_pred))
print("Test Macro-F1:", f1_score(test["label"], test_pred, average="macro"))
'''
saga：
词表大小： 123944
训练集矩阵形状： (204800, 123944)
Validation Accuracy: 0.8892578125
Validation Macro-F1: 0.8891379210784262
Test Accuracy: 0.8869921875
Test Macro-F1: 0.8869327030027339
'''
'''
lbfgs:
词表大小： 123944
训练集矩阵形状： (204800, 123944)
Validation Accuracy: 0.8887109375
Validation Macro-F1: 0.88858709315708
Test Accuracy: 0.8865625
Test Macro-F1: 0.8865045427430736
'''