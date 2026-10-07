from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec

from time import perf_counter
'''
Pretrained GloVe: glove
Train on AGNews: ag
Train on NYT: nyt
'''
METHOD = "glove"
DIM = 100

root = Path(__file__).resolve().parent
nyt_dir = root / "data" / "NYTprocessed"


def tokenize_with_nltk(text):
    tokens = word_tokenize(text.lower(), preserve_line=True)
    return [
        token for token in tokens
        if any(char.isalnum() for char in token)
    ]


def load_glove(zip_path):
    vectors = {}

    with ZipFile(zip_path) as archive:
        with archive.open("glove.6B.100d.txt") as file:
            for line in file:
                parts = line.decode("utf-8").split()

                # 1 个词 + 100 个数字
                if len(parts) == 101:
                    word = parts[0]
                    vector = np.array(parts[1:], dtype=np.float32)
                    vectors[word] = vector

    return vectors


def train_word2vec(sentences):
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        sg=0,
        epochs=5,
        workers=1,
        seed=42,
    )
    return model.wv


def make_document_vectors(tokenized_documents, word_vectors):
    X = np.zeros((len(tokenized_documents), DIM), dtype=np.float32)
    empty_count = 0

    for row, tokens in enumerate(tokenized_documents):
        vectors = [
            word_vectors[token]
            for token in tokens
            if token in word_vectors
        ]

        if vectors:
            X[row] = np.mean(vectors, axis=0)
        else:
            empty_count += 1

    return X, empty_count


train = pd.read_csv(nyt_dir / "train.csv")
val = pd.read_csv(nyt_dir / "val.csv")
test = pd.read_csv(nyt_dir / "test.csv")

train_tokens = [tokenize_with_nltk(text) for text in train["text"]]
val_tokens = [tokenize_with_nltk(text) for text in val["text"]]
test_tokens = [tokenize_with_nltk(text) for text in test["text"]]


if METHOD == "ag":
    ag_path = root / "data" / "ag_news" / "data" / "train-00000-of-00001.parquet"
    ag_texts = pd.read_parquet(ag_path, columns=["text"])["text"]
    ag_tokens = [tokenize_with_nltk(text) for text in ag_texts]

    word_vectors = train_word2vec(ag_tokens)

elif METHOD == "nyt":
    word_vectors = train_word2vec(train_tokens)

elif METHOD == "glove":
    glove_path = root / "data" / "glove.6B.zip"
    word_vectors = load_glove(glove_path)

else:
    raise ValueError("METHOD must be 'ag', 'nyt', 'glove'")

print("METHOD：", METHOD)
print("len(word_vectors)：", len(word_vectors))


X_train, empty_train = make_document_vectors(train_tokens, word_vectors)
X_val, empty_val = make_document_vectors(val_tokens, word_vectors)
X_test, empty_test = make_document_vectors(test_tokens, word_vectors)

print("Document matrix shapes：", X_train.shape, X_val.shape, X_test.shape)
print("Documents with no known words:：", empty_train, empty_val, empty_test)

for solver in ("lbfgs",):
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
    print(f"fit time: {fit_seconds * 1000:.2f} ms")
    print(f"perdict time: {predict_seconds * 1000:.2f} ms")
    print(f"num_of_iter: {model.n_iter_}")
    print("Validation Accuracy:", accuracy_score(val["label"], val_pred))
    print("Validation Macro-F1:", f1_score(val["label"], val_pred, average="macro"))
    print("Test Accuracy:", accuracy_score(test["label"], test_pred))
    print("Test Macro-F1:", f1_score(test["label"], test_pred, average="macro"))

#old version saga maxiter 1000
'''                                                                              
Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'                                                                                 
METHOD： ag                                                                                                                                        
len(word_vectors)： 57176                                                                                                                          
Document matrix shapes： (204800, 100) (25600, 100) (25600, 100)                                                                                   
Documents with no known words:： 3 2 2                                                                                                             
Validation Accuracy: 0.752578125                                                                                                                   
Validation Macro-F1: 0.7516476436481353                                                                                                            
Test Accuracy: 0.751484375                                                                                                                         
Test Macro-F1: 0.7502565744322334
'''
'''
Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'                                                                                 
Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'                                                                                 
METHOD： nyt                                                                                                                                       
len(word_vectors)： 69873                                                                                                                          
Document matrix shapes： (204800, 100) (25600, 100) (25600, 100)                                                                                   
Documents with no known words:： 0 0 1                                                                                                             
C:\\Users\\isHtL\\.conda\\envs\\NLP312\\Lib\\site-packages\\sklearn\\linear_model\\_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge                                                                                                                             
  warnings.warn(                                                                                                                                   
Validation Accuracy: 0.8436328125                                                                                                                  
Validation Macro-F1: 0.8433573078234208                                                                                                            
Test Accuracy: 0.84359375
Test Macro-F1: 0.8433312660357782
'''
'''
METHOD： glove
len(word_vectors)： 400000
Document matrix shapes： (204800, 100) (25600, 100) (25600, 100)
Documents with no known words:： 1 0 0
C:\\Users\\isHtL\\.conda\\envs\\NLP312\\Lib\\site-packages\\sklearn\\linear_model\\_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  warnings.warn(
Validation Accuracy: 0.820234375
Validation Macro-F1: 0.8197032847642278
Test Accuracy: 0.821953125
Test Macro-F1: 0.8213322458127911
'''
#new version
#nyt glove with saga can't converge with max_iter=1000 as above
#nyt glove with saga fit time too long with max_iter=3000, so only lbfgs tested
'''
Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'
METHOD： ag
len(word_vectors)： 57176
Document matrix shapes： (204800, 100) (25600, 100) (25600, 100)
Documents with no known words:： 3 2 2

Solver: lbfgs
fit time: 4150.27 ms
perdict time: 4.86 ms
num_of_iter: [122]
Validation Accuracy: 0.7527734375
Validation Macro-F1: 0.7518183214700418
Test Accuracy: 0.7511328125
Test Macro-F1: 0.749926042409228

Solver: saga
fit time: 27447.30 ms
perdict time: 2.32 ms
num_of_iter: [56]
Validation Accuracy: 0.752578125
Validation Macro-F1: 0.7516476436481353
Test Accuracy: 0.751484375
Test Macro-F1: 0.7502565744322334
'''
'''
Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'
Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'
METHOD： nyt
len(word_vectors)： 69873
Document matrix shapes： (204800, 100) (25600, 100) (25600, 100)
Documents with no known words:： 0 0 1

Solver: lbfgs
fit time: 4247.51 ms
perdict time: 2.35 ms
num_of_iter: [126]
Validation Accuracy: 0.843515625
Validation Macro-F1: 0.8432594927492946
Test Accuracy: 0.843359375
Test Macro-F1: 0.8430915378848677
'''

'''
METHOD： glove
len(word_vectors)： 400000
Document matrix shapes： (204800, 100) (25600, 100) (25600, 100)
Documents with no known words:： 1 0 0

Solver: lbfgs
fit time: 4945.64 ms
perdict time: 4.72 ms
num_of_iter: [139]
Validation Accuracy: 0.820078125
Validation Macro-F1: 0.8195598896088471
Test Accuracy: 0.8215625
Test Macro-F1: 0.8209545738375164
'''