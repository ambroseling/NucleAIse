from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC

data = pd.read_csv("all_datas.csv")

x = data["sequence"]
y = data["goa"].apply(lambda k: k.strip("[]").replace("'", "").split(', ')).values
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)
seq_list = x.to_list()


def compute_kmer_frequencies(sequence, k) -> dict:
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

    total_kmers = sum(kmer_counts.values())
    kmer_frequencies = {kmer: count/total_kmers for kmer, count in kmer_counts.items()}
    return kmer_frequencies


kmer_freq = [compute_kmer_frequencies(seq, 1) for seq in seq_list]
vector = DictVectorizer(sparse=False)
X_vectorized = vector.fit_transform(kmer_freq)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC()
svm_classifier = OneVsRestClassifier(svm)

param_grid = {
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__gamma': [1, 0.1, 0.01, 0.001],
    'estimator__kernel': ['rbf']
}

grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
