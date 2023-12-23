import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

data = pd.read_csv("sp_db.csv")

x = data["sequence"]
y = data["goa"].apply(
    lambda k: [item for item in k.strip("[]").replace("'", "").split(', ') if item != '']
).values
seq_list = x.to_list()

label_list = set()
for lst in y:
    for label in lst:
        label_list.add(label)
print(len(label_list))


def compute_kmer_frequencies(sequence, k) -> dict:
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

    total_kmers = sum(kmer_counts.values())
    kmer_frequencies = {kmer: count / total_kmers for kmer, count in
                        kmer_counts.items()}
    return kmer_frequencies


kmer_freq = [compute_kmer_frequencies(seq, 1) for seq in seq_list]
vectorizer = DictVectorizer(sparse=False)
X_vectorized = vectorizer.fit_transform(kmer_freq)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y,
                                                    test_size=0.1,
                                                    random_state=1)


mlb = MultiLabelBinarizer(classes=list(label_list))

y_train = mlb.fit_transform(y_train)
print(X_test.shape, X_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_test = mlb.transform(y_test)

print(type(y_test))

y_train_labels = mlb.inverse_transform(y_train)
y_test_labels = mlb.inverse_transform(y_test)

# Convert to sets for easier comparison
train_label_set = {label for labels in y_train_labels for label in labels}
test_label_set = {label for labels in y_test_labels for label in labels}

# Find labels present in test set but not in training set
missing_labels = test_label_set - train_label_set

print(len(missing_labels))
print("Labels present in test but missing in train:", missing_labels)
