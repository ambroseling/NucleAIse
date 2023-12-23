import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, \
    zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC

data = pd.read_csv("all_data.csv")

x = data["sequence"]
y = data["goa"].apply(
    lambda k: [item for item in k.strip("[]").replace("'", "").split(', ')
               if item != '']).values

seq_list = x.to_list()
label_list = []  # all labels
for lst in y:
    for label in lst:
        if label not in label_list:
            label_list.append(label)


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

# Transforming the k-mer Frequencies into a Feature Matrix:
vector = DictVectorizer(sparse=False)
X_vectorized = vector.fit_transform(kmer_freq)

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y,
                                                    test_size=0.2,
                                                    random_state=1)


# Transforming target variable to a multilabel format
mlb = MultiLabelBinarizer(classes=label_list)
y_train = mlb.fit_transform(y_train)

# Standardization of feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Creating an SVM classifier
svm_classifier = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.5))

# Other possible kernels
# svm_classifier = OneVsRestClassifier(SVC(kernel='linear', C=1, random_state=67))
# svm_classifier = OneVsRestClassifier(SVC(kernel='poly', degree=3, coef0=1))  # 0.49603174603174605
# svm_classifier = OneVsRestClassifier(SVC(kernel='sigmoid', coef0=.2)) # 0.8421052631578947

# Training the SVM model on the training data
svm_classifier.fit(X_train, y_train)

# Making predictions on the test data
y_pred = svm_classifier.predict(X_test)
y_test = mlb.transform(y_test)

# Calculating accuracy and other evaluation metrics Accuracy: 0.9708994708994709
accuracy = accuracy_score(y_test, y_pred)
loss = hamming_loss(y_test, y_pred)
loss_ind = zero_one_loss(y_test, y_pred)
loss_whole = zero_one_loss(y_test, y_pred, normalize=False)

print(f'Hamming Loss: {loss}')  # Hamming Loss: 9.219777437219492e-05
print(f'Accuracy: {accuracy}')
print(f'Zero_One_loss while considering individual label: {loss_ind}')
print(f'Zero_One_loss while considering the entire set of labels: {loss_whole}')

# Report
# report = classification_report(y_test, y_pred, zero_division=0)
# print('Classification Report:')
# print(report)


# Plotting



