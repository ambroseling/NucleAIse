# Encoding sequences
from sklearn.feature_extraction import DictVectorizer


def compute_frequencies(sequence) -> dict:
    kmer_counts = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0,
                   'I': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0,
                   'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}
    for ch in sequence:
        if ch in kmer_counts.keys():
            kmer_counts[ch] += 1
        elif ch == 'X':
            for k in kmer_counts:
                kmer_counts[k] += 1
        elif ch == 'B':
            kmer_counts['D'] += 1
            kmer_counts['N'] += 1
        elif ch == 'Z':
            kmer_counts['E'] += 1
            kmer_counts['Q'] += 1
        else:
            kmer_counts['I'] += 1
            kmer_counts['L'] += 1

    total_kmers = sum(kmer_counts.values())
    kmer_frequencies = {kmer: count / total_kmers for kmer, count in
                        kmer_counts.items()}
    return kmer_frequencies


def encoding_sequence(seq_list):
    kmer_freq = [compute_frequencies(seq) for seq in seq_list]
    # Transforming the k-mer Frequencies into a Feature Matrix:
    vector = DictVectorizer(sparse=False)
    x_vectorized = vector.fit_transform(kmer_freq)
    return x_vectorized

