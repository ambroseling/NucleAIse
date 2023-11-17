from Bio import SeqIO
import pandas as pd 
                                                           
# from SeqIO.website         
def batch_iterator(fasta_iter, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.Align.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """

    batch = []
    for entry in fasta_iter:
        batch.append(entry)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def length_iter(filename):
    count = 0
    with open(filename,'r') as handle: 
        for i in SeqIO.FastaIO.FastaIterator(handle): 
            count += 1
    return count 



if __name__ == "__main__":
    filename = 'preprocessing\\data\\uniref50_batched\\group_1.fastq'
    record_iter = SeqIO.parse(open(filename), "fasta")
    batch_size = 30000

    for i, batch in enumerate(batch_iterator(record_iter, 5000)):
        output_files = "group_%i.fasta" % (i + 1)
        with open(output_files, "w") as handle:
            count = SeqIO.write(batch, handle, "fasta")


# sp have 570157 proteins 
# kaggle have 142246