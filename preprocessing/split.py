from Bio import SeqIO
import pandas as pd 
from read_fasta_sp import list_files_in_folder, extract
import random
from read_fasta_50 import process_and_save_data_files
import matplotlib as plt
                                                           
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


def combine(folder, filename): 
    file_paths = list_files_in_folder(folder)
    total_data = []
    for path in file_paths: 
        total_data.append(pd.read_csv(path))
    
    pd.concat(total_data).to_csv(filename)
    
def fasta_data_only(handle): 
    protein_dict = pd.DataFrame(columns = ['ID', 'OX','OS','Genename','Sequence', "GOA"])
    protein_iter = SeqIO.FastaIO.FastaIterator(handle)
    id = []
    OX_val = []
    OS_val = []
    seq = []
    GN_val = []

    for p in protein_iter: 
        temp_id = p.id.split('|')
        
        id.append(temp_id)
        seq.append(p.seq)
        os_value, ox_value, Gname = extract(p.description)
        OX_val.append(ox_value)
        OS_val.append(os_value)
        GN_val.append(Gname)

    protein_dict['ID'] = id
    protein_dict['Sequence'] = seq
    protein_dict['OX'] = OX_val
    protein_dict['OS'] = OS_val
    protein_dict['Genename'] = GN_val 

    # protein_dict.to_csv(filename)

def GOA_split(filename, num_prot_per_goa=100,num_GOA=100):
    # Load the CSV file into a DataFrame
    csv_file_path = filename
    data = pd.read_csv(csv_file_path)

    # Split 'goa' column into a list of items, then explode the dataframe
    goa_data = data['goa'].str.strip("[]").str.replace("'", "").str.split(', ')
    goa_exploded = goa_data.explode()

    # Counting the occurrences of each goa item
    goa_counts = goa_exploded.value_counts()

    # Select the first 100 most frequent GOA terms
    top_goa = goa_counts.head(num_GOA)

    # Finding proteins associated with each of the top 100 GOA terms
    protein_associations = {}
    all_proteins_set = set()

    for goa_term in top_goa.index:
        num_prot_in_goa = num_prot_per_goa 
        proteins = data[data['goa'].str.contains(f"\\b{goa_term}\\b", regex=True)]['ID']
        unique_proteins = proteins.tolist()
        protein_associations[goa_term] = unique_proteins

        if len(unique_proteins) < num_prot_per_goa: 
            num_prot_in_goa = len(unique_proteins)

        random.shuffle(unique_proteins)
        all_proteins_set.update(unique_proteins[:num_prot_in_goa])

    # Convert the set of all proteins to a list to get a list of unique proteins
    all_unique_proteins = list(all_proteins_set)
    filtered_df = pd.DataFrame()
    
    filtered_df = data[data['ID'].isin(all_unique_proteins)]
    filtered_df.to_csv('filtered.csv')
    return filtered_df                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

if __name__ == "__main__":
    GOA_split("sp_db.csv", num_GOA=10)

    # fasta_SPdata_only('preprocessing\\data\\batched_sp_db\\group_1.fastq')
    # savefiles = list_files_in_folder("processed_50\\group7")
    # process_and_save_data_files(savefiles, fasta_data_only, 'SP_data_fasta')
    # combine("processed_50\\uniref50", 'uniref50_db.csv')
    '''     filename = 'uniref50.fasta'
    record_iter = SeqIO.parse(open(filename), "fasta")
    batch_size = 2000000

    for i, batch in enumerate(batch_iterator(record_iter, batch_size)): 
        output_files = "50group_%i.fasta" % (i + 1)
        with open(output_files, "w") as handle:
            count = SeqIO.write(batch, handle, "fasta")
    '''


# sp have 570157 proteins 
# kaggle have 142246