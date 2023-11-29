from Bio import SeqIO
import pandas as pd 
from read_fasta_sp import list_files_in_folder, extract
import requests, json
from read_fasta_50 import process_and_save_data_files
                                                           
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
    
    db = pd.concat(total_data).to_csv(filename)
    

def fasta_SPdata_only(handle): 
    protein_dict = pd.DataFrame(columns = ['ID', 'OX','OS','Genename','Sequence', "GOA"])
    protein_iter = SeqIO.FastaIO.FastaIterator(handle)
    id = []
    OX_val = []
    OS_val = []
    seq = []
    GN_val = []
    goa = []

    for p in protein_iter: 
        temp_id = p.id.split('|')
        accession_id = temp_id[1] if (len(temp_id) >2) else None

        uniprot_url = "https://rest.uniprot.org/uniprotkb/{accession_id}?format=json"
        url = uniprot_url.format(accession_id = accession_id) 
        try: 
            uniprot_response = requests.get(url, timeout = 20).json()
            if (uniprot_response.get('uniProtKBCrossReferences') != None) and len(p.seq) <= 10000: 
                for ref in uniprot_response['uniProtKBCrossReferences']:
                    if ref['database'] == 'GO':
                        goa.append(ref['id'])
        
                id.append(temp_id)
                seq.append(p.seq)
                os_value, ox_value, Gname = extract(p.description)
                OX_val.append(ox_value)
                OS_val.append(os_value)
                GN_val.append(Gname)

        except requests.exceptions.Timeout: 
            print('This request timed out')
        except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
            print("COnnection error")
      

    protein_dict['ID'] = id
    protein_dict['Sequence'] = seq
    protein_dict['OX'] = OX_val
    protein_dict['OS'] = OS_val
    protein_dict['Genename'] = GN_val 
    protein_dict['GOA'] = goa

    # protein_dict.to_csv(filename)
 
    
if __name__ == "__main__":
    # fasta_SPdata_only('preprocessing\\data\\batched_sp_db\\group_1.fastq')
    savefiles = list_files_in_folder("preprocessing\\data\\batched_sp_db")
    process_and_save_data_files(savefiles, fasta_SPdata_only, 'SP_data_fasta')
    # combine("processed_sp", 'sp_db.csv')
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