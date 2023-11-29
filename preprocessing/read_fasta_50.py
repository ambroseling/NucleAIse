import requests
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import time
import re
import multiprocessing
from read_fasta_sp import get_protein_json, transfer_to_pandas, list_files_in_folder
import os 

def read_fasta_50(filename):
    protein_dict = SeqIO.to_dict(SeqIO.parse(filename, "fasta"))
    id_list = list(protein_dict.keys())
    for i, id in enumerate(id_list):
        id_list[i] = id[9:] 
        protein_dict[id_list[i]] = protein_dict.pop(id)
    
    id_list_new = list(protein_dict.keys())
    return id_list_new, protein_dict

def extract_50(sentence): 
    ox_pattern = r'TaxID=([^\s]+)'
    os_pattern = r'Tax=(.*?)(?: TaxID|$)'

    # Use regular expressions to find matches in the sentence
    os_match = re.search(os_pattern, sentence)
    ox_match = re.search(ox_pattern, sentence)
    # Gname_match = re.search(Gname, sentence)
    # Check if matches were found and extract the values
    if os_match:
        os_value = os_match.group(1)
    else:
        os_value = None

    if ox_match:
        ox_value = ox_match.group(1)
    else:
        ox_value = None

    return os_value, ox_value


def get_info_50(data,id):
    '''We have ids and proteins. Proteins is a dictionary with id as key, ID/Name/Description...as value. The goal of this function is to form a dictionary of dictionary. The format is like this:
    {‘ID’:{‘OX’: OX_number}, {’sequence’: sequence_string}, {‘OS’: OS_number}.}
    Inputs: id is the key, such as P79742. the type of data is SeqRecord, which is the same as proteins
    '''
    result_dict = {}
    description_string = data.description
    seq_object = data.seq
    sequence = str(seq_object)
    # start_index = seq_to_string.index("'") + 1
    # end_index = seq_to_string.rindex("'")
    #
    # # Extract the substring between single quotes
    # sequence = seq_to_string[start_index:end_index]
    OS_value,OX_value,  = extract_50(description_string)
    ox_data = {"OX":OX_value}
    sequence_data = {"sequence":sequence}
    os_data = {"OS":OS_value}

    subcell_location, goa, interactant_id_one,interactant_id_two = get_protein_json(id)

    if(len(goa) != 0 and (len(interactant_id_one) > 0 or len(interactant_id_two) > 0 )
                     and len(sequence) <= 6000):
        subcell_location_data = {"subcell_location":subcell_location}
        goa_data = {"goa":goa}
        interactant_id_one_data = {"interactant_id_one":interactant_id_one}
        interactant_id_two_data = {"interactant_id_two":interactant_id_two}

        result_dict[id] =[ox_data,os_data, subcell_location_data,sequence_data,goa_data,interactant_id_one_data,interactant_id_two_data] 

    return result_dict


def all_data_50(files):
    id_list, proteins = read_fasta_50(files)
    whole_dict = {}

    for id in id_list:
        result_dict = get_info_50(proteins[id], id)
        if result_dict:  # Check if result_dict is not empty
            whole_dict.update(result_dict)
        # print(id)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    return transfer_to_pandas(whole_dict)

def process_and_save_data_files(data_files, output_dir, num_processes=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a pool of worker processes.
    with multiprocessing.Pool(processes=num_processes) as pool:
        processed_data_list = pool.map(all_data_50, data_files)
    # print(processed_data_list)
    # Save the processed data to files.
    for i, processed_data in enumerate(processed_data_list):
        if processed_data is not None: 
            output_file = os.path.join(output_dir, f'processed_{i}.csv')
            processed_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    # id_list, protein_dict = read_fasta_50('group_1.fasta')

    savefiles = list_files_in_folder("preprocessing\\data\\uniref50_group2")
    print('start')
    # print(savefiles)
    # savefiles = ['preprocessing\\New folder\\small_test.fasta', 'preprocessi
    # ng\\New folder\\small_test.fasta']
    start = time.time()
    
    process_and_save_data_files(savefiles, 'uniref_data',8)
    end = time.time() 
    print(end-start)