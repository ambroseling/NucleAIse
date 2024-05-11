import requests, json 
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import time
import re
import multiprocessing
import os 

def read_fasta(filename):
    protein_dict = SeqIO.to_dict(SeqIO.parse(filename, "fasta"))
    id_list = list(protein_dict.keys())
    for i in id_list:
        if len(i) > 9: # longest uniprot Uniq ID is 8 digits in sp only 
            if i[:2] == "sp": # data from Uniprot Swiss prot
                j = i[3:].index("|")
                temp = i[3:j+3]
                protein_dict[temp] = protein_dict[i] # replace with new key that is only
                del protein_dict[i]

    id_list_new = list(protein_dict.keys())
    return id_list_new, protein_dict

def repeated_protein(file1, file2):
    id_list1, protein_dict1 = read_fasta(file1)
    id_list2, protein_dict2 = read_fasta(file2)

    protein_dict1.update(protein_dict2)
    id_list = list(protein_dict1.keys())

    return id_list, protein_dict1

def fasta_loop(accession_id):# no longer needed
    uniprot_url = "https://rest.uniprot.org/uniprotkb/{ids}?format=fasta"
    url = uniprot_url.format(ids = accession_id) 
    uniprot_response = requests.get(url).text
        
        # proteins_dict = SeqIO.to_dict(uniprot_response)
    # uniprot_id = (uniprot_response.split('|'))
    OX_id =(uniprot_response.split('OX='))
    OS_name =(uniprot_response.split('OS='))
    Gene_name =(uniprot_response.split('GN='))
    seq =(uniprot_response.split('\n',maxsplit=1))
              
    if ((len(OX_id)>=2) and (len(OS_name) >= 2)
                       and (len(Gene_name)>=2)
                       and (len(seq) > 1)):
                       # and (len(seq[0] < 6000))):

        # uniprot_id = uniprot_id[1]
        OX_id = OX_id[1].split('GN')[0].replace(' ','')
        OS_name = OS_name[1].split('OX')[0].replace(' ','')
        seq = seq[1].replace('\n','')
        Gene_name = Gene_name[1].split('PE')[0].replace(' ','')

    return OX_id, OS_name, Gene_name, seq

def get_fasta(accession_id_list):# nolonger needed
    id_list = []
    OX_id_list = []
    OS_name_list = []
    Gene_name_list = []
    seq_list = []
    # print(accession_id_list)
    for i in accession_id_list:
        
        one, two, three, four,five = fasta_loop(i)
        id_list.append(one)
        OX_id_list.append(two)
        OS_name_list.append(three)
        Gene_name_list.append(four)
        seq_list.append(five)
    count = [len(id_list), len(OX_id_list), len(OS_name_list), len(Gene_name_list), len(seq_list)]

    return count, id_list, OX_id_list, OS_name_list, Gene_name_list, seq_list

def extract(sentence): # for sp only
    '''
    for sp ONLY, not for uniref50 
    '''
    #This function is used to extract OX,OS information from protein_dict->description
    # Define regular expressions to match the patterns "OS=..." and "OX=..."

    os_pattern = r'OS=([^OX]+)'
    ox_pattern = r'OX=([^\s]+)'
    Gname = r'GN=([^\s]+)'

    # Use regular expressions to find matches in the sentence
    os_match = re.search(os_pattern, sentence)
    ox_match = re.search(ox_pattern, sentence)
    Gname_match = re.search(Gname, sentence)
    # Check if matches were found and extract the values
    if os_match:
        os_value = os_match.group(1)
    else:
        os_value = None

    if ox_match:
        ox_value = ox_match.group(1)
    else:
        ox_value = None

    if Gname_match:
        Gname = Gname_match.group(1).strip()
    else:
        Gname = None 
    return os_value, ox_value, Gname

def get_protein_json(accession_id):
    '''
    this func was mostly adapted from preprocess.py

    we didn't get the residue cause the protein sequence is the residue, they can be translated through a table
    '''
    subcell_location = []
    goa = []
    interactant_id_one = []
    interactant_id_two = []
    uniprot_url = "https://rest.uniprot.org/uniprotkb/{accession_id}?format=json"
    url = uniprot_url.format(accession_id = accession_id) 
    try: 
        uniprot_response = requests.get(url, timeout = 20).json()
    except requests.exceptions.Timeout: 
        print('This request timed out')
        return subcell_location, goa, interactant_id_one,interactant_id_two
    except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
        print("COnnection error")
        return subcell_location, goa, interactant_id_one,interactant_id_two

    subcell_location = []
    goa = []
    interactant_id_one = []
    interactant_id_two = []
    comments = uniprot_response.get('comments') 
    KBC = uniprot_response.get('uniProtKBCrossReferences')

    # subcellular location
    if ((comments != None) and (KBC != None)): 
        for comment in uniprot_response['comments']:
            # if ((comment['commentType']  == "SUBCELLULAR LOCATION") and (comment.get('subcellularLocations'))): 
            if comment.get('subcellularLocations') != None: 
                for location in comment['subcellularLocations']: 
                    subcell_location.append(location['location']['value'])

            # interactant 
            # if ((comment['commentType']  == "INTERACTION") and (comment.get('interactions'))): 
            if comment.get('interactions') != None: 
                for interactant in comment['interactions']: 
                    one = interactant['interactantOne'].get('uniProtKBAccession')
                    two = interactant['interactantTwo'].get('uniProtKBAccession')
                    if one != None:
                        interactant_id_one.append(interactant['interactantOne']['uniProtKBAccession'])
                    if two != None:
                        interactant_id_two.append(interactant['interactantTwo']['uniProtKBAccession'])

        for ref in uniprot_response['uniProtKBCrossReferences']:
            if ref['database'] == 'GO':
                goa.append(ref['id'])

    return subcell_location, goa, interactant_id_one, interactant_id_two

def get_info(data,id):
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
    OS_value,OX_value, genename = extract(description_string)
    ox_data = {"OX":OX_value}
    sequence_data = {"sequence":sequence}
    os_data = {"OS":OS_value}
    genes = {"Gene name":genename}

  #   four_info = get_protein_json(id)

    subcell_location, goa, interactant_id_one,interactant_id_two = get_protein_json(id)

    if(len(subcell_location) != 0 and len(goa) != 0 
                                    and len(interactant_id_one) > 0
                                    and len(interactant_id_two) > 0 
                                    and len(sequence) <= 6000):
        subcell_location_data = {"subcell_location":subcell_location}
        goa_data = {"goa":goa}
        interactant_id_one_data = {"interactant_id_one":interactant_id_one}
        interactant_id_two_data = {"interactant_id_two":interactant_id_two}

        result_dict[id] =[ox_data,os_data, subcell_location_data,genes,sequence_data,goa_data,interactant_id_one_data,interactant_id_two_data] 

    return result_dict

def transfer_to_pandas(data):
    data_list = []

    for key, values in data.items():
        entry = {'ID': key}
        for item in values:
            entry.update(item)
        data_list.append(entry)

    # Create a DataFrame from the restructured data
    df = pd.DataFrame(data_list)
    # df.to_csv(savefile, index=False)
    return df

def all_data(files):
    id_list, proteins = read_fasta(files)
    whole_dict = {}

    for id in id_list:
        result_dict = get_info(proteins[id], id)
        if result_dict:  # Check if result_dict is not empty
            whole_dict.update(result_dict)
        # print(id)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    return transfer_to_pandas(whole_dict)

def list_files_in_folder(folder_path):
    file_paths = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return file_paths
    
    # Iterate through the files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths

def process_and_save_data_files(data_files, output_dir, num_processes=8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print(data_files)
    # Create a pool of worker processes.

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.map() to apply the process_data_file function to each data file.
        # The data_files list is split among worker processes, and the function is executed in parallel.
        processed_data_list = pool.map(all_data, data_files)
    # print(processed_data_list)
    # Save the processed data to files.

    for i, processed_data in enumerate(processed_data_list):
        if processed_data is not None: 
            output_file = os.path.join(output_dir, f'processed_{i}.csv')
            processed_data.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    
    # all_data('preprocessing\\New folder\\small_test.fasta' , 'test.csv')
    savefiles = list_files_in_folder("preprocessing/data/batched2_sp")
    print('start')
    # print(savefiles)
    # savefiles = ['preprocessing\\New folder\\small_test.fasta', 'preprocessi
    # ng\\New folder\\small_test.fasta']
    start = time.time()
    
    process_and_save_data_files(savefiles, 'sp_data',8)
    end = time.time() 
    print(end-start)