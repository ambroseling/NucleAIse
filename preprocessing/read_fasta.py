import requests
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import time
import re

def read_fasta(filename):
    '''
    Based on the data we gather, from kaggle(had been cleaned), UniRef50(consists most of our data), and Uniprot SP(Swiss prot)；
    they are all quite different, pls don't blame me for hard coding some of it
    '''
    protein_dict = SeqIO.to_dict(SeqIO.parse(filename, "fasta"))
    id_list = list(protein_dict.keys())
    for i in id_list:
        if len(i) > 9: # longest uniprot Uniq ID is 8 digits
            if i[:2] == "sp": # data from Uniprot Swiss prot
                j = i[3:].index("|")
                temp = i[3:j+3]
                protein_dict[temp] = protein_dict[i] # replace with new key that is only
                del protein_dict[i]

            elif i[:9] == "UniRef50_": # data from uniref50, doesn'have this data set yet, don't know the format yet, it's download
                continue

    id_list_new = list(protein_dict.keys())
    return id_list_new, protein_dict


def repeated_protein(file1, file2):
    id_list1, protein_dict1 = read_fasta(file1)
    id_list2, protein_dict2 = read_fasta(file2)

    protein_dict1.update(protein_dict2)
    id_list = list(protein_dict1.keys())

    return id_list, protein_dict1

def extract(sentence): #This function is used to extract OX,OS information from protein_dict->description
    # Define regular expressions to match the patterns "OS=..." and "OX=..."
    os_pattern = r'OS=([^\s]+)'
    ox_pattern = r'OX=([^\s]+)'

    # Use regular expressions to find matches in the sentence
    os_match = re.search(os_pattern, sentence)
    ox_match = re.search(ox_pattern, sentence)
    # Check if matches were found and extract the values
    if os_match:
        os_value = os_match.group(1)
    else:
        os_value = None

    if ox_match:
        ox_value = ox_match.group(1)
    else:
        ox_value = None
    return os_value,ox_value

def get_protein_json(accession_id):
    '''
    this func was mostly adapted from preprocess.py

    we didn't get the residue cause the protein sequence is the residue, they can be translated through a table
    '''
    uniprot_url = "https://rest.uniprot.org/uniprotkb/{accession_id}?format=json"
    url = uniprot_url.format(accession_id = accession_id) 
    uniprot_response = requests.get(url).json()

    subcell_location = []
    goa = []
    interactant_id_one = []
    interactant_id_two = []
    motif_pair = []
    comments = uniprot_response.get('comments') 

    # subcellular location
    if comments != None: 
        for comment in uniprot_response['comments']:
            if comment['commentType']  == "SUBCELLULAR LOCATION": 
                for location in comment['subcellularLocations']: 
                    subcell_location.append(location['location']['value'])
            
            # interactant 
            if comment['commentType']  == "INTERACTION": 
                for interactant in comment['interactions']: 
                    one = interactant['interactantOne'].get('uniProtKBAccession')
                    two = interactant['interactantTwo'].get('uniProtKBAccession')
                    if one != None:
                        interactant_id_one.append(interactant['interactantOne']['uniProtKBAccession'])
                    if two != None:
                        interactant_id_two.append(interactant['interactantTwo']['uniProtKBAccession'])

    else:
        subcell_location = []
        interactant_id_one = []
        interactant_id_two =[]
    
    KBC = uniprot_response.get('uniProtKBCrossReferences')
    motif = uniprot_response.get('features')
    # GOA
    if KBC != None: 
        for ref in uniprot_response['uniProtKBCrossReferences']:
            if ref['database'] == 'GO':
                goa.append(ref['id'])
    else: 
        goa = []
    
    if motif != None: 
        for motif in uniprot_response['features']: 
            if motif['type'] == 'Motif': 
                motif_loc = [motif['location']['start']['value'], motif['location']['end']['value']]
                motif_pair.append(motif_loc)
    else: 
        motif_pair = [0]


    return subcell_location, goa, interactant_id_one, interactant_id_two, motif_pair

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
    pair = extract(description_string)
    OS_value,OX_value = pair
    ox_data = {"OX":OX_value}
    sequence_data = {"sequence":sequence}
    os_data = {"OS":OS_value}
    four_info = get_protein_json(id)
    subcell_location, goa, interactant_id_one,interactant_id_two, motif = four_info#！！！！！！！！！！！！！！！！！！！第一个，把motif加进去

    if(len(subcell_location) != 0 and len(goa) != 0 
                                  and len(interactant_id_one) > 0
                                  and len(interactant_id_two) > 0 
                                  and len(sequence) <= 5000):#第二个！！！！！！！！！！！！！！！！！把 and （len motif = 0）加进去
        subcell_location_data = {"subcell_location":subcell_location}
        goa_data = {"goa":goa}
        interactant_id_one_data = {"interactant_id_one":interactant_id_one}
        interactant_id_two_data = {"interactant_id_two":interactant_id_two}
        motif_data = {"motif_loc":motif}

        #第三个！！！！！！！！！！！！！！！！！！！！！1 motif_data = {},然后下一行把motif加进result_dict
        result_dict[id] =[ox_data,sequence_data,os_data,subcell_location_data,goa_data,interactant_id_one_data,interactant_id_two_data, motif_data] #number 4 ！！！！！！！！！！！！！！！！！！！！这儿！


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
    df.to_csv('sample_data.csv', index=False)
    return df

def all_data(ids, proteins):
    whole_dict = {}
    for id in ids:
        result_dict = get_info(proteins[id], id)
        if result_dict:  # Check if result_dict is not empty
            whole_dict.update(result_dict)
    return transfer_to_pandas(whole_dict)


if __name__ == "__main__":
    start=time.time()
    id_list, protein_dict = read_fasta("small_test.fasta")
    mid = time.time()
    # ids,proteins= repeated_protein("small_test.fasta", "small_test.fasta")

    # print(proteins[ids[0]].description)
    # print(extract(proteins[ids[0]].description))
    # print(get_info(proteins[ids[1]],ids[1]))
    # print(transfer_to_pandas(get_info(proteins[ids[0]],ids[0])))
    print(all_data(id_list, protein_dict))
    end = time.time() 
    print("Read Fasta take:", (mid-start), '/n', "All data took:" ,(end - mid), '/n', 'Over all took:' ,(end - start))
