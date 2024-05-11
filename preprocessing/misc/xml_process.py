import xml.etree.ElementTree as ET
import pandas as pd
import requests
from Bio.PDB import *
import numpy as np
import matplotlib.pyplot as plt

# file =
def get_interactor_list(file):
    tree = ET.parse(file)
    root = tree.getroot()

    id_list = []
    interactorList_name = root.tag.replace('entrySet', 'interactorList')
    xref_name = root.tag.replace('entrySet', 'xref')

    for entry in root:
        for source in entry:
            if source.tag == interactorList_name:
                for interactant in source:
                    if interactant[1].tag == xref_name:
                        for ref in interactant[1]:
                            if ref.attrib['db'] == 'uniprotkb':
                                id_list.append(ref.attrib['id'])

    return id_list


def get_residue(id_list):
    for accession_id in id_list:
        # Protein Structure
        alphafold_base_url = "https://alphafold.ebi.ac.uk/api/prediction/{accession_id}?key=AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94"
        alphafold_url = alphafold_base_url.format(accession_id=accession_id)
        alphafold_response = requests.get(alphafold_url)
        # Get Residues from PDB 3D representation
        pdb_url = alphafold_response.json()[0]["pdbUrl"]
        pdb_response = requests.get(pdb_url, allow_redirects=True)
        with open('preprocessing/structure.pdb', 'wb') as pdb_file:
            pdb_file.write(pdb_response.content)

        # PDB Parser Docs (https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ)
        parser = PDBParser()
        structure = parser.get_structure(accession_id + ' Structure', 'preprocessing/structure.pdb')
        print("Residues: ", end='')
        # PDB Structure Docs (https://biopython.org/docs/1.75/api/Bio.PDB.Structure.html)
        for residue in structure.get_residues():
            # Residue Module Docs (https://biopython.org/docs/1.75/api/Bio.PDB.Residue.html)
            print(residue.get_resname(), end=', ')
        print()

        # Get Residue Error From AlphaFold Contact Map
        error_map_url = alphafold_response.json()[0]["paeDocUrl"]
        error_map_response = requests.get(error_map_url, allow_redirects=True)
        structure_error = error_map_response.json()[0]['predicted_aligned_error']
        print("Error Map: {rows}x{cols}".format(rows=len(structure_error), cols=len(structure_error[0])))

if __name__ == "__main__":
    all_interactors = []
    for i in range(8):  # Adjust the range based on how many files you have
        # Generate the filename
        filename = f"caeel_6239_0{i+1}.xml"
        # Get the interactor list for the current file
        current_interactors = get_interactor_list(filename)

        # Extend the main list with the interactors from the current file
        all_interactors.extend(current_interactors)
        if(i == 5):
            all_interactors.extend(get_interactor_list("caeel_6239_06_negative.xml"))

    for i in range(9,21):  # Adjust the range based on how many files you have
        # Generate the filename
        filename = f"caeel_6239_{i+1}.xml"
        # Get the interactor list for the current file
        current_interactors = get_interactor_list(filename)

        # Extend the main list with the interactors from the current file
        all_interactors.extend(current_interactors)

    # for i in range(len(all_interactors)):
        # print(i)
        # print(all_interactors[i])


    output_file = "caeel.txt"

    # Write to the output file
    with open(output_file, 'w') as f:
        for interactor in all_interactors:
            f.write(f"{interactor}\n")

    print(f"Interactors have been written to {output_file}")







#print(root.tag)


#print(root[0][0].tag)

