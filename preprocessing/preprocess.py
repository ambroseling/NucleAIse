import requests
from Bio.PDB import *
import numpy as np
import matplotlib.pyplot as plt

class Protein:
    def __init__(
        self, 
        sequence,
        name,
        accession_id,
        genes,
        organism,
        taxonomy,
        locations,
        structure,
        structure_error
    ):
        self.sequence = sequence
        self.name = name
        self.accession_id = accession_id
        self.genes = genes
        self.organism = organism
        self.taxonomy = taxonomy
        self.locations = locations
        self.structure = structure
        self.structure_error = structure_error
        self.goa = None

    def set_goa(self, goa):
        self.goa = goa


def get_proteins_by_organism(organism_id, limit=1000):
    uniprot_base_url = "https://rest.uniprot.org/uniprotkb/search?query=(reviewed:true)%20AND%20(organism_id:{organism_id})"
    uniprot_url = uniprot_base_url.format(organism_id=organism_id)
    uniprot_response = requests.get(uniprot_url)
    proteins = []
    for entry in uniprot_response.json()['results']:
        # Sequence
        sequence = entry['sequence']['value']
        print("Seqeuence: {sequence}".format(sequence=sequence))

        # Protein Name
        name = entry['proteinDescription']['recommendedName']['fullName']['value']
        print("Name: {name}".format(name=name))

        # Primary Accession ID
        accession_id = entry['primaryAccession']
        print("Accession: {accession_id}".format(accession_id=accession_id))

        # Gene Name
        genes = []
        print("Genes: ", end='')
        for gene in entry['genes']:
            genes.append(gene['geneName']['value'])
            print(gene['geneName']['value'], end=', ')
        print()

        # Organism Scientific Name
        organism = entry['organism']['scientificName']
        print("Organism: {organism}".format(organism=organism))

        # Taxonomy Lineage
        taxonomy = entry['organism']['lineage']
        print("Taxonomy: {taxonomy}".format(taxonomy=taxonomy))

        # Subcellular Locations
        locations = []
        print("Locations: ", end='')
        for comment in entry['comments']:
            if comment['commentType'] == "SUBCELLULAR LOCATION":
                for location in comment['subcellularLocations']:
                    locations.append(location['location']['value'])
                    print(location['location']['value'], end=', ')
        print()

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

        # Create Protein object
        prot = Protein(
            sequence=sequence,
            name=name,
            accession_id=accession_id,
            genes=genes,
            organism=organism,
            taxonomy=taxonomy,
            locations=locations,
            structure=structure,
            structure_error=structure_error
        )

        # Add GO Annotations
        go_annotations = []
        print("GO Annotations: ", end='')
        for reference in entry['uniProtKBCrossReferences']:
            if reference['database'] == 'GO':
                go_annotations.append(reference['id'])
                print(reference['id'], end=', ')
        print('\n')
        prot.set_goa(go_annotations)

        proteins.append(prot)   
        if len(proteins) >= limit:
            break


    return proteins


if __name__ == "__main__":
    get_proteins_by_organism(9606, limit=10)