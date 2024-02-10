import psycopg2
import requests

conn = psycopg2.connect("dbname='yourdbname' user='yourusername' host='yourhost' password='yourpassword'")
conn.autocommit = False  # Important for working with large objects
cursor = conn.cursor()

alphafold_url_template = 'https://alphafold.ebi.ac.uk/api/prediction/{accession_id}'

def fetch_and_save_pdb(accession_id):
    alphafold_url = alphafold_url_template.format(accession_id=accession_id)
    alphafold_response = requests.get(alphafold_url)

    if alphafold_response.status_code == 200:
        pdb_url = alphafold_response.json()['pdbUrl']
        pdb_data_response = requests.get(pdb_url, allow_redirects=True)
        if pdb_data_response.status_code == 200:
            pdb_data = pdb_data_response.content
            lobj = conn.lobject(None, 'wb', 0)
            lobj_oid = lobj.oid
            lobj.write(pdb_data)
            lobj.close()
            return lobj_oid
    return None

def update_table_with_pdb_oid():
    # Fetch all rows to process
    cursor.execute('SELECT id, accession_id FROM your_table_name WHERE pdb_data_oid IS NULL;')
    rows = cursor.fetchall()

    for row_id, accession_id in rows:
        print(f"Processing: {accession_id}")
        lobj_oid = fetch_and_save_pdb(accession_id)
        if lobj_oid:
            cursor.execute('UPDATE your_table_name SET pdb_data_oid = %s WHERE id = %s;', (lobj_oid, row_id))
            conn.commit()
            print(f"Updated row {row_id} with PDB data OID: {lobj_oid}")
        else:
            print(f"Failed to fetch PDB data for accession ID {accession_id}")

# Run the update function
update_table_with_pdb_oid()

cursor.close()
conn.close()
