import torch
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import esm 
from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5Model,T5EncoderModel
import re



test_proteins = [
('P48347','MENEREKQVYLAKLSEQTERYDEMVEAMKKVAQLDVELTVEERNLVSVGYKNVIGARRASWRILSSIEQKEESKGNDENVKRLKNYRKRVEDELAKVCNDILSVIDKHLIPSSNAVESTVFFYKMKGDYYRYLAEFSSGAERKEAADQSLEAYKAAVAAAENGLAPTHPVRLGLALNFSVFYYEILNSPESACQLAKQAFDDAIAELDSLNEESYKDSTLIMQLLRDNLTLWTSDLNEEGDERTKGADEPQDEN'),
('P41932','MSDTVEELVQRAKLAEQAERYDDMAAAMKKVTEQGQELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTEGSEKKQQLAKEYRVKVEQELNDICQDVLKLLDEFLIVKAGAAESKVFYLKMKGDYYRYLAEVASEDRAAVVEKSQKAYQEALDIAKDKMQPTHPIRLGLALNFSVFYYEILNTPEHACQLAKQAFDDAIAELDTLNEDSYKDSTLIMQLLRDNLTLWTSDVGAEDQEQEGNQEAGN'),
('Q20655','MSDGKEELVNRAKLAEQAERYDDMAASMKKVTELGAELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTEGSEKKQQMAKEYREKVEKELRDICQDVLNLLDKFLIPKAGAAESKVFYLKMKGDYYRYLAEVASGDDRNSVVEKSQQSYQEAFDIAKDKMQPTHPIRLGLALNFSVFFYEILNAPDKACQLAKQAFDDAIAELDTLNEDSYKDSTLIMQLLRDNLTLWTSDAATDDTDANETEGGN'),
('P46077','MAAPPASSSAREEFVYLAKLAEQAERYEEMVEFMEKVAEAVDKDELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEESRGNDDHVTTIRDYRSKIESELSKICDGILKLLDTRLVPASANGDSKVFYLKMKGDYHRYLAEFKTGQERKDAAEHTLTAYKAAQDIANAELAPTHPIRLGLALNFSVFYYEILNSPDRACNLAKQAFDEAIAELDTLGEESYKDSTLIMQLLRDNLTLWTSDMQDESPEEIKEAAAPKPAEEQKEI'),
('P48349','MAATLGRDQYVYMAKLAEQAERYEEMVQFMEQLVTGATPAEELTVEERNLLSVAYKNVIGSLRAAWRIVSSIEQKEESRKNDEHVSLVKDYRSKVESELSSVCSGILKLLDSHLIPSAGASESKVFYLKMKGDYHRYMAEFKSGDERKTAAEDTMLAYKAAQDIAAADMAPTHPIRLGLALNFSVFYYEILNSSDKACNMAKQAFEEAIAELDTLGEESYKDSTLIMQLLRDNLTLWTSDMQEQMDEA'),
('P31946','MTMDKSELVQKAKLAEQAERYDDMAAAMKAVTEQGHELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTERNEKKQQMGKEYREKIEAELQDICNDVLELLDKYLIPNATQPESKVFYLKMKGDYFRYLSEVASGDNKQTTVSNSQQAYQEAFEISKKEMQPTHPIRLGLALNFSVFYYEILNSPEKACSLAKTAFDEAIAELDTLNEESYKDSTLIMQLLRDNLTLWTSENQGDEGDAGEGEN'),
('P62258','MDDREDLVYQAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGEDKLKMIREYRQMVETELKLICCDILDVLDKHLIPAANTGESKVFYYKMKGDYHRYLAEFATGNDRKEAAENSLVAYKAASDIAMTELPPTHPIRLGLALNFSVFYYEILNSPDRACRLAKAAFDDAIAELDTLSEESYKDSTLIMQLLRDNLTLWTSDMQGDGEEQNKEALQDVEDENQ'),
]
data = [
('P46077','MAAPPASSSAREEFVYLAKLAEQAERYEEMVEFMEKVAEAVDKDELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEESRGNDDHVTTIRDYRSKIESELSKICDGILKLLDTRLVPASANGDSKVFYLKMKGDYHRYLAEFKTGQERKDAAEHTLTAYKAAQDIANAELAPTHPIRLGLALNFSVFYYEILNSPDRACNLAKQAFDEAIAELDTLGEESYKDSTLIMQLLRDNLTLWTSDMQDESPEEIKEAAAPKPAEEQKEI'),
]

# ESM 2
def get_esm(data):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33][0,1: results["representations"][33].shape[1]-1]
    print("=== ESM shape: ===")
    print(token_representations.shape)
    return token_representations
# # ProteinBERT

def get_bert(data):
    """
    Function to collect last hidden state embedding vector from pre-trained ProtBERT Model

    INPUTS:
    - sequence (str) : protein sequence (ex : AAABBB) from fasta file
    - len_seq_limit (int) : maximum sequence lenght (i.e nb of letters) for truncation

    OUTPUTS:
    - output_hidden : last hidden state embedding vector for input sequence of length 1024
    """
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    sequence_w_spaces = ' '.join(list(data[0][1]))
    encoded_input = tokenizer(
        sequence_w_spaces,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_tensors='pt')
    output = model(**encoded_input)
    output_hidden = output['last_hidden_state'][0,1: output['last_hidden_state'].shape[1]-1]
    print("==== ProteinBERT shape: ====")
    print(output_hidden.shape)
    return output_hidden


def get_t5(data):
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    # prepare your protein sequences as a list
    sequence_examples = [data[0][1]]
    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    # For feature extraction we recommend to use the encoder embedding
    emb = embedding_repr.last_hidden_state[0,:len(data[0][1])]
    print('==== T5 shape: ====')
    print(emb.shape)
    return emb


get_esm(data)
get_bert(data)
get_t5(data)

#PCA
X = torch.tensor([
    get_esm(data)[0],
    get_bert(data)[0],
    get_t5(data)[0]
])
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
plt.plot(X_transformed)
plt.show()

#tSNE
X_embedded = TSNE(n_components=2, learning_rate='auto',
             init='random', perplexity=3).fit_transform(X)