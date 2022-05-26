import torch
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('pretrained/distilbert-multilingual-nli-stsb-quora-ranking')
master_dict = [
                 '人名',
                 '人名的车',
                 '交通事件',
                 '交通事件的人名',
                 '交通事件的车'
                 ]
def get_master_seps():
    master_dict_representation = model.encode(master_dict, convert_to_tensor=True)
    return master_dict_representation,master_dict

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
