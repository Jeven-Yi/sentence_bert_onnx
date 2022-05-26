import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('pretrained/distilbert-multilingual-nli-stsb-quora-ranking')
data = {'input_ids': torch.tensor([[ 101,2179, 2774, 5718, 6305, 6670, 7637,  102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
torch.onnx.export(model,data,'./onnx/model.onnx',opset_version=11,verbose=True)
# sentence = '人名的红色车'
# sentence_rep = model.encode(sentence,convert_to_tensor=True)
# print(sentence_rep.shape)