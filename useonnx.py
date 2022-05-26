import torch
import numpy as np
import onnxruntime
from generals import get_master_seps,cos_sim
data = {'input_ids': np.array([[ 101, 2179, 2774, 5718, 6305, 6670, 7637,  102]],dtype=np.int64),
        'attention_mask': np.array([[1, 1, 1, 1, 1, 1, 1, 1]],dtype=np.int64)}
onnx_model = onnxruntime.InferenceSession('./onnx/model.onnx')
inputs = {onnx_model.get_inputs()[0].name:data['input_ids'],
          onnx_model.get_inputs()[1].name: data['attention_mask']}
outputs = onnx_model.run(None,inputs)
data_rep = torch.from_numpy(outputs[-1])
database,databasenames = get_master_seps()
cosin_sim = cos_sim(data_rep,database)
print('最符合要求的是：',databasenames[np.argmax(cosin_sim)],end=' ')
print('相似度分别为：',cosin_sim[0])