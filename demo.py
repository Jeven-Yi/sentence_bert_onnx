from sentence_transformers import SentenceTransformer
# 保存预训练模型的文件夹
save_path = "./pretrained/distilbert-multilingual-nli-stsb-quora-ranking/"
#
# 加载模型
model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')
# 保存模型到指定文件夹
model.save(save_path)
mode = SentenceTransformer(save_path)
print('OK')