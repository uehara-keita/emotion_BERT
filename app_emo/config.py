from pathlib import Path

BASE_DIR = Path("/Users/uta/PycharmProjects/emotion_bert/app_emo")
VOCAB_FILE = BASE_DIR / "vocab/eng/bert-base-uncased-vocab.txt"
BERT_CONFIG = BASE_DIR / "weights/eng/bert_config.json"
model_file = BASE_DIR / "weights/eng/pytorch_model.bin"
MODEL_FILE = BASE_DIR / "weights/eng/bert_fine_tuning_IMDb.pth"
PKL_FILE = BASE_DIR / "data/eng/text.pkl"
DATA_PATH = BASE_DIR / "data/eng"
max_length = 256

