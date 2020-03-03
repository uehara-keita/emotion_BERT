import re
import string
import torch
import torchtext
import pickle
from app_emo.bert import get_config, load_vocab, BertModel, BertTokenizer, BertForIMDb, set_learned_params
from app_emo.config import *

def pickle_dump(TEXT, path):
    with open(path, 'wb') as f:
        pickle.dump(TEXT, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        TEXT = pickle.load(f)
    return TEXT

def build_bert_model():

    # BERT設定ファイル
    config = get_config(file_path=BERT_CONFIG)
    # BERTモデル
    net_bert = BertModel(config)
    print("-" * 20 + "BERTモデル 設定完了" + "-" * 20)
    # 感情分析用　BERT
    net = BertForIMDb(net_bert)
    print("-" * 20 + "感情分析モデル 設定完了" + "-" * 20)
    # パラメーターセット
    net.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
    print("-" * 20 + "fine tuning 設定完了" + "-" * 20)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.eval()
    net.to(device)

    return net
# テキストの正規化
def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    # カンマ、ピリオド以外の記号をスペースに変換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, '')
    # 記号前後にスペースを挿入
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")

    return text

# 前処理+単語分割
def tokenizer_with_preprocessing(text):
    tokenizer_bert = BertTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=True
    )
    text = preprocessing_text(text)
    ret = tokenizer_bert.tokenize(text)

    return ret


def create_tensor(text, max_length, TEXT):
    token_ids = torch.ones((max_length)).to(torch.int64)
    ids_list = list(map(lambda x: TEXT.vocab.stoi[x], text))
    print(ids_list)
    for i, index in enumerate(ids_list):
        token_ids[i] = index
    return token_ids

# ボキャブラリーデータ(text.pkl)の生成
def create_vocab_text():
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True,
                                batch_first=True, fix_length=max_length, init_token="[CLS]", eos_token="[SEP]", pad_token="[PAD]", unk_token="[UNK]")
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=DATA_PATH, train='train_dumy.tsv', test='test_dumy.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)]
    )
    vocab_bert, ids_to_tokens_bert = load_vocab(vocab_file=VOCAB_FILE)
    TEXT.build_vocab(train_val_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert
    pickle_dump(TEXT, PKL_FILE)

    return TEXT

# 入力文章の前処理
def convert_to_model_format(input_seq, TEXT):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 入力文章の単語分割
    input_text = tokenizer_with_preprocessing(input_seq)
    # 文章の先頭に[CLS]を追加
    input_text.insert(0, "[CLS]")
    # 文章の末尾に[SEP]を追加
    input_text.append("[SEP]")
    # テンソルサイズの変形
    text = create_tensor(input_text, max_length, TEXT)
    # torch.Size([256]) -> torch.Size([1, 256])
    text = text.unsqueeze_(0)
    input = text.to(device)

    return input

# 入力文章の推論
def predict_preds(input_text, net):
    # vocabデータをロード
    TEXT = pickle_load(PKL_FILE)
    input = convert_to_model_format(input_text, TEXT)
    input_pad = 1
    input_mask = (input != input_pad)
    outputs, attention_probs = net(input, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False,
                                   attention_show_flg=True)
    _, preds = torch.max(outputs, 1)
    return preds