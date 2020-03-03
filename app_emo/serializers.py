from rest_framework import serializers
from app_emo.config import *
from app_emo.predict import predict_preds
from app_emo.bert import get_config, BertModel, BertForIMDb
import torch

class BertPredictSerializer(serializers.Serializer):
    input_text = serializers.CharField()
    emotion = serializers.SerializerMethodField()

    def get_emotion(self, obj):
        # Bertモデルの各種設定
        config = get_config(file_path=BERT_CONFIG)
        # Bertモデル
        net_bert = BertModel(config)
        # ネガポジ用の全結合層を追加
        net = BertForIMDb(net_bert)
        # ハイパーパラメータのロード
        net.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
        # 検証モード
        net.eval()
        # 推論結果の取得
        label = predict_preds(obj['input_text'], net).numpy()[0]

        return label

