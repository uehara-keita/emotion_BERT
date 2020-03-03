from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.response import  Response
from app_emo.serializers import BertPredictSerializer

class BertPredictAPIView(generics.GenericAPIView):
    # BERT分類予測クラス
    serializer_class = BertPredictSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(request.data)
        return Response(serializer.data, status=status.HTTP_200_OK)