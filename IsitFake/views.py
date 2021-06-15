from django.shortcuts import render
from django.http import HttpResponse
from . import Checker
import joblib
import json
# Create your views here.
# f = open('IsitFake/Models/tfidf_vocab.json')
# Vocab = json.load(f)
#f.close()
#Vocab = joblib.load(open("IsitFake/Models/tfidf_vocab.pkl", "rb"))
def index(request):
    print("-------------------------------",request.POST)
    news,val = Checker.extractor(request.POST)
    return render(request,"IsitFake/index.html",{
        'isIt':val,'news':news
    })