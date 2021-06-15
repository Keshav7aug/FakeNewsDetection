import joblib
import os
print(os.getcwd())
print('LOADING')
with open('IsitFake/myData/Models/tokenizer.pkl', 'rb') as handle:
        tokenizer = joblib.load(handle)