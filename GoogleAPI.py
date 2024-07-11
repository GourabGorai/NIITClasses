import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator


def translate_text(text, target_lang, service='google', api_key=None):
    try:
        if service == 'google':
            # Automatically detect source language
            translator = GoogleTranslator(target=target_lang)
            translated_text = translator.translate(text)
        elif service == 'microsoft' and api_key:
            # Microsoft Translator requires an API key
            from deep_translator import MicrosoftTranslator
            # Automatically detect source language
            translator = MicrosoftTranslator(api_key=api_key, target=target_lang)
            translated_text = translator.translate(text)
        else:
            raise ValueError("Unsupported translation service or missing API key for Microsoft Translator")

        return translated_text

    except Exception as e:
        return f"An error occurred: {str(e)}"


data=pd.read_csv('orders.csv')
mdata=pd.DataFrame(data)
print(mdata.head(2000))
mdata=mdata.head(2000)
print(mdata.isna().sum())
mdata['spec'].fillna('unknown',inplace=True)
print(mdata.isna().sum())
l=[]
for i in mdata['spec']:
    text=i
    l.append(translate_text(text,'en'))
print(l)
mdata['spec']=l
mdata.to_csv('orders.csv',index=False)
