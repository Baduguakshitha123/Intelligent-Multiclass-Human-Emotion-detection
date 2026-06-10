import torch
import transformers
import pandas as pd
import numpy as np
import nltk
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def test_installations():
    
    # Test Transformers
    print(f"\nTransformers version: {transformers.__version__}")
    print("Testing DistilBERT loading...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        print("Transformers working correctly")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test NLTK
    try:
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        print(f" NLTK working (loaded {len(stop_words)} stopwords)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test Pandas
    print(f"\nPandas version: {pd.__version__}")
    test_df = pd.DataFrame({'test': [1,2,3]})
    print(f" Pandas working (created test DataFrame)")
    
    print("\n" + "="*50)
    print("Installation Test Complete")
    print("="*50)

if __name__ == "__main__":
    test_installations()