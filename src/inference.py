import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os

class EmotionPredictor:
    def __init__(self, model_path):
        """
        Load trained model for inference
        model_path: path to saved model directory
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        with open(os.path.join(model_path, 'label_mappings.json'), 'r') as f:
            mappings = json.load(f)
            self.label2id = mappings['label2id']
            self.id2label = mappings['id2label']
    
    def predict(self, text, return_probabilities=False):
        """
        Predict emotion for a single text
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        
        predicted_emotion = self.id2label[str(predicted_class_id)]
        
        if return_probabilities:
            # Get top 3 predictions
            probs = probabilities.cpu().numpy()[0]
            top_3_idx = probs.argsort()[-3:][::-1]
            top_3 = [(self.id2label[str(i)], probs[i]) for i in top_3_idx]
            return predicted_emotion, top_3
        else:
            return predicted_emotion
    
    def predict_batch(self, texts):
        """
        Predict emotions for multiple texts
        """
        predictions = []
        for text in texts:
            pred = self.predict(text)
            predictions.append(pred)
        return predictions
    
    def analyze_text(self, text):
        """
        Comprehensive analysis of a single text
        """
        predicted_emotion, top_3 = self.predict(text, return_probabilities=True)
        
        print(f"\n{'='*50}")
        print(f"Text: {text}")
        print(f"{'='*50}")
        print(f"Predicted Emotion: {predicted_emotion}")
        print("\nTop 3 Predictions:")
        for emotion, prob in top_3:
            print(f"  {emotion}: {prob:.4f} ({prob*100:.2f}%)")
        
        return predicted_emotion, top_3