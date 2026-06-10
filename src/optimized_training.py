import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

class OptimizedEmotionTrainer:
    def __init__(self, model_name='distilbert-base-uncased', num_classes=10):
        self.model_name = model_name
        self.num_classes = num_classes
          # Set device FIRST
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")      # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Initialize tokenizer and model
        print("Loading DistilBERT tokenizer and model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        self.label2id = None
        self.id2label = None
        
    def create_optimized_loaders(self, train_texts, train_labels, val_texts, val_labels, 
                                 test_texts=None, test_labels=None,
                                 batch_size=32, max_length=64):
        """Create optimized data loaders with better performance"""
        
        print("\n Creating optimized data loaders...")
        print(f"Batch size: {batch_size}, Max length: {max_length}")
        
        # Create label mappings
        unique_labels = sorted(set(train_labels))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        print(f"Classes: {list(self.label2id.keys())}")
        
        # Convert labels to ids
        train_labels_ids = [self.label2id[l] for l in train_labels]
        val_labels_ids = [self.label2id[l] for l in val_labels]
        
        # Tokenize all texts at once (much faster!)
        print("Tokenizing training data...")
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        print("Tokenizing validation data...")
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Create tensor datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels_ids)
        )
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(val_labels_ids)
        )
        
        # Create loaders with optimizations
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for MPS
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Handle test data if provided
        if test_texts is not None and test_labels is not None:
            test_labels_ids = [self.label2id[l] for l in test_labels]
            test_encodings = self.tokenizer(
                test_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            test_dataset = TensorDataset(
                test_encodings['input_ids'],
                test_encodings['attention_mask'],
                torch.tensor(test_labels_ids)
            )
            
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            print(f"Test batches: {len(self.test_loader)}")
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
    def train_epoch(self, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Store predictions
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating', leave=False):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        return avg_loss, accuracy, f1, predictions, true_labels
    
    def train(self, num_epochs=3, learning_rate=3e-5):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(optimizer, scheduler)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            
            # Validation
            val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                print(f"New best model! Val F1: {val_f1:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with Val F1: {best_val_f1:.4f}")
        
        return history
    
    def evaluate_test_set(self):
        """Final evaluation on test set"""
        if not hasattr(self, 'test_loader'):
            print(" No test loader found. Skipping test evaluation.")
            return None
        
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        
        test_loss, test_acc, test_f1, predictions, true_labels = self.evaluate(self.test_loader)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test F1 Score (weighted): {test_f1:.4f} ({test_f1*100:.2f}%)")
        
        # Detailed classification report
        target_names = [self.id2label[i] for i in range(len(self.id2label))]
        print("\nDetailed Classification Report:")
        report = classification_report(true_labels, predictions, target_names=target_names)
        print(report)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'predictions': predictions,
            'true_labels': true_labels,
            'report': report
        }
    
    def save_model(self, save_dir='models'):
        """Save model and tokenizer"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f"emotion_model_{timestamp}")
        
        # Save model
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save label mappings
        with open(os.path.join(model_path, 'label_mappings.json'), 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, indent=2)
        
        print(f"Model saved to {model_path}")
        return model_path