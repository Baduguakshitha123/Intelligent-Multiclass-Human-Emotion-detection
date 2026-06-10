import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import itertools

class ModelEvaluator:
    def __init__(self, model_trainer):
        self.trainer = model_trainer
        self.id2label = model_trainer.id2label
        
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # F1 Score plot
        axes[2].plot(epochs, history['train_f1'], 'b-', label='Training F1')
        axes[2].plot(epochs, history['val_f1'], 'r-', label='Validation F1')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Training and Validation F1 Score')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.id2label.values()),
                   yticklabels=list(self.id2label.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, labels, title="Class Distribution"):
        """Plot class distribution"""
        plt.figure(figsize=(12, 6))
        
        # Count occurrences
        unique, counts = np.unique(labels, return_counts=True)
        class_names = [self.id2label[i] for i in unique]
        
        # Bar plot
        bars = plt.bar(range(len(unique)), counts)
        plt.xticks(range(len(unique)), class_names, rotation=45, ha='right')
        plt.xlabel('Emotion Classes')
        plt.ylabel('Count')
        plt.title(title)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, test_results):
        """Generate comprehensive evaluation report"""
        report = f"""
        {'='*60}
        EMOTION DETECTION MODEL EVALUATION REPORT
        {'='*60}
        
        Model: DistilBERT (Fine-tuned)
        Number of Classes: {len(self.id2label)}
        
        PERFORMANCE METRICS:
        --------------------
        Test Accuracy:  {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)
        Test F1 Score:  {test_results['test_f1']:.4f} ({test_results['test_f1']*100:.2f}%)
        Test Loss:      {test_results['test_loss']:.4f}
        
        {'='*60}
        """
        
        print(report)
        
        # Save report to file
        with open('evaluation_report.txt', 'w') as f:
            f.write(report)
        
        return report