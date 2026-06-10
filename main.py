import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from src.data_preprocessing import EmotionDataPreprocessor
from src.optimized_training import OptimizedEmotionTrainer

def main():
    print("INTELLIGENT TEXT ANALYTICS SYSTEM")
    print("Multiclass Human Emotion Detection")
    
    # 1. Data Preprocessing with FULL cleaning pipeline
    print("\nPHASE 1: DATA PREPROCESSING & CLEANING")
    
    data_paths = [
        'data/goemotions_1.csv',
        'data/goemotions_2.csv',
        'data/goemotions_3.csv'
    ]
    
    # Initialize preprocessor
    preprocessor = EmotionDataPreprocessor(data_paths)
    
    # Run complete preprocessing pipeline and SAVE cleaned data
    cleaned_df = preprocessor.prepare_dataset(save_cleaned=True)
    
    # Show saved files
    print("\nSaved Files:")
    print("  • cleaned_data/cleaned_emotion_dataset.csv (Full dataset)")
    print("  • cleaned_data/sample_cleaned_emotion_dataset.csv (First 100 rows)")
    print("  • cleaned_data/dataset_stats.json (Statistics)")
    print("  • cleaned_data/class_distribution.csv (Class distribution)")
    
    # Split data
    splits = preprocessor.split_data(cleaned_df)
    
    # 2. Use full cleaned dataset
    print("\nPreparing final training data...")
    
    train_texts = splits['train'][0]
    train_labels = splits['train'][1]
    val_texts = splits['val'][0]
    val_labels = splits['val'][1]
    test_texts = splits['test'][0]
    test_labels = splits['test'][1]
    
    print(f"\nFINAL CLEANED DATASET STATISTICS:")
    print(f"  • Training samples:   {len(train_texts):,}")
    print(f"  • Validation samples: {len(val_texts):,}")
    print(f"  • Test samples:       {len(test_texts):,}")
    print(f"  • Total cleaned:      {len(cleaned_df):,}")
    
    # 3. Train on CLEANED data
    print("\nPHASE 2: MODEL TRAINING ON CLEANED DATA")
    
    trainer = OptimizedEmotionTrainer(
        model_name='distilbert-base-uncased',
        num_classes=len(cleaned_df['emotion_10'].unique())
    )
    
    trainer.create_optimized_loaders(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        batch_size=32,
        max_length=64
    )
    
    # Train on clean data
    history = trainer.train(
        num_epochs=5,
        learning_rate=3e-5
    )
    
    # 4. Evaluate on clean test set
    print("\nPHASE 3: FINAL EVALUATION ON CLEAN DATA")
    print("-"*70)
    test_results = trainer.evaluate_test_set()
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs("results", exist_ok=True)

    cm = confusion_matrix(
        test_results['true_labels'],
        test_results['predictions']
    )

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[
            trainer.id2label[i]
            for i in range(len(trainer.id2label))
        ],
        yticklabels=[
            trainer.id2label[i]
            for i in range(len(trainer.id2label))
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Emotion Classification Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.close()
    
    # 5. Save model
    print("\nPHASE 4: SAVING MODEL")
    model_path = trainer.save_model('models')
    
    # 6. Final Results
    print("FINAL RESULTS ON CLEANED DATASET")
    
    if test_results:
        print(f"\nACHIEVED METRICS:")
        print(f"   • Accuracy: {test_results['test_accuracy']*100:.2f}%")
        print(f"   • F1 Score: {test_results['test_f1']*100:.2f}%")
        
        if test_results['test_accuracy'] >= 0.90:
            print(f"\nTARGET ACHIEVED!")
    
    print(f"\nModel saved: {model_path}")
    print("\nCleaned dataset saved in: cleaned_data/")
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()