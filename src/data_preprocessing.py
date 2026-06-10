import pandas as pd
import numpy as np
import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import emoji
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# The 28 emotion columns in GoEmotions dataset
GOEMOTIONS_28_COLUMNS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Mapping dictionary for 28 → 10 emotions
EMOTION_MAP_28_TO_10 = {
    # Positive emotions group
    'admiration': 'positive',
    'amusement': 'positive',
    'approval': 'positive',
    'caring': 'positive',
    'excitement': 'positive',
    'gratitude': 'positive',
    'joy': 'positive',
    'love': 'positive',
    'optimism': 'positive',
    'pride': 'positive',
    'relief': 'positive',
    
    # Anger group
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    
    # Fear group
    'fear': 'fear',
    'nervousness': 'fear',
    
    # Sadness group
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'grief': 'sadness',
    'remorse': 'sadness',
    
    # Disgust
    'disgust': 'disgust',
    
    # Surprise group
    'surprise': 'surprise',
    'confusion': 'surprise',
    'curiosity': 'surprise',
    
    # Neutral
    'neutral': 'neutral',
    
    # Others
    'realization': 'neutral',
    'embarrassment': 'negative_mixed',
    'desire': 'positive'
}

# Final 10 emotion classes
EMOTION_CLASSES_10 = [
    'positive', 'anger', 'fear', 'sadness', 'disgust', 
    'surprise', 'neutral', 'negative_mixed'
]

def map_emotion_to_10(emotion_28):
    """Map a single emotion from 28-class to 10-class"""
    return EMOTION_MAP_28_TO_10.get(emotion_28, 'neutral')

def get_emotion_from_row(row):
    """
    Extract the emotion label from a row of 28 emotion columns
    Each row has exactly one emotion labeled as 1, others 0
    """
    for emotion in GOEMOTIONS_28_COLUMNS:
        if emotion in row.index and row[emotion] == 1:
            return emotion
    return 'neutral'  # fallback

def get_emotion_mapping_df(df):
    """
    Apply mapping to DataFrame
    GoEmotions format: each row has 28 emotion columns with binary values
    """
    # Get the emotion for each row (the one with value 1)
    df['emotion_28'] = df.apply(get_emotion_from_row, axis=1)
    
    # Map to 10 emotions
    df['emotion_10'] = df['emotion_28'].apply(map_emotion_to_10)
    
    print(f"\nEmotion mapping complete!")
    print(f"\n28-class distribution (top 10):")
    print(df['emotion_28'].value_counts().head(10))
    print(f"\n10-class distribution:")
    print(df['emotion_10'].value_counts())
    
    return df


class EmotionDataPreprocessor:
    def __init__(self, data_paths):
        """
        Initialize preprocessor with paths to CSV files
        data_paths: list of paths to goemotions CSV files
        """
        self.data_paths = data_paths
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def load_and_combine_data(self):
        """Load and combine multiple CSV files"""
        dfs = []
        for path in self.data_paths:
            print(f"Loading {path}...")
            df = pd.read_csv(path)
            print(f"  Shape: {df.shape}")
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nRaw combined dataset shape: {combined_df.shape}")
        return combined_df
    
    def remove_unwanted_records(self, df):
        """
        Remove unwanted records from the dataset
        """
        initial_count = len(df)
        print(f"\nStep 1: Removing unwanted records...")
        
        # 1. Remove rows where 'example_very_unclear' is 1 (ambiguous examples)
        if 'example_very_unclear' in df.columns:
            unclear_count = df['example_very_unclear'].sum()
            df = df[df['example_very_unclear'] != 1]
            print(f"  • Removed {unclear_count} ambiguous/unclear examples")
        
        # 2. Remove rows where all emotion columns are 0 (no emotion)
        emotion_cols = [col for col in df.columns if col in GOEMOTIONS_28_COLUMNS]
        
        # Check if any row has all zeros (no emotion)
        zero_emotion_rows = (df[emotion_cols].sum(axis=1) == 0)
        zero_count = zero_emotion_rows.sum()
        df = df[~zero_emotion_rows]
        print(f"  • Removed {zero_count} rows with no emotion labels")
        
        # 3. Remove rows with multiple emotions (more than one 1)
        multiple_emotions = (df[emotion_cols].sum(axis=1) > 1)
        multiple_count = multiple_emotions.sum()
        df = df[~multiple_emotions]
        print(f"  • Removed {multiple_count} rows with multiple emotions")
        
        # 4. Remove very short texts (less than 3 characters after cleaning)
        df['temp_clean'] = df['text'].fillna('').astype(str).apply(
            lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()).strip()
        )
        short_texts = df[df['temp_clean'].str.len() < 3]
        short_count = len(short_texts)
        df = df[df['temp_clean'].str.len() >= 3]
        print(f"  • Removed {short_count} very short/low-content texts")
        
        # Drop temporary column
        df = df.drop('temp_clean', axis=1)
        
        removed_count = initial_count - len(df)
        print(f"Total removed: {removed_count} records")
        print(f"Cleaned dataset shape: {df.shape}")
        
        return df
    
    def clean_text_data(self, df):
        """
        Advanced text cleaning to remove noise and anomalies
        """
        print(f"\nStep 2: Cleaning text data...")
        def advanced_text_cleaner(text):
            if pd.isna(text) or not isinstance(text, str):
                return ""
            # 1. Convert to lowercase
            text = text.lower()
            # 2. Expand contractions (don't -> do not)
            try:
                text = contractions.fix(text)
            except:
                pass
            # 3. Replace emojis with text descriptions
            text = emoji.demojize(text) 
            # 4. Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
            # 5. Remove user mentions (@username)
            text = re.sub(r'@\w+', ' ', text)
            # 6. Remove special characters and digits (keep only letters and spaces)
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            # 7. Remove extra whitespace
            text = ' '.join(text.split())
            # 8. Remove repeated characters (more than 2 times)
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            # 9. Remove common noise patterns
            noise_patterns = [
                r'\b(amp|rt|via)\b',  # Common social media noise
                r'\[.*?\]',             # Square bracket content
                r'\(.*?\)',             # Parenthesis content
            ]
            for pattern in noise_patterns:
                text = re.sub(pattern, ' ', text)
            
            # 10. Final cleanup
            text = ' '.join(text.split())
            
            return text
        
        # Apply cleaning
        df['cleaned_text'] = df['text'].apply(advanced_text_cleaner)
        
        # Remove rows that became empty after cleaning
        empty_clean = df[df['cleaned_text'].str.len() == 0]
        empty_count = len(empty_clean)
        df = df[df['cleaned_text'].str.len() > 0]
        print(f"  • Removed {empty_count} rows that became empty after cleaning")
        
        # Show sample of cleaning
        print("\n Sample of text cleaning:")
        for i in range(min(3, len(df))):
            print(f"    Original: {df['text'].iloc[i][:50]}...")
            print(f"    Cleaned:  {df['cleaned_text'].iloc[i][:50]}...")
            print()
        
        return df
    
    def remove_anomalies(self, df):
        """
        Detect and remove statistical anomalies
        """
        print(f"\nStep 3: Removing anomalies...")
        
        # 1. Remove texts with extreme lengths (outliers)
        text_lengths = df['cleaned_text'].str.len()
        Q1 = text_lengths.quantile(0.25)
        Q3 = text_lengths.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = max(0, Q1 - 1.5 * IQR)
        upper_bound = Q3 + 1.5 * IQR
        
        length_outliers = df[(text_lengths < lower_bound) | (text_lengths > upper_bound)]
        outlier_count = len(length_outliers)
        
        df = df[(text_lengths >= lower_bound) & (text_lengths <= upper_bound)]
        print(f"  • Removed {outlier_count} text length outliers")
        print(f"    Text length range: {int(lower_bound)} - {int(upper_bound)} characters")
        
        # 2. Remove duplicate texts (keep first occurrence)
        duplicate_count = df.duplicated(subset=['cleaned_text']).sum()
        df = df.drop_duplicates(subset=['cleaned_text'], keep='first')
        print(f"  Removed {duplicate_count} duplicate texts")
        
        # 3. Check for and remove texts with excessive punctuation/symbols
        def count_noise(text):
            if len(text) == 0:
                return 0
            # Count non-alphabetic characters
            return len(re.findall(r'[^a-zA-Z\s]', text)) / len(text)
        
        noise_ratio = df['cleaned_text'].apply(count_noise)
        noisy_texts = df[noise_ratio > 0.3]  # More than 30% non-alphabetic
        noise_count = len(noisy_texts)
        df = df[noise_ratio <= 0.3]
        print(f"  • Removed {noise_count} texts with excessive noise (>30% special chars)")
        
        return df
    
    def final_preprocessing(self, df):
        """
        Final preprocessing steps before tokenization
        """
        print(f"\nStep 4: Final preprocessing...")
        
        def final_clean(text):
            if not isinstance(text, str) or not text:
                return ""
            
            # Tokenize
            words = text.split()
            
            # Remove stopwords
            words = [w for w in words if w not in self.stop_words]
            
            # Lemmatize
            words = [self.lemmatizer.lemmatize(w) for w in words]
            
            # Remove very short words
            words = [w for w in words if len(w) > 2]
            
            return ' '.join(words)
        
        df['final_text'] = df['cleaned_text'].apply(final_clean)
        
        # Final check for empty texts
        empty_final = df[df['final_text'].str.len() == 0]
        empty_count = len(empty_final)
        df = df[df['final_text'].str.len() > 0]
        print(f"  • Removed {empty_count} texts that became empty after final cleaning")
        
        return df
    
    def balance_dataset(self, df, target_column='emotion_10'):
        """
        Properly balance dataset so ALL classes have equal samples
        """
        print(f"\nStep 5: Balancing dataset...")
        
        from imblearn.over_sampling import RandomOverSampler
        
        # Prepare features and target
        X = df[['final_text']]
        y = df[target_column]
        
        # Check original distribution
        print("\n  Original class distribution (UNBALANCED):")
        class_counts = y.value_counts().sort_index()
        
        # Check if we have multiple classes
        if len(class_counts) < 2:
            print(f"  WARNING: Only {len(class_counts)} class found!")
            print(f"  Class: {class_counts.index[0]} with {class_counts.iloc[0]} samples")
            return df  # Return original if only one class
        
        for emotion, count in class_counts.items():
            percentage = count/len(y)*100
            print(f"    {emotion}: {count:6d} ({percentage:5.1f}%)")
        # Find the target size (samples in largest class)
        target_size = class_counts.max()
        print(f"\n  Target samples per class: {target_size}")
        # Create sampling strategy dictionary
        sampling_strategy = {}
        for emotion in class_counts.index:
            sampling_strategy[emotion] = target_size
        # Apply RandomOverSampler with explicit strategy
        print(f" Applying RandomOverSampler to balance all classes...")
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        #Create balanced dataframe
        balanced_df = pd.DataFrame({
            'final_text': X_resampled['final_text'],
            target_column: y_resampled
        })
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\n  Final balanced dataset size: {len(balanced_df):,}")
        print(f"  Balanced class distribution (ALL CLASSES EQUAL):")
        new_counts = balanced_df[target_column].value_counts().sort_index()
        for emotion, count in new_counts.items():
            percentage = count/len(balanced_df)*100
            print(f"    {emotion}: {count:6d} ({percentage:5.1f}%)") 
        # Verify all classes are balanced
        unique_counts = balanced_df[target_column].value_counts().unique()
        if len(unique_counts) == 1:
            print(f"\n  PERFECT BALANCE: All {len(new_counts)} classes have {unique_counts[0]} samples")
        else:
            print(f"\n  Warning: Classes not perfectly balanced")
            print(f"     Unique counts: {unique_counts}")
        return balanced_df
    def save_cleaned_dataset(self, df, filename='cleaned_emotion_dataset.csv'):
        """
        Save the cleaned and balanced dataset to a CSV file
        """
        print(f"\nSaving cleaned dataset to disk...")
        # Create data directory if it doesn't exist
        if not os.path.exists('cleaned_data'):
            os.makedirs('cleaned_data')
            print(f"  • Created directory: cleaned_data/")
        # Full path for saving
        filepath = os.path.join('cleaned_data', filename)
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"  • Saved to: {filepath}")
        print(f"  • File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        # Also save a sample for quick viewing
        sample_path = os.path.join('cleaned_data', 'sample_' + filename)
        df.head(100).to_csv(sample_path, index=False)
        print(f"  • Sample (100 rows) saved to: {sample_path}")
        # Save dataset statistics - FIXED: removed .tolist() on list
        stats = {
            'total_samples': len(df),
            'num_classes': df['emotion_10'].nunique(),
            'classes': sorted(df['emotion_10'].unique()),  # Already a list, no .tolist() needed
            'samples_per_class': df['emotion_10'].value_counts().to_dict(),
            'avg_text_length': float(df['final_text'].str.len().mean()),
            'min_text_length': int(df['final_text'].str.len().min()),
            'max_text_length': int(df['final_text'].str.len().max()),
            'text_length_std': float(df['final_text'].str.len().std())
        }
        stats_path = os.path.join('cleaned_data', 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  • Statistics saved to: {stats_path}")
        # Also save class distribution as CSV
        dist_path = os.path.join('cleaned_data', 'class_distribution.csv')
        df['emotion_10'].value_counts().sort_index().to_csv(dist_path)
        print(f"  • Class distribution saved to: {dist_path}")
        return filepath
    def prepare_dataset(self, save_cleaned=True):
        """
        Complete dataset preparation pipeline with all cleaning steps
        """
        print("\n" + "="*70)
        print("COMPLETE DATA PREPROCESSING PIPELINE")
        print("="*70)
        # Step 1: Load data
        df = self.load_and_combine_data()
        print(f"\nInitial dataset size: {len(df):,} samples")
        # Step 2: Remove unwanted records
        df = self.remove_unwanted_records(df)
        print(f"After removing unwanted records: {len(df):,} samples")
        # Step 3: Map emotions to 10 classes
        df = get_emotion_mapping_df(df)
        print(f"After emotion mapping: {len(df):,} samples")
        # Step 4: Clean text data
        df = self.clean_text_data(df)
        print(f"After text cleaning: {len(df):,} samples")
        # Step 5: Remove anomalies
        df = self.remove_anomalies(df)
        print(f"After removing anomalies: {len(df):,} samples")
        # Step 6: Final preprocessing
        df = self.final_preprocessing(df)
        print(f"After final preprocessing: {len(df):,} samples")
        # Step 7: Balance dataset
        balanced_df = self.balance_dataset(df, target_column='emotion_10')
        # Step 8: Save cleaned dataset (optional)
        if save_cleaned:
            self.save_cleaned_dataset(balanced_df)
        print("\n" + "="*70)
        print("DATA PREPROCESSING COMPLETE")
        print("="*70)
        print(f"\nFINAL DATASET STATISTICS:")
        print(f"  • Original size (after cleaning): {len(df):,}")
        print(f"  • Balanced size: {len(balanced_df):,}")
        print(f"  • Number of classes: {balanced_df['emotion_10'].nunique()}")
        print(f"  • Classes: {sorted(balanced_df['emotion_10'].unique())}")
        
        if balanced_df['emotion_10'].nunique() > 1:
            print(f"  • Samples per class: {balanced_df['emotion_10'].value_counts().iloc[0]:,}")
        return balanced_df
    def split_data(self, df, text_column='final_text', label_column='emotion_10', 
                   test_size=0.15, val_size=0.15):
        """
        Split data into train, validation, and test sets
        """
        print(f"\nSplitting data into train/val/test sets...")
        # First split: train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df[text_column], df[label_column], 
            test_size=test_size, 
            random_state=42, 
            stratify=df[label_column],
            shuffle=True
        )
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            random_state=42,
            stratify=y_train_val,
            shuffle=True
        )
        print(f"\nSplit complete!")
        print(f"  • Train set:      {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  • Validation set: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  • Test set:       {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)") 
        # Verify distributions
        print(f"\n  Train set distribution:")
        train_dist = pd.Series(y_train).value_counts().sort_index()
        for label, count in train_dist.items():
            print(f"    {label}: {count} ({count/len(y_train)*100:.1f}%)")
        return {
            'train': (list(X_train), list(y_train)),
            'val': (list(X_val), list(y_val)),
            'test': (list(X_test), list(y_test))
        }
    def load_cleaned_dataset(self, filename='cleaned_emotion_dataset.csv'):
        """
        Load a previously saved cleaned dataset
        """
        filepath = os.path.join('cleaned_data', filename)
        if not os.path.exists(filepath):
            print(f"Cleaned dataset not found at {filepath}")
            return None
        
        print(f"\nLoading cleaned dataset from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"  • Loaded {len(df):,} samples")
        print(f"  • Classes: {sorted(df['emotion_10'].unique())}")
        print(f"  • Class distribution:")
        for emotion, count in df['emotion_10'].value_counts().sort_index().items():
            print(f"    {emotion}: {count}")
        return df