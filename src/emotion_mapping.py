"""
Emotion mapping from 28 GoEmotions classes to 10 primary emotions
Based on psychological emotion taxonomy
"""

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
        if row[emotion] == 1:
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
    
    print(f"Emotion mapping complete!")
    print(f"28-class distribution:")
    print(df['emotion_28'].value_counts().head(10))
    print(f"\n10-class distribution:")
    print(df['emotion_10'].value_counts())
    
    return df