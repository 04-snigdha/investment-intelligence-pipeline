import pandas as pd
import re

def preprocess_financial_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw financial news data and formats it for modeling.
    """
    # 1. Keep only the columns we care about
    df = df[['sentence', 'answer']].copy()
    df = df.rename(columns={'answer': 'sentiment_label'})
    
    # 2. Clean the text: lowercase and remove special characters
    def clean_text(text):
        text = str(text).lower()
        # Remove punctuation but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text) 
        return text

    df["cleaned_sentence"] = df["sentence"].apply(clean_text)
    
    # 3. Drop any empty rows
    df = df.dropna(subset=['cleaned_sentence', 'sentiment_label'])
    
    # Reorder columns for clarity
    return df[['sentiment_label', 'sentence', 'cleaned_sentence']]