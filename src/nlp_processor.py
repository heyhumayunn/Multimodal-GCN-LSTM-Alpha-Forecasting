from typing import Optional

import pandas as pd
import torch
from transformers import pipeline

class NLPProcessor:
    """
    Encapsulates financial NLP inference tasks including sentiment analysis (FinBERT)
    and zero-shot entity relationship classification (BART).
    """
    
    # Artifact Configuration
    SENTIMENT_MODEL = "ProsusAI/finbert"
    RELATION_MODEL = "valhalla/distilbart-mnli-12-3"
    
    RELATION_LABELS = [
        "Competitor", "Supplier", "Customer", 
        "Partner", "Regulatory", "Neutral"
    ]

    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize transformer pipelines.
        
        Args:
            device_id: GPU index (e.g., 0) or -1 for CPU. Auto-detected if None.
        """
        self.device = device_id if device_id is not None else (0 if torch.cuda.is_available() else -1)
        target_device = "GPU" if self.device >= 0 else "CPU"
        
        print(f"[INFO] Initializing NLP pipelines on {target_device}...")
        
        self.sentiment_pipe = pipeline(
            "text-classification",
            model=self.SENTIMENT_MODEL,
            device=self.device,
            truncation=True,
            max_length=512
        )
        
        self.relation_pipe = pipeline(
            "zero-shot-classification",
            model=self.RELATION_MODEL,
            device=self.device
        )

    def process_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies NLP inference to news headlines/summaries.
        Returns the modified DataFrame with 'Sentiment_Score' and 'Relation_Type'.
        """
        if df.empty:
            return df

        # Context aggregation
        df['AI_Text'] = df['Headline'].astype(str) + ". " + df['Summary'].astype(str)
        corpus = df['AI_Text'].tolist()

        # Inference: Sentiment
        sentiment_results = self.sentiment_pipe(corpus)
        
        df['Sentiment_Score'] = [
            r['score'] if r['label'] == 'positive' else 
            -r['score'] if r['label'] == 'negative' else 0.0 
            for r in sentiment_results
        ]

        # Inference: Relationships
        relation_results = self.relation_pipe(
            corpus, 
            candidate_labels=self.RELATION_LABELS
        )
        
        df['Relation_Type'] = [r['labels'][0] for r in relation_results]
        
        return df