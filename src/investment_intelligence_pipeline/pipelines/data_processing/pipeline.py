from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_financial_news

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_financial_news,
                inputs="financial_news_raw",      # Reads from catalog
                outputs="financial_news_processed", # Writes to catalog
                name="preprocess_financial_news_node",
            ),
        ]
    )