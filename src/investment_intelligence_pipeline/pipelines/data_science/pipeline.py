from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                # 'params:' tells Kedro to look in parameters.yml instead of catalog.yml
                inputs=["financial_news_processed", "params:model_options"],
                # These outputs stay in memory (RAM) to be passed to the next nodes
                outputs=["train_data", "test_data"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["train_data", "params:model_options"],
                outputs="sentiment_model", # This saves to your data/06_models folder!
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["sentiment_model", "test_data"],
                outputs="model_metrics",   # This saves to your data/08_reporting folder!
                name="evaluate_model_node",
            ),
        ]
    )