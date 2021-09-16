import pandas as pd

def transformation(df_aggregated_data:pd.DataFrame, list_feature_name: list,power_transform):
    """
    Power Transformation
    Args:
        df_aggregated_data  (df):  input data after preprocessing with one row (its median measurement within the window) per cluster id
        list_feature_name   (list): list of features for segmentation
        power_transform     (model): pretrained power transformation parameters
    Returns:
        df_model_data_transformed   (df): output data after power transformation

    """
    # for modeling transformation
    df_model_data = df_aggregated_data.drop(['cluster_id'], axis=1)[list_feature_name]

    # box-cox + standardscalar()
    df_model_data_transformed = df_model_data + 1
    df_model_data_transformed = power_transform.transform(df_model_data_transformed)


    return df_model_data_transformed