from sklearn.utils.validation import check_is_fitted
from sklearn.impute import MissingIndicator

def get_names_out_from_ColumnTransformer(column_transformer, df=None, verbose=False):
    """
    Returns a list of the feature names produced by a Column Transformer
    It should probably do the same as the ColumnTransformer method get_feature_names_out() 
    but it didn't work for me

    Parameters
    ----------
    column_transformer: an instance of a fitted Column Transformer;
                        the second element of each tuple must be a Pipeline, not a transformer
    df: the dataframe passed to the fit method of a Column Transformer 
        (only needed if remainder step has passthrough strategy)
    verbose: bool

    Returns
    -------
    list of column names
    """
    check_is_fitted(column_transformer)
    col_names = []
    # column_transformer.transformers_ is a list of tuples
    # each tuple (outer_pipeline) has three elements: 
    # name of the pipeline -> outer_pipeline[0]
    # the fitted pipeline -> outer_pipeline[1]
    # list of features fed into the pipeline -> outer_pipeline[2]
    for outer_pipeline in [p for p in column_transformer.transformers_ if p[0] != 'remainder']:
        features_in = outer_pipeline[2]
        if verbose: print(f"features in '{outer_pipeline[0]}': {features_in}")
        for inner_pipeline_step in outer_pipeline[1].steps:
            if verbose: print(f"  features in '{inner_pipeline_step[0]}': {features_in}")
            # inner_pipeline_step is a tuple of two elements
            # name of the transformer -> inner_pipeline_step[0]
            # the fitted transformer -> inner_pipeline_step[1]
            transformer = inner_pipeline_step[1]
            if hasattr(transformer, 'get_feature_names_out'):
                features_out = transformer.get_feature_names_out(features_in).tolist()
            else:
                # if a transformer doesn't have get_feature_names_out method
                # features in = features out
                features_out = features_in
            # if an imputer has add_indicator=True make a name for it
            if hasattr(transformer, 'indicator_') \
            and transformer.indicator_ is not None:
                features_out += [features_in[i] + '_missing' for i in transformer.indicator_.features_]
            if isinstance(transformer, MissingIndicator):
                features_out = [features_in[i] + '_missing' for i in transformer.features_]    
            # features_out is features_in for the next inner_pipeline_step
            features_in = features_out
            if verbose: print(f"  features out '{inner_pipeline_step[0]}': {features_in}")
        col_names.extend(features_out)
    # add passthrough-ed columns
    if 'remainder' in column_transformer.named_transformers_.keys() \
    and column_transformer.named_transformers_['remainder'] == 'passthrough':
        assert df is not None, "df is None"
        remainder = column_transformer.transformers_[-1]
        passthrough_features = df.columns[remainder[2]].tolist()
        col_names.extend(passthrough_features)
    return col_names