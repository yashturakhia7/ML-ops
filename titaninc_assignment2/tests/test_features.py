from classification_model.config.core import config
from classification_model.processing.features import TemporalVariableTransformer
from classification_model.processing.features import ExtractLetterTransformer

from classification_model.predict import make_prediction
import math
import numpy as np

def test_temporal_variable_transformer(sample_input_data):
    # Given
    extracter = ExtractLetterTransformer(
        variables=config.model_config.cabin_var_imputation
    )
    assert sample_input_data["cabin"].iat[10] == "G6"

    # When
    subject = extracter.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[10] == "G"
