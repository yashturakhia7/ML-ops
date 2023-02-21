from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from classification_model.config.core import config
from classification_model.processing import features as pp

# feature scaling
from sklearn.preprocessing import StandardScaler

# for encoding categorical variables
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder

from classification_model.processing.features import ExtractLetterTransformer
from sklearn.linear_model import LogisticRegression


titanic_pipe = Pipeline(steps=[('categorical_imputation',
                 CategoricalImputer(fill_value='missing',variables=['sex', 'cabin', 'embarked',
                                               'title'])),
                ('missing_indicator',
                 AddMissingIndicator(variables=['age', 'fare'])),
                ('median_imputation',
                 MeanMedianImputer(variables=['age', 'fare'])),
                ('extract_letter',
                 ExtractLetterTransformer(variables=['cabin'])),
                ('rare_label_encoder',
                 RareLabelEncoder(n_categories=1,
                                  variables=['sex', 'cabin', 'embarked',
                                             'title'])),
                ('categorical_encoder',
                 OneHotEncoder(drop_last=True,
                               variables=['sex', 'cabin', 'embarked',
                                          'title'])),
                ('scaler', StandardScaler()),
                ('Logit', LogisticRegression(C=0.0005, random_state=0))
    ]
)
