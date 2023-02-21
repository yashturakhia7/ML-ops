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

# # set up the pipeline
# titanic_pipe = Pipeline([

#     # ===== IMPUTATION =====
#     # impute categorical variables with string 'missing'
#     ('categorical_imputation', CategoricalImputer(imputation_method='missing', variables=CATEGORICAL_VARIABLES, fill_value='missing', ignore_format=True)),

#     # add missing indicator to numerical variables
#     ('missing_indicator', AddMissingIndicator(variables=NUMERICAL_VARIABLES)),

#     # impute numerical variables with the median
#     ('median_imputation', MeanMedianImputer(variables=NUMERICAL_VARIABLES)),


#     # Extract first letter from cabin
#     ('extract_letter', ExtractLetterTransformer(variables=CABIN)),


#     # == CATEGORICAL ENCODING ======
#     # remove categories present in less than 5% of the observations (0.05)
#     # group them in one category called 'Rare'
#     ('rare_label_encoder', RareLabelEncoder(tol=0.05, n_categories=1, variables=CATEGORICAL_VARIABLES, replace_with='Rare', ignore_format=True)),


#     # encode categorical variables using one hot encoding into k-1 variables
#     ('categorical_encoder', OneHotEncoder(variables=CATEGORICAL_VARIABLES, drop_last=True)),

#     # scale using standardization
#     ('scaler', StandardScaler()),

#     # logistic regression (use C=0.0005 and random_state=0)
#     ('Logit', LogisticRegression(C=0.0005, random_state=0)),
# ])