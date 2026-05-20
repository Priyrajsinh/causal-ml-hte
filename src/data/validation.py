"""Pandera schema and validator for LaLonde NSW/CPS DataFrames (rule C34)."""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

LALONDE_SCHEMA = DataFrameSchema(
    {
        "treat": Column(int, Check(lambda s: s.isin([0, 1]))),
        "age": Column(int, Check(lambda s: s.between(17, 55))),
        "education": Column(int, Check(lambda s: s.between(0, 18))),
        "black": Column(int, Check(lambda s: s.isin([0, 1]))),
        "hispanic": Column(int, Check(lambda s: s.isin([0, 1]))),
        "married": Column(int, Check(lambda s: s.isin([0, 1]))),
        "nodegree": Column(int, Check(lambda s: s.isin([0, 1]))),
        "re74": Column(float, Check(lambda s: s >= 0.0)),
        "re75": Column(float, Check(lambda s: s >= 0.0)),
        "re78": Column(float, Check(lambda s: s >= 0.0)),
    },
    strict=True,
)


def validate_lalonde_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run LALONDE_SCHEMA. Raise SchemaError on first violation. Rule C34."""
    return LALONDE_SCHEMA.validate(df)
