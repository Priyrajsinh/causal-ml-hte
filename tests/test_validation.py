import pytest


def test_validate_accepts_clean_df(dummy_lalonde_df):
    """validate_lalonde_df must return the DataFrame unchanged for a valid input."""
    from src.data.validation import validate_lalonde_df

    result = validate_lalonde_df(dummy_lalonde_df)
    assert result.shape == dummy_lalonde_df.shape


def test_validate_rejects_invalid_treat(dummy_lalonde_df):
    """validate_lalonde_df must raise SchemaError when treat contains 2."""
    import pandera

    from src.data.validation import validate_lalonde_df

    bad_df = dummy_lalonde_df.copy()
    bad_df.loc[0, "treat"] = 2
    with pytest.raises(pandera.errors.SchemaError):
        validate_lalonde_df(bad_df)


def test_validate_rejects_negative_re74(dummy_lalonde_df):
    """validate_lalonde_df must raise SchemaError when re74 is negative."""
    import pandera

    from src.data.validation import validate_lalonde_df

    bad_df = dummy_lalonde_df.copy()
    bad_df.loc[0, "re74"] = -100.0
    with pytest.raises(pandera.errors.SchemaError):
        validate_lalonde_df(bad_df)


def test_validate_rejects_extra_column(dummy_lalonde_df):
    """validate_lalonde_df must raise SchemaErrors on an unexpected extra column."""
    import pandera

    from src.data.validation import validate_lalonde_df

    bad_df = dummy_lalonde_df.copy()
    bad_df["extra"] = 0
    # strict=True raises SchemaErrors (plural) for column-presence violations
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_lalonde_df(bad_df)
