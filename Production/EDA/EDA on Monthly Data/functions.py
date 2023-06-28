from typing import Dict, Literal, Optional
import calendar
import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Format column names to SCREAMING_SNAKE_CASE
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.upper()

    # To avoid confusion in case of a csv file
    df = df.replace(",", "", regex=True)

    df = df.drop(index=[0]).reset_index(drop=True)
    df = df.fillna(0)

    df[["NPDCODE", "YEAR", "MONTH"]] = df[["NPDCODE", "YEAR", "MONTH"]].astype(int)
    df[["ON_STREAM", "OIL", "GAS", "WATER", "WI", "GI"]] = df[
        ["ON_STREAM", "OIL", "GAS", "WATER", "WI", "GI"]
    ].astype(float)

    # Convert months to month abbreviations, 1 -> Jan, 2 -> Feb, ...etc
    abbr = dict(enumerate(calendar.month_abbr))
    abbr.pop(0)
    df["MONTH"] = df["MONTH"].map(abbr)
    df["MONTH"] = pd.Categorical(
        df["MONTH"], categories=list(abbr.values()), ordered=True
    )

    return df


def annual_data(
    data: pd.DataFrame, category: Optional[Literal["production", "injection"]] = None
) -> pd.DataFrame:
    df = data.groupby("YEAR", as_index=False)[
        ["ON_STREAM", "OIL", "GAS", "WATER", "WI", "GI"]
    ].sum()

    if not category:
        return df

    category = category.strip().lower()

    if category in ["production", "prod"]:
        df["CUM_OIL"] = df["OIL"].cumsum()
        df["CUM_GAS"] = df["GAS"].cumsum()
        df["CUM_WATER"] = df["WATER"].cumsum()
        df = df.drop(columns=["ON_STREAM", "GI", "WI"])

    elif category in ["injection", "inj"]:
        df = df[["YEAR", "GI", "WI"]]

    return df


def monthly_data(
    data: pd.DataFrame,
    parameter: Optional[Literal["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]] = None,
) -> pd.DataFrame:
    df = data.groupby(["YEAR", "MONTH"], as_index=False)[
        ["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]
    ].sum()

    if not parameter:
        return df

    parameter = parameter.strip().upper().replace(" ", "_")
    df = df.pivot_table(values=parameter, index="MONTH", columns="YEAR")

    return df


def wellbores_data(
    data: pd.DataFrame,
    category: Optional[Literal["production", "injection", "hybrid"]] = None,
) -> pd.DataFrame:
    df = data.groupby("WELLBORE_NAME", as_index=False)[
        ["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]
    ].sum()

    if not category:
        return df

    category = category.strip().lower()

    if category in ["production", "prod"]:
        df = df.query("OIL > 0 or GAS > 0")
    elif category in ["injection", "inj"]:
        df = df.query("GI > 0 or WI > 0")
    elif category in ["hybrid", "hb"]:
        df = df.query("(OIL > 0 or GAS > 0) and (GI > 0 or WI > 0)")

    df = df.reset_index(drop=True)

    return df


def well_data(data: pd.DataFrame, well_name: str) -> pd.DataFrame:
    well_name = well_name.strip().upper()
    df = data.query("`WELLBORE_NAME` == @well_name").reset_index(drop=True)

    return df


def well_annual_data(
    data: pd.DataFrame,
    well_name: str,
    category: Optional[Literal["production", "injection"]] = None,
) -> pd.DataFrame:
    df = well_data(data=data, well_name=well_name)
    df = df.groupby("YEAR", as_index=False)[
        ["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]
    ].sum()

    if not category:
        return df

    category = category.strip().lower()
    if category in ["production", "prod"]:
        df["CUM_OIL"] = df["OIL"].cumsum()
        df["CUM_GAS"] = df["GAS"].cumsum()
        df["CUM_WATER"] = df["WATER"].cumsum()
        df = df.drop(columns=["ON_STREAM", "GI", "WI"])

    elif category in ["injection", "inj"]:
        df = df[["YEAR", "GI", "WI"]]

    return df


# If the argument 'parameter' is not provided, the df returned would be the same as the one returned by the function 'well_data'
def well_monthly_data(
    data: pd.DataFrame,
    well_name: str,
    parameter: Optional[Literal["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]] = None,
) -> pd.DataFrame:
    df = well_data(data=data, well_name=well_name)

    if not parameter:
        return df

    parameter = parameter.strip().upper().replace(" ", "_")
    df = df.pivot_table(values=parameter, index="MONTH", columns="YEAR", fill_value=0)

    return df


def generate_annual_dataframes(
    data: pd.DataFrame, well_name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    dataframes_collection = {}

    dataframes_collection["all"] = (
        annual_data(data) if well_name is None else well_annual_data(data, well_name)
    )
    dataframes_collection["production"] = (
        annual_data(data, "production")
        if well_name is None
        else well_annual_data(data, well_name, "production")
    )
    dataframes_collection["injection"] = (
        annual_data(data, "injection")
        if well_name is None
        else well_annual_data(data, well_name, "injection")
    )

    return dataframes_collection


def generate_monthly_dataframes(
    data: pd.DataFrame, well_name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    dataframes_collection = {}

    dataframes_collection["ON_STREAM"] = (
        monthly_data(data, "ON_STREAM")
        if well_name is None
        else well_monthly_data(data, well_name, "ON_STREAM")
    )
    dataframes_collection["OIL"] = (
        monthly_data(data, "OIL")
        if well_name is None
        else well_monthly_data(data, well_name, "OIL")
    )
    dataframes_collection["GAS"] = (
        monthly_data(data, "GAS")
        if well_name is None
        else well_monthly_data(data, well_name, "GAS")
    )
    dataframes_collection["WATER"] = (
        monthly_data(data, "WATER")
        if well_name is None
        else well_monthly_data(data, well_name, "WATER")
    )
    dataframes_collection["GI"] = (
        monthly_data(data, "GI")
        if well_name is None
        else well_monthly_data(data, well_name, "GI")
    )
    dataframes_collection["WI"] = (
        monthly_data(data, "WI")
        if well_name is None
        else well_monthly_data(data, well_name, "WI")
    )

    return dataframes_collection


def generate_wellbores_dataframes(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dataframes_collection = {}

    dataframes_collection["all"] = wellbores_data(data, category="all")
    dataframes_collection["production"] = wellbores_data(data, category="production")
    dataframes_collection["injection"] = wellbores_data(data, category="injection")

    return dataframes_collection
