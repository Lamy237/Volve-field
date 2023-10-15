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


def get_annual_data(
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


def get_monthly_data(
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


def get_well_data(data: pd.DataFrame, well_name: str) -> pd.DataFrame:
    well_name = well_name.strip().upper()
    df = data.query("`WELLBORE_NAME` == @well_name").reset_index(drop=True)

    return df


def get_well_annual_data(
    data: pd.DataFrame,
    well_name: str,
    category: Optional[Literal["production", "injection"]] = None,
) -> pd.DataFrame:
    df = get_well_data(data=data, well_name=well_name)
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
def get_well_monthly_data(
    data: pd.DataFrame,
    well_name: str,
    parameter: Optional[Literal["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]] = None,
) -> pd.DataFrame:
    df = get_well_data(data=data, well_name=well_name)

    if not parameter:
        return df

    parameter = parameter.strip().upper().replace(" ", "_")
    df = df.pivot_table(values=parameter, index="MONTH", columns="YEAR", fill_value=0)

    return df


def determine_well_type(data: pd.DataFrame, well_name: str) -> str:
    well_df = get_well_data(data, well_name)

    if sum(well_df["OIL"] + well_df["GAS"]) > 0:
        if sum(well_df["GI"] + well_df["WI"]) > 0:
            return "HYBRID"
        else:
            return "PRODUCTION"
    else:
        return "INJECTION"


def wellbores_details(data: pd.DataFrame) -> pd.DataFrame:
    details_dict = {
        "WELLBORE_NAME": [],
        "WELLBORE_TYPE": [],
        "FIRST_RECORD": [],
        "LAST_RECORD": [],
    }

    for wellbore in data["WELLBORE_NAME"].unique():
        well_df = get_well_data(data, wellbore)
        details_dict["WELLBORE_NAME"].append(wellbore)
        details_dict["WELLBORE_TYPE"].append(determine_well_type(data, wellbore))
        details_dict["FIRST_RECORD"].append(well_df["YEAR"].min())
        details_dict["LAST_RECORD"].append(well_df["YEAR"].max())

    return pd.DataFrame(details_dict).sort_values("FIRST_RECORD").reset_index(drop=True)


def annual_data(
    data: pd.DataFrame,
    well_name: Optional[str] = None,
    category: Optional[Literal["production", "injection"]] = None,
) -> pd.DataFrame:
    if well_name:
        return get_well_annual_data(data, well_name, category)
    else:
        return get_annual_data(data, category)


def monthly_data(
    data: pd.DataFrame,
    well_name: Optional[str] = None,
    parameter: Optional[Literal["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]] = None,
) -> pd.DataFrame:
    if well_name:
        return get_well_monthly_data(data, well_name, parameter)
    else:
        return get_monthly_data(data, parameter)


def generate_annual_dataframes(
    data: pd.DataFrame, well_name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    dataframes_collection = {}

    dataframes_collection["PRODUCTION"] = annual_data(
        data, well_name, category="production"
    )
    dataframes_collection["INJECTION"] = annual_data(
        data, well_name, category="injection"
    )

    return dataframes_collection


def generate_monthly_dataframes(
    data: pd.DataFrame, well_name: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    dataframes_collection = {}
    parameters = ["ON_STREAM", "OIL", "GAS", "WATER", "GI", "WI"]

    for parameter in parameters:
        dataframes_collection[parameter] = monthly_data(data, well_name, parameter)

    return dataframes_collection


def generate_wellbores_dataframes(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dataframes_collection = {}
    categories = ["PRODUCTION", "INJECTION", "HYBRID"]

    for category in categories:
        dataframes_collection[category] = wellbores_data(data, category)

    return dataframes_collection
