# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# data_cleaning.py
import pandas as pd
import numpy as np


def clean_data(df):
    # Count null values in each column
    df_null = df.isnull().sum().to_dict()
    print(df_null)
    
    # Drop rows with any null value across all columns
    df_without_nulls = df.dropna()

    # Display the count of null values after dropping
    df_without_nulls_null_count = df_without_nulls.isnull().sum().to_dict()
    print(df_without_nulls_null_count)
    
    # Replace spaces with underscores and convert column names to lowercase
    df.columns = [value.lower().replace(' ', '_') for value in df.columns]

    # Convert 'yearstart' and 'yearend' to date format
    df['yearstart'] = pd.to_datetime(df['yearstart'].astype(str), format='%Y')
    df['yearend'] = pd.to_datetime(df['yearend'].astype(str), format='%Y')

    # Separate 'geolocation' into latitude and longitude columns
    df[['latitude', 'longitude']] = df['geolocation'].str.extract(r'\(([^ ]+) ([^ ]+)\)').apply(pd.to_numeric, errors='coerce')

    # Check for NULL values in latitude and longitude columns
    result = df[['latitude', 'longitude']].isnull().sum()
    print(result)

    # Convert 'datavalue' to numeric
    df['datavalue'] = pd.to_numeric(df['datavalue'], errors='coerce')

    return df



