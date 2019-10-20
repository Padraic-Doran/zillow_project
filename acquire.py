
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import env
from scipy import stats
import csv
from os import path


def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

    
def write_csv_of_data():
    df = get_zillow_data()
    df.to_csv("./zillow.csv")

def read_csv_of_data():
    return 

def get_zillow_data_from_mysql():
   
    # Use a double "%" in order to escape %'s default string formatting behavior.
    query = '''Select props.`id` as property_id, 
    props.`bathroomcnt` as bathroom_count,
    props.`bedroomcnt` as bedroom_count,
    props.`calculatedfinishedsquarefeet` as calc_finish_sq_ft,
    props.`taxvaluedollarcnt` as assessed_property_value,
    props.`taxamount` as tax_paid,
    (props.`taxamount` / props.`taxvaluedollarcnt`) * 100 as tax_rate,
    props.`fips` as county_code,
    svi.`COUNTY` as county
    from zillow.properties_2017 as props
        JOIN zillow.predictions_2017 as pred
        on zillow.props.id = zillow.pred.id
    JOIN svi_db.`svi2016_us_county` as svi
    on zillow.props.fips = svi_db.svi.fips
    Where transactiondate like "2017-05%%" or transactiondate like "2017-06%%" 
    and props.`propertylandusetypeid` not in (31, 246, 247, 248)
    AND
    props.`calculatedfinishedsquarefeet` IS NOT NULL
    AND 
    props.`bathroomcnt` != 0
    AND
    props.`bedroomcnt` != 0

    '''

    url = get_url("zillow") 
    df = pd.read_sql(query, url)
    return df

def get_zillow_data():
    """
        reads from .csv or issues slq query, writes that sql as a .csv, and returns the data.
    """
    filename = "./zillow.csv"
    if path.exists(filename):
        print(f'Reading data from {filename}')
    else:
        print(f'Reading data from query, writing to {filename}, and returning the dataframe')
        write_csv_of_data()

    # Return the dataframe read from the csv
    return pd.read_csv(filename)

def clean_data(df):
    df = df.dropna()
    df = df[df.bathroom_count > 0]
    df = df[df.bedroom_count > 0]
    return df


def wrangle_zillow():
    df = get_zillow_data()
    df = clean_data(df)
    return df

