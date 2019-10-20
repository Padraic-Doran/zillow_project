
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import env
from scipy import stats
import csv

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

    
def get_data_from_mysql():
    query = '''
    Select props.`id` as property_id, 
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

    df = pd.read_sql(query, get_db_url('zillow'))
    return df    
   
    
def clean_data(df):
    df = df.dropna()
    df = df[df.bathroom_count > 0]
    df = df[df.bedroom_count > 0]
    return df


def wrangle_zillow():
    df = get_data_from_mysql()
    df = clean_data(df)
    return df


