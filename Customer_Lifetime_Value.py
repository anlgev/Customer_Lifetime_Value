##############################################################
# Project: Customer Life Time Value Prediction with BG-NBD ve Gamma-Gamma Model
##############################################################
# Data Set Information:
# This Online Retail II data set contains all the transactions occurring for a UK-based and registered,
# non-store online retail between 01/12/2009 and 09/12/2011. The company mainly sells unique all-occasion gift-ware.
# Many customers of the company are wholesalers.

# Invoice: Invoice number. if the invoice values contain 'C' it means return
# StockCode: Product (item) code.
# Description: Product (item) name.
# Quantity: The quantities of each product (item) per Invoice.
# InvoiceDate: Invoice date and time.
# UnitPrice: Unit price.
# CustomerID: Customer number.
# Country: Country name.
############################################


##############################################################
# 1. Data and Data Preperation
##############################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_frequency_recency_matrix

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("dataset/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.shape
df.head()
df.info()
df.describe().T

# Check Outlier
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df.isnull().sum()

# Delete missing values
df.dropna(inplace=True)

# Delete return invoice
df = df[~df["Invoice"].str.startswith("C", na=False)]

df = df[df["Quantity"] > 0]


# Calculate Total Price
df["TotalPrice"] = df["Quantity"] * df["Price"]



#############################################
# Creating RFM Metrics
#############################################

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)


rfm = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (today_date - date.max()).days,
                                                     lambda date: (today_date - date.min()).days],
                                     'Invoice': lambda num: len(num),
                                     'TotalPrice': lambda price: price.sum()})


rfm.columns = rfm.columns.droplevel(0)
rfm.columns = ['Recency', "Tenure", 'Frequency', "Monetary"]
rfm.head()


# Create Monetary avg

temp_df = df.groupby(["Customer ID", "Invoice"]).agg({"Invoice": "count", "TotalPrice": ["mean"]})
temp_df.head()
temp_df = temp_df.reset_index()

temp_df.columns = temp_df.columns.droplevel(0)
temp_df.columns = ["Customer ID", "Invoice", "total_price_count", "total_price_mean"]

temp_df2 = temp_df.groupby(["Customer ID"], as_index=False).agg({"total_price_mean": ["mean"]})
temp_df2.head()

temp_df2.columns = temp_df2.columns.droplevel(0)
temp_df2.columns = ["Customer ID", "monetary_avg"]

# Add monetary avg column to rfm dataframe
rfm.index.isin(temp_df2["Customer ID"]).all()   # check two dataframe Customer ID equal
rfm = rfm.merge(temp_df2, how="left", on="Customer ID")
rfm.set_index("Customer ID", inplace=True)
rfm.head()

rfm.index = rfm.index.astype(int)

# convert daily values to weekly for recency and tenure

rfm["Recency_weekly"] = rfm["Recency"] / 7
rfm["T_weekly"] = rfm["Tenure"] / 7

rfm = rfm[(rfm['monetary_avg'] > 0)]
rfm_cltv = rfm.copy()
rfm_cltv.head()

# Assumptions of BG / NBD and gamma gamma model:
rfm_cltv[['monetary_avg', 'Recency_weekly']].corr()
rfm_cltv["Frequency"] = rfm_cltv["Frequency"].astype(int)

##############################################################
# 2. BG/NBD Model
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(rfm_cltv['Frequency'],
        rfm_cltv['Recency_weekly'],
        rfm_cltv['T_weekly'])


# Add 1 week values
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        rfm_cltv['Frequency'],
                                                        rfm_cltv['Recency_weekly'],
                                                        rfm_cltv['T_weekly']).sort_values(ascending=False).head(10)


rfm_cltv["expected_number_of_purchases"] = bgf.predict(1,
                                                       rfm_cltv['Frequency'],
                                                       rfm_cltv['Recency_weekly'],
                                                       rfm_cltv['T_weekly'])

rfm_cltv.head()



# Add 1 month values

bgf.predict(4,
            rfm_cltv['Frequency'],
            rfm_cltv['Recency_weekly'],
            rfm_cltv['T_weekly']).sort_values(ascending=False).head(10)

rfm_cltv["expected_number_of_purchases"] = bgf.predict(4,
                                                       rfm_cltv['Frequency'],
                                                       rfm_cltv['Recency_weekly'],
                                                       rfm_cltv['T_weekly'])


# Looking 1 month total earnings

bgf.predict(4,
            rfm_cltv['Frequency'],
            rfm_cltv['Recency_weekly'],
            rfm_cltv['T_weekly']).sum()



################################################################
# Evaluation of prediction Results
################################################################

plot_period_transactions(bgf)
plt.show()

#######################################
# Visualizing our Frequency/Recency Matrix
#######################################

# Consider: a customer bought from you every day for three weeks straight, and we haven’t heard from them in months.
# What are the chances they are still “alive”? Pretty small. On the other hand, a customer who historically buys from
# you once a quarter, and bought last quarter, is likely still alive. We can visualize this relationship using the
# Frequency/Recency matrix, which computes the expected number of transactions an artificial customer is to make in
# the next time period, given his or her recency (age at last purchase) and frequency (the number of repeat
# transactions he or she has made).

plot_frequency_recency_matrix(bgf)
plt.show()

# We can see that if a customer has bought 25 times from you, and their latest purchase was when they were 35 weeks old
# (given the individual is 35 weeks old), then they are your best customer (bottom-right). Your coldest customers are
# those that are in the top-right corner: they bought a lot quickly, and we haven’t seen them in weeks.
# There’s also that beautiful “tail” around (5,25). That represents the customer who buys infrequently, but we’ve seen
# him or her recently, so they might buy again - we’re not sure if they are dead or just between purchases.
# Another interesting matrix to look at is the probability of still being alive:

plot_probability_alive_matrix(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Model
##############################################################


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm_cltv['Frequency'], rfm_cltv['monetary_avg'])

ggf.conditional_expected_average_profit(rfm_cltv['Frequency'],
                                        rfm_cltv['monetary_avg']).head(10)

ggf.conditional_expected_average_profit(rfm_cltv['Frequency'],
                                        rfm_cltv['monetary_avg']).sort_values(ascending=False).head(10)

rfm_cltv["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm_cltv['Frequency'],
                                                                              rfm_cltv['monetary_avg'])

rfm_cltv.sort_values("expected_average_profit", ascending=False).head(20)


# Prediction CLTV for 3 month

cltv = ggf.customer_lifetime_value(bgf,
                                   rfm_cltv['Frequency'],
                                   rfm_cltv['Recency_weekly'],
                                   rfm_cltv['T_weekly'],
                                   rfm_cltv['monetary_avg'],
                                   time=3,
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()
rfm_cltv_final = rfm_cltv.merge(cltv, on="Customer ID", how="left")
rfm_cltv_final.head()

rfm_cltv_final.sort_values('clv', ascending=False).head(10)
