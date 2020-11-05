# ************************************************** Import Libraries **************************************************
import numpy as np
import pandas as pd
import itertools


# ************************************************** Def check common CustomerId ***************************************
def data_quality_issues(cust_demo, cust_add, trans):

    # Number of Unique Customer ID
    print(cust_demo.customer_id.nunique())
    print(cust_add.customer_id.nunique())
    print(trans.customer_id.nunique())

    # data_quality_issues

    # A. Drop Irrelevant columns:
    cust_demo.drop(['first_name', 'last_name', 'deceased_indicator', 'default'], axis=1, inplace=True)
    # print(cust_demo.shape)
    cust_add.drop(['address', 'country', 'property_valuation'], axis=1, inplace=True)
    # print(cust_add.shape)
    trans.drop(['transaction_id', 'list_price', 'product_first_sold_date', 'online_order'], axis=1, inplace=True)
    # print(trans.shape)

    # B. Missing Records
    # Customer Demographic merged with Customer Address
    cust_demo_add = pd.merge(cust_demo, cust_add, how='inner', on='customer_id')
    print(cust_demo_add.shape)

    # Customer Demographic merged with Transactions
    cust_demo_trans = pd.merge(trans, cust_demo, how='inner', on='customer_id')
    print(cust_demo_trans.shape)

    # C. Missing Values
    print(cust_demo_add.isna().sum())
    print(cust_demo_trans.isna().sum())

    # D. Inconsistent Values
    # Gender values
    print(cust_demo_add['gender'].unique())
    cust_demo_add['gender'] = cust_demo_add['gender'].replace(['F', 'Femal'], 'Female')
    cust_demo_add['gender'] = cust_demo_add['gender'].replace(['M'], 'Male')
    print(cust_demo_add['gender'].unique())

    # age column
    cust_demo_add['age'] = cust_demo_add['DOB'].apply(lambda x: (pd.datetime.now().year - x.year))
    age_df = cust_demo_add[cust_demo_add['age'] >= 90.00]

    # state column
    print(cust_demo_add['state'].unique())
    cust_demo_add['state'] = cust_demo_add['state'].replace(['NSW'], 'New South Wales')
    cust_demo_add['state'] = cust_demo_add['state'].replace(['VIC'], 'Victoria')
    print(cust_demo_add['state'].unique())

    cust_demo_add.to_csv("OldCustCleanData.csv")


# ************************************************** Def Main **********************************************************
def main():
    cust_demo = pd.read_excel("E:\KeepLearning\KPMG\module1_dataset.xlsx", sheet_name='CustomerDemographic')
    print(cust_demo.shape)
    cust_add = pd.read_excel("E:\KeepLearning\KPMG\module1_dataset.xlsx", sheet_name='CustomerAddress')
    print(cust_add.shape)
    trans = pd.read_excel("E:\KeepLearning\KPMG\module1_dataset.xlsx", sheet_name='Transactions')
    print(trans.shape)
    data_quality_issues(cust_demo, cust_add, trans)


main()