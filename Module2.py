# ************************************************** Import Libraries **************************************************
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

# ************************************************** Def data_exploration **********************************************
def data_exploration(df):

    # select required columns
    df = df[['gender', 'age']]

    # replace nan values of age column with mean value
    df['age'].fillna(df['age'].mean(), inplace=True)
    df['age'] = df['age'].astype(int)

    # create age categories
    df['AgeCategory'] = np.where((df. age <= 30), '<=30',  # when... then
                            np.where((df.age > 30) & (df.age <= 45), '>30<=45',  # when... then
                                     np.where((df.age > 45) & (df.age <= 60), '>40<=60',  # when... then
                                              '>60')))  # else

    # plot the graphs:

    # Use seaborn style defaults and set the default figure size
    sns.set(rc={'figure.figsize': (5, 4)})

    # 1.
    # number of customers under each age category
    fig, rS = plt.subplots()
    num_cust_age_cat = df.groupby('AgeCategory').count().reset_index()
    num_cust_age_cat.rename(columns={"age": "NumberofCustomers"}, inplace=True)
    num_cust_age_cat['NumberofCustomers'] = (num_cust_age_cat['NumberofCustomers'] / num_cust_age_cat['NumberofCustomers'].sum()) * 100
    rS.bar(num_cust_age_cat['AgeCategory'], num_cust_age_cat['NumberofCustomers'], width=0.4)

    rS.set_ylabel("% of Customers", size=10, family='Arial')
    rS.set_xlabel("Customer Age Category", size=10, family='Arial')
    rS.set_title("Customer Age Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(num_cust_age_cat['AgeCategory'], num_cust_age_cat['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rS.legend()
    plt.show()

    # 2.
    # number of Female, Male and Unknown customers under each age category
    fig, rA = plt.subplots()
    num_cust_gen = df.groupby(['AgeCategory', 'gender']).count().reset_index()
    num_cust_gen.rename(columns={"age": "NumberofCustomers"}, inplace=True)
    num_cust_gen['NumberofCustomers'] = (num_cust_gen['NumberofCustomers'] / num_cust_gen['NumberofCustomers'].sum()) * 100
    f_cust = num_cust_gen[num_cust_gen['gender'] == 'Female']
    m_cust = num_cust_gen[num_cust_gen['gender'] == 'Male']
    rA.plot(f_cust['AgeCategory'], f_cust['NumberofCustomers'], marker='o', linestyle='-', label='Female Customers')
    rA.plot(m_cust['AgeCategory'], m_cust['NumberofCustomers'], marker='o', linestyle='-', label= 'Male Customers')
    rA.set_ylabel("% of Customers", size=10, family='Arial')
    rA.set_xlabel("Customer Age Category", size=10, family='Arial')
    rA.set_title("Gender Wise Customer Age Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(f_cust['AgeCategory'], f_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

        # zip joins x and y coordinates in pairs
    for x, y in zip(m_cust['AgeCategory'], m_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rA.legend()
    plt.show()



# ************************************************** Def model development ************************************************
def model_development(df):

    # select required columns
    df = df[['gender', 'past_3_years_bike_related_purchases']]
    df.rename(columns={"past_3_years_bike_related_purchases": "bikepurchase"}, inplace=True)

    # replace nan values of age column with mean value
    df['bikepurchase'].fillna(df['bikepurchase'].mean(), inplace=True)
    df['bikepurchase'] = df['bikepurchase'].astype(int)

    # create bike categories
    df['BikePurchaseCategory'] = np.where((df.bikepurchase <= 25), '<=25',  # when... then
                                 np.where((df.bikepurchase > 25) & (df.bikepurchase <= 50), '>25<=50',  # when... then
                                          np.where((df.bikepurchase > 50) & (df.bikepurchase <= 75), '>50<=75',  # when... then
                                                   '>75')))  # else

    # plot the graphs:

    # Use seaborn style defaults and set the default figure size
    sns.set(rc={'figure.figsize': (5, 4)})

    # 1.
    # number of customers under each bike category
    fig, rS = plt.subplots()
    num_cust_bp_cat = df.groupby('BikePurchaseCategory').count().reset_index()
    num_cust_bp_cat.rename(columns={"bikepurchase": "NumberofCustomers"}, inplace=True)
    num_cust_bp_cat['NumberofCustomers'] = (num_cust_bp_cat['NumberofCustomers'] / num_cust_bp_cat['NumberofCustomers'].sum()) * 100
    rS.bar(num_cust_bp_cat['BikePurchaseCategory'], num_cust_bp_cat['NumberofCustomers'], width=0.4)

    rS.set_ylabel("% of Customers", size=10, family='Arial')
    rS.set_xlabel("Customer Bike Purchase Category in past 3 Years", size=10, family='Arial')
    rS.set_title("% of Customers buying Bikes Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(num_cust_bp_cat['BikePurchaseCategory'], num_cust_bp_cat['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rS.legend()
    plt.show()

    # 2.
    # number of Female, Male and Unknown customers under each bike category
    fig, rA = plt.subplots()
    num_cust_gen = df.groupby(['BikePurchaseCategory', 'gender']).count().reset_index()
    num_cust_gen.rename(columns={"bikepurchase": "NumberofCustomers"}, inplace=True)
    num_cust_gen['NumberofCustomers'] = (num_cust_gen['NumberofCustomers'] / num_cust_gen[
        'NumberofCustomers'].sum()) * 100
    f_cust = num_cust_gen[num_cust_gen['gender'] == 'Female']
    m_cust = num_cust_gen[num_cust_gen['gender'] == 'Male']
    rA.plot(f_cust['BikePurchaseCategory'], f_cust['NumberofCustomers'], marker='o', linestyle='-', label='Female Customers')
    rA.plot(m_cust['BikePurchaseCategory'], m_cust['NumberofCustomers'], marker='o', linestyle='-', label='Male Customers')
    rA.set_ylabel("% of Customers", size=10, family='Arial')
    rA.set_xlabel("Customer Bike Purchase Category in past 3 Years", size=10, family='Arial')
    rA.set_title("Gender Wise % of Customers buying Bikes Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(f_cust['BikePurchaseCategory'], f_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

        # zip joins x and y coordinates in pairs
    for x, y in zip(m_cust['BikePurchaseCategory'], m_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rA.legend()
    plt.show()



# ************************************************** Def interpretation **********************************************
def interpretation(df):

    # select required columns
    df = df[['gender', 'job_industry_category', 'wealth_segment', 'owns_car', 'age', 'state']]

    # plot the graphs:

    # Use seaborn style defaults and set the default figure size
    sns.set(rc={'figure.figsize': (14, 6)})

    # 1.
    # number of customers under each job_industry_category
    fig, rS = plt.subplots()
    num_cust_age_cat = df.groupby('job_industry_category').count().reset_index()
    num_cust_age_cat.rename(columns={"age": "NumberofCustomers"}, inplace=True)
    num_cust_age_cat['NumberofCustomers'] = (num_cust_age_cat['NumberofCustomers'] / num_cust_age_cat['NumberofCustomers'].sum()) * 100
    rS.bar(num_cust_age_cat['job_industry_category'], num_cust_age_cat['NumberofCustomers'], width=0.4)

    rS.set_ylabel("% of Customers", size=10, family='Arial')
    rS.set_xlabel("Customer Job Industry Category", size=10, family='Arial')
    rS.set_title("Customer Job Industry Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(num_cust_age_cat['job_industry_category'], num_cust_age_cat['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rS.legend()
    plt.show()

    # 2.
    # number of customers under each wealth segment
    sns.set(rc={'figure.figsize': (5, 4)})
    fig, rS = plt.subplots()
    num_cust_age_cat = df.groupby('wealth_segment').count().reset_index()
    num_cust_age_cat.rename(columns={"age": "NumberofCustomers"}, inplace=True)
    num_cust_age_cat['NumberofCustomers'] = (num_cust_age_cat['NumberofCustomers'] / num_cust_age_cat[
        'NumberofCustomers'].sum()) * 100
    rS.bar(num_cust_age_cat['wealth_segment'], num_cust_age_cat['NumberofCustomers'], width=0.4)

    rS.set_ylabel("% of Customers", size=10, family='Arial')
    rS.set_xlabel("Customer Wealth Segment Category", size=10, family='Arial')
    rS.set_title("Customer Wealth Segment Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(num_cust_age_cat['wealth_segment'], num_cust_age_cat['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rS.legend()
    plt.show()

    # number of Female, Male and Unknown customers under each wealth segment category
    fig, rA = plt.subplots()
    num_cust_gen = df.groupby(['wealth_segment', 'gender']).count().reset_index()
    num_cust_gen.rename(columns={"age": "NumberofCustomers"}, inplace=True)
    num_cust_gen['NumberofCustomers'] = (num_cust_gen['NumberofCustomers'] / num_cust_gen[
        'NumberofCustomers'].sum()) * 100
    f_cust = num_cust_gen[num_cust_gen['gender'] == 'Female']
    m_cust = num_cust_gen[num_cust_gen['gender'] == 'Male']
    rA.plot(f_cust['wealth_segment'], f_cust['NumberofCustomers'], marker='o', linestyle='-',
            label='Female Customers')
    rA.plot(m_cust['wealth_segment'], m_cust['NumberofCustomers'], marker='o', linestyle='-',
            label='Male Customers')
    rA.set_ylabel("% of Customers", size=10, family='Arial')
    rA.set_xlabel("Customer Wealth Segment Category", size=10, family='Arial')
    rA.set_title("Gender Wise % of Customers Wealth Segment Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(f_cust['wealth_segment'], f_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

        # zip joins x and y coordinates in pairs
    for x, y in zip(m_cust['wealth_segment'], m_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rA.legend()
    plt.show()

    # 3.
    # number of customers under each state owning car
    sns.set(rc={'figure.figsize': (5, 4)})
    # number of Female, Male and Unknown customers under each wealth segment category
    fig, rA = plt.subplots()
    num_cust_gen = df.groupby(['state', 'owns_car']).count().reset_index()
    num_cust_gen.rename(columns={"age": "NumberofCustomers"}, inplace=True)
    num_cust_gen['NumberofCustomers'] = (num_cust_gen['NumberofCustomers'] / num_cust_gen[
        'NumberofCustomers'].sum()) * 100
    f_cust = num_cust_gen[num_cust_gen['owns_car'] == 'Yes']
    m_cust = num_cust_gen[num_cust_gen['owns_car'] == 'No']
    rA.plot(f_cust['state'], f_cust['NumberofCustomers'], marker='o', linestyle='-',
            label='Owns Car')
    rA.plot(m_cust['state'], m_cust['NumberofCustomers'], marker='o', linestyle='-',
            label='Not Owns Car')
    rA.set_ylabel("% of Customers", size=10, family='Arial')
    rA.set_xlabel("Customer State Names", size=10, family='Arial')
    rA.set_title("% of Customers owning cars State wise Distribution", size=10, family='Arial')
    # zip joins x and y coordinates in pairs
    for x, y in zip(f_cust['state'], f_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

        # zip joins x and y coordinates in pairs
    for x, y in zip(m_cust['state'], m_cust['NumberofCustomers']):
        label = "{:.1f}".format(y)
        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    rA.legend()
    plt.show()


# ************************************************** Def Main **********************************************************
def main():
    old_cust = pd.read_csv("OldCustCleanData.csv")
    print(old_cust.shape)


    new_cust = pd.read_excel("NewCustomers.xlsx")
    print(new_cust.shape)
    # crate age column
    # age column
    new_cust['age'] = new_cust['DOB'].apply(lambda x: (pd.datetime.now().year - x.year))

    data_exploration(new_cust)
    data_exploration(old_cust)

    model_development(new_cust)
    model_development(old_cust)

    interpretation(new_cust)
    interpretation(old_cust)






main()