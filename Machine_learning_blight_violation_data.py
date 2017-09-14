
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32

# In[50]:

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

def blight_model():
    
    # Cleaning Up the Data
    df = pd.read_csv("train.csv",encoding='ISO-8859-1')
    df1 = pd.read_csv("test.csv",encoding='ISO-8859-1')

    #Fill the na values
    df_label = df['compliance'].fillna(0)

    # Removing the Unnecessary features
    df = df.drop( ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status',
                   'agency_name', 'inspector_name', 'violation_code','country','mailing_address_str_name',
                  'city','state','violator_name','violation_street_name','violation_description', 
                   'compliance_detail','mailing_address_str_number','ticket_issued_date','hearing_date',
                   'non_us_str_code','compliance'],axis=1)
    df = df.fillna(0)
    df1 = df1.drop( ['agency_name', 'inspector_name', 'violation_code','country','mailing_address_str_name',
                   'city','state','violator_name','violation_street_name','violation_description',
                     'mailing_address_str_number','non_us_str_code','ticket_issued_date','hearing_date'],axis=1)
    df1 = df1.fillna(0)
    df = pd.get_dummies(data=df, columns=['grafitti_status', 'disposition'])
    df1 = pd.get_dummies(data=df1, columns=['grafitti_status', 'disposition'])

    #Calculating new features using the existing ones to improve the predictions of the classifier
    df['late_amount'] = df['judgment_amount']*df['late_fee']
    df1['late_amount'] = df1['judgment_amount']*df1['late_fee']
    
    #df['hearing_date'] = pd.to_datetime(df['hearing_date']).fillna(0)
    #df['ticket_issued_date'] = pd.to_datetime(df['ticket_issued_date']).fillna(0)
    #df['date_diff'] = (pd.to_datetime(df['hearing_date']).dt.date - 
    #                         pd.to_datetime(df['ticket_issued_date']).dt.date).fillna(0)

    # Converting the datatype according to the data.
    df = df.convert_objects(convert_numeric=True).fillna(0)
    df1 = df1.convert_objects(convert_numeric=True).fillna(0)
    df1.violation_zip_code = df1.violation_zip_code.astype('float').fillna(0)
    #print(df1.dtypes)

    #Splitting the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(df, df_label, random_state=0)

    #Fit the classifier and predict the values for test set
    #clf = DecisionTreeClassifier().fit(X_train, y_train)
    clf = GradientBoostingClassifier().fit(X_train, y_train)
    tree_predicted = clf.predict(X_test)
    # clf = lr.fit(X_train, y_train).decision_function(X_test)

    #Calculating the Area Under the Curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, tree_predicted)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    
    #print(tree_predicted.shape)
    #print(df1['ticket_id'].shape)
    
    df1['disposition_SET-ASIDE (PENDING JUDGMENT)'] = df['disposition_SET-ASIDE (PENDING JUDGMENT)']
    
    #print(len(df.columns),"---",len(df1.columns))
    #print(df.head())
    #print(df1.head())

    
    preds = clf.predict(df1)
    preds = pd.DataFrame(data=preds)
    preds.set_index(df1['ticket_id'],inplace=True)

    
    #preds.rename("compliance",inplace=True)
    #print(preds.head())
    #print(preds.dtypes)
    #print()
    #print('Accuracy of DT classifier on training set: {:.2f}'
    # .format(clf.score(X_train, y_train)))
    #print('Accuracy of DT classifier on test set: {:.2f}'
    # .format(clf.score(X_test, y_test)))
    #print('ROC Score on test set: {:.2f}'
    # .format(roc_auc_lr))
    #print(df.head())


    
    return preds
blight_model()


# In[ ]:



