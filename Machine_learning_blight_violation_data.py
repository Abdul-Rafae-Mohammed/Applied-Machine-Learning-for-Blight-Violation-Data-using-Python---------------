
# coding: utf-8


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




