There are 2 Parts in this Project – Part-1: Document Similarity, Part-2: Topic Modelling.
Part-1: I used the Paraphrase Dataset to find out if one text is a paraphrase of another. Here I used the NLTK library’s WORDNET API to find the document similarity.
Part-2: I used the Newsgroup Dataset to perform Topic Modelling by applying the Latent 
Dirichlet Allocation (LDA) model to topic model the dataset.



# ## Project Description - Understanding and Predicting Property Maintenance Fines
# 
#This Project is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
#The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary 
#Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of 
#the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs)
#are issued by the city to individuals who allow their properties to remain in a deteriorated condition. 
#
#Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid.
#Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
#
# 
#The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. 
#This is where predictive modeling comes in. For this project, the task is to predict whether a given blight ticket will be paid 
#on time.
# 
#All data for this project has been provided through the [Detroit Open Data Portal](https://data.detroitmi.gov/). 
# 
# ___
# 
#There are two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files 
#corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. 
#The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, 
#False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. 
#Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
#Note: All tickets where the violators were found not responsible are not considered during evaluation.
#They are included in the training set as an additional source of data for visualization, and to enable unsupervised and 
#semi-supervised approaches. However, they are not included in the test set.
# 
# 
#train.csv - the training set (all tickets issued 2004-2011)
#test.csv - the test set (all tickets issued 2012-2016)
#addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#Note: misspelled addresses may be incorrectly geolocated.
# 
#
# 
#**Data fields**
# 
#train.csv & test.csv
# 
#ticket_id - unique identifier for tickets
#agency_name - Agency that issued the ticket
#inspector_name - Name of inspector that issued the ticket
#violator_name - Name of the person/organization that the ticket was issued to
#violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#ticket_issued_date - Date and time the ticket was issued
#hearing_date - Date and time the violator's hearing was scheduled
#violation_code, violation_description - Type of violation
#disposition - Judgment and judgement type
#fine_amount - Violation fine amount, excluding fees
#admin_fee - $20 fee assigned to responsible judgments
#state_fee - $10 fee assigned to responsible judgments
#late_fee - 10% fee assigned to responsible judgments
#discount_amount - discount applied, if any
#clean_up_cost - DPW clean-up or graffiti removal cost
#judgment_amount - Sum of all fines and fees
#grafitti_status - Flag for graffiti violations
#     
#train.csv only
# 
#payment_amount - Amount paid, if any
#payment_date - Date payment was made, if it was received
#payment_status - Current payment status as of Feb 1 2017
#balance_due - Fines and fees still owed
#collection_status - Flag for payments in collections
#compliance [target variable for prediction] 
#Null = Not responsible
#0 = Responsible, non-compliant
#1 = Responsible, compliant
#compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# 
#The predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# 
#Example:
# 
#ticket_id
#284932    0.531842
#285362    0.401958
#285361    0.105928
#285338    0.018572
#...
#376499    0.208567
#376500    0.818759
#369851    0.018528
#Name: compliance, dtype: float32
