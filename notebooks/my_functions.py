import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import OrdinalEncoder

def data_preprocessing(df_raw:pd.DataFrame):
    df = df_raw.copy()

    # Remove "city_" prefix 
    df['city'] = df['city'].str.split('_').str[1].astype('float')

    df['major_discipline'] = ['No Major' 
                              if (pd.isnull(i) and (df['education_level'][index]=='Primary School' or df['education_level'][index]=='High School')) 
                              else i 
                              for index, i in enumerate(df['major_discipline'])]

    df['enrolled_university'] = ['no_enrollment' 
                                 if (pd.isnull(i) and df['major_discipline'][index]=='No Major') 
                                 else i 
                                 for index, i in enumerate(df['enrolled_university'])]
    
    df['company_type'] = ['None' if (pd.isnull(i) and df['last_new_job'][index]=='never') else i for index, i in enumerate(df['company_type'])]
    df['company_type'] = df['company_type'].fillna('Not given')
    
    df['company_size'] = ['None' if (pd.isnull(i) and df['last_new_job'][index]=='never') else i for index, i in enumerate(df['company_size'])]
    
    df['gender'] = df['gender'].fillna('Not given')
    
    df['education_level'] = OrdinalEncoder(
        categories = [['Primary School', 'High School', 'Graduate', 'Masters', 'Phd']],
        handle_unknown = 'use_encoded_value',
        unknown_value = np.nan,
    ).fit_transform(df[['education_level']])
    
    df['experience'] = OrdinalEncoder(
        categories = [['<1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','>20']],
        handle_unknown = 'use_encoded_value',
        unknown_value = np.nan,
    ).fit_transform(df[['experience']])

    df['company_size'] = OrdinalEncoder(
        categories = [['None', '<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+']],
        handle_unknown = 'use_encoded_value',
        unknown_value = -1,
       ).fit_transform(df[['company_size']])    
    
    df['last_new_job'] = OrdinalEncoder(
        categories = [['never','1','2','3','4','>4']],
        handle_unknown = 'use_encoded_value',
        unknown_value = np.nan,
    ).fit_transform(df[['last_new_job']])    

    # Switch to boolean variables
    df['relevent_experience'] = df['relevent_experience'] == 'Has relevent experience'

    df['target_label'] = df['target'].map({0:'Not looking for job change', 1:'Looking for a job change'})
    
    return df 

def percentage_of_quitters(data:pd.DataFrame, category:str):
    results = {}
    for type in data[category].unique():
        left_the_job = data.loc[data[category] == type]["target"]
        results.update({type : sum(left_the_job)/len(left_the_job)*100})
    return results

def proportion_z_test(data:pd.DataFrame, category:str, label:str, target:str):
    left_the_job = data.loc[data[category] == label]["target"]
    left_the_job_tot = data["target"] 
    
    p = sum(left_the_job)/len(left_the_job)
    p0 = sum(left_the_job_tot)/len(left_the_job_tot)
        
    return (p-p0)/math.sqrt(p0*(1-p0)/len(left_the_job))

def print_percentage_and_ztest(data:pd.DataFrame, category:str, labels = None):
    results = {}
    percentages = percentage_of_quitters(data, category)
    for key in percentages:
        z_score = proportion_z_test(data, category, key,'target')
        if labels is None:
            results.update({key : (z_score, percentages[key])})
        else:
            results.update({labels[int(key)] : (z_score, percentages[key])})
    results = {i[0]: i[1] for i in sorted(results.items(), key=lambda x: x[1], reverse = True)}
    for element in results.items():
        print("Percentage of "+element[0]+f" quitting Pear Inc: {round(element[1][1],2)}%; z_score = {round(element[1][0],2)}")            
    return results