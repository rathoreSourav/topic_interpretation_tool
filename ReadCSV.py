
import pandas as pd

def getCSVFile():

    print("CSV should contain two columns only")
    print("Company Name and URLs")
    #inp = input('Please paste the csv filename with full path')
    print("Reading and storing data input/CompaniesList.csv file")
    data = pd.read_csv('./input/CompaniesList.csv')
    return data