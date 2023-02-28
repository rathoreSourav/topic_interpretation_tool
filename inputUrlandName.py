__author__      = "Sourav Singh"
__version__ = "1.0.1"

import csv
import streamlit as st

def write_to_csv(data):
    header = ["Name", "URL"]
    with open('./input/CompaniesList.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

def takeUserData():
    st.title("Prepare data using web scraping")

    #input_data = st.text_input("Enter names and URLs separated by commas:")
    lines = []
    text = st.text_area("Enter names and URLs separated by commas, one entry per line:")
    if text:
        lines = text.splitlines()

    if st.button("Submit"):
        data = []
        for item in lines:
            name, url = item.strip().split(',')
            data.append([name, url])

        print(data)    
        write_to_csv(data)
        st.session_state['listsaved'] = True
        st.success("Data saved to CSV!")
