# import library
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urlparse
from StoreToCSV import saveToCSV
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
import csv
import streamlit as st

from ReadCSV import *

HTTPS= "https://"
COUNT = 0
header = ["name", "url", "content"]

company_list=[]

# scraped_data_dict = {
#   "company": "",
#   "url": "",
#   "data": ""
# }
def increment():
    global COUNT
    COUNT = COUNT+1
    print(COUNT)

scraped_data_list=[]

def write_datas():
    with open('./input/scraped_data.csv', 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            #writer.writerow(row)

def write_data(data):
    with open('./input/scraped_data.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
            for row in data:
                writer.writerow(row)   

def rem_if_not_in_list():            
    df = pd.read_csv('./input/scraped_data.csv')
    list(set(company_list))
    # Drop the rows where col2 is not in the accepted_values list
    df = df[df['name'].isin(company_list)]
    df.to_csv('./input/scraped_data.csv')


def is_valid_url(url):
    val = URLValidator()
    try:
        val(url)
        #print(url)
        return True
    except:
        #print(url)
        return False

def is_http_or_https(url):
    url=url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    else:
        #st.write("not a http url: ", url)
        return HTTPS+url

    

#Scrape these linked urls
#output should have company's name, url, data
def web_scrap(link):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0'}
        link=is_http_or_https(link)
        reqs=requests.get(link, headers=headers, timeout=30)
        
        #if reqs.status_code == 200:
        contents=reqs.text
        soups = BeautifulSoup(contents, 'html.parser')
        p_tags = soups.find_all('p')
        p_texts = [p.text for p in p_tags]
        data = ' '.join(str(x) for x in p_texts)

        #data = ' '.join(s.replace("\n", " ") for s in soups.stripped_strings)
        data = re.sub(r'[\n\t\r]+', ' ', data)
        data = re.sub(r'[\\]+', ' ', data)
        #data = data.replace(',', ' ')
            #data = data.replace(',', ' ')
            #data = data.replace(';', ' ')
            #data = data.replace(',', ' ')
            #data = data.replace('"', "")
            #data = data.replace("\\" ' ')
            

            #data = '\n'.join(s.strip() for s in soups.stripped_strings)


        st.write('scraping: ', link, ' , total word counts: ', len(data))
        return data
        # else:
        #     st.write("url: ", link)
        #     return ""    

            #Extract text from p tags
            
            # p_tags = soups.find_all('p')
            # p_texts = [p.text for p in p_tags]

            # # Extract text from h1 tags
            # h1_tags = soups.find_all('h1')
            # h1_texts = [h1.text for h1 in h1_tags]

            # # Extract text from span tags
            # span_tags = soups.find_all('span')
            # span_texts = [span.text for span in span_tags]
            # space = ' '

            # p_texts = ''.join(str(x) for x in p_texts)
            # h1_texts = ''.join(str(x) for x in h1_texts)
            # span_texts = ''.join(str(x) for x in span_texts)

            # # Combine the extracted texts
            # text = p_texts+ space + h1_texts + space+ span_texts
            # return re.sub(r'[\ \n]{2,}', '', str(text))
    except:
        st.write("invalid url: ", link)

# Request to website and download HTML contents
def start_scrape(company_name, url):
    company_list.append(company_name)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0'}
    url=is_http_or_https(url)
    #print('url is:*****************,', url)    
    req=requests.get(url, headers= headers)
    content=req.text
    soup = BeautifulSoup(content, 'html.parser')
    links = {url}
    if "www.g2.com" in url:
        st.write('its a review site')
        links.add(url)
    else:    
        for link in soup.findAll('a'):
            currURL = link.get('href')
            #print("link is:", link.get('href'))
            #print('url is: ', url)
            if(is_valid_url(currURL)):
                links.add(currURL)

    soup = re.sub(r'[\ \n]{2,}', '', soup.get_text())
    thisdict = dict(name = company_name, url = url, content = soup)
    scraped_data_list.append(thisdict)
    #print(links)
    #try:

    for link in links:
        data=[]
        #st.write('link: ',link)
        raw_data = web_scrap(link)
        #st.write(raw_data)
        filter = ''.join([chr(i) for i in range(1, 32)])
        try:
            raw_data.translate(str.maketrans('', '', filter))
        except:
            print("excetipn ")    
        #st.write(raw_data)
        data.append([company_name, link, raw_data])
        write_data(data)
        thisdict = dict(name = company_name, url = link, content = raw_data)
        scraped_data_list.append(thisdict)
        increment()


def do_scrap():
    df = getCSVFile()
    #print(len(df))
    write_datas()
    max_value = len(df)
    # Create the progress bar
    progress_bar = st.progress(0)
    for index, row in df.iterrows():
        #print(row)
        try:
            start_scrape(company_name=row['Name'], url=row['URL'])
        except Exception as e:
            # code to handle all types of exceptions
            print("Error:", e)
        progress_bar.progress((index + 1)/max_value)    
         
    rem_if_not_in_list()