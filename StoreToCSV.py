import csv

def saveToCSV(data):

    keys = data[0].keys()

    with open('./output/scraped_datasss.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)