import pandas as pd






def remove_none_column():
	df = pd.read_csv('./input/scraped_data.csv')
	df['content'] = df['content'].str.strip()
	df['content'] = df['content'].str.strip()
	df.dropna(inplace=False)
	df.to_csv('./input/scraped_data_droped.csv')
	print(df['name'])



remove_none_column()	
