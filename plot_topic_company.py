import pandas as pd
import matplotlib.pyplot as plt


def plot():
	# Load the data into a pandas DataFrame
	df = pd.read_csv('extracted_doc.csv')

	# Count the number of companies that have documents on each topic
	topic_counts = df.groupby('Keywords')['Name'].nunique()

	# Plot the count of companies for each topic as a bar plot
	plt.bar(topic_counts.index, topic_counts.values)
	plt.xlabel('Keywords')
	plt.ylabel('Number of Companies')
	plt.title('Number of Companies with Documents on a Topic')

	# Show the plot
	plt.show()
	#plt.save()