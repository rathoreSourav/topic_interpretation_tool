import matplotlib.pyplot as plt
import pandas as pd

# Create some sample data
data = {'Group': ['A', 'A', 'B', 'B', 'B', 'C', 'C'],
        'Company': ['X', 'Y', 'X', 'X', 'Z', 'X', 'Y']}
df = pd.DataFrame(data)

# Group the data by Group and Company
grouped = df.groupby(['Group', 'Company']).size().reset_index(name='Count')

# Plot the data for each group
for group, subset in grouped.groupby('Group'):
    ax = subset.plot(x='Company', y='Count', kind='bar')
    plt.title(f'Companies in Group {group}')
    plt.xlabel('Company')
    plt.ylabel('Count')
    plt.show()
