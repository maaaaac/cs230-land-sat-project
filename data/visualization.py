import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load your CSV file
df = pd.read_csv('mass_manhole_conv_coord.csv', low_memory=False)

# Count occurrences of each unique value in the specified column
value_counts = df['ManholeType'].value_counts(dropna=False)

# Reset index to convert Series to DataFrame
value_counts_df = value_counts.reset_index()
value_counts_df.columns = ['Value', 'Count']  # Rename the columns

# Create a bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
sns.barplot(x='Value', y='Count', data=value_counts_df)

# Add titles and labels
plt.title('Count of Unique Values in your_column_name')
plt.xlabel('Unique Values')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to make room for labels
# Print the counts of unique values
print("Counts of Unique Values:")
print(value_counts_df)
# Show the plot
plt.savefig('unique_value_counts.png', format='png')  # You can specify a different filename if needed

