import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Sample transaction data (replace with your own dataset)
data = {
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [['A', 'B', 'C'], ['A', 'C'], ['B', 'D'], ['A', 'B', 'C', 'D'], ['B', 'D']]
}

df = pd.DataFrame(data)

# Convert the transaction data into a one-hot encoded DataFrame
oht = df['Items'].str.join('|').str.get_dummies()

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(oht, min_support=0.2, use_colnames=True)

# Generate association rules from the frequent itemsets
association_rules_df = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
# Display the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)

