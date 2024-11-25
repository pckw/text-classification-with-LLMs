import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle



df = pd.DataFrame()
dirname = "data/news/csv"
for file in os.listdir(dirname):
    data = pd.read_csv(os.path.join(dirname, file))
    df = pd.concat([df, data], ignore_index=True)


df = df.drop(['headlines', 'description', 'url'], axis=1)

# convert target to numeric
valid_label = df['category'].unique()
category_dict = {l: i for i, l in enumerate(valid_label)}
df["category"] = df["category"].map(category_dict)

# stratified train test split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])

# contruct output
train_output = {
    'data': train['content'],
    'target': train['category'],
    'target_names': valid_label
}

test_output = {
    'data': test['content'],
    'target': test['category'],
    'target_names': valid_label
}

# store train and test data
with open("data/news/train.pkl", "wb") as pickle_file:
    pickle.dump(dict(train_output), pickle_file)

with open("data/news/test.pkl", "wb") as pickle_file:
    pickle.dump(dict(test_output), pickle_file)