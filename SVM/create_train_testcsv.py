import pandas as pd

from sklearn.model_selection import train_test_split

import sys
import os

misc_path = os.path.abspath('../preprocessing/misc')

if misc_path not in sys.path:
    sys.path.append(misc_path)

from split import GOA_split

# creating train.csv, test.csv
data = pd.read_csv("../preprocessing/data/sp_db.csv")

x = data.drop('goa', axis=1)
y = data["goa"]

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=1)
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train, columns=['goa'])
train_data = pd.concat([X_train_df, y_train_df], axis=1)

X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test, columns=['goa'])
test_data = pd.concat([X_test_df, y_test_df], axis=1)


# Save to CSV
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

fil_df = GOA_split("train.csv")


fil_data = pd.read_csv("filtered.csv")
