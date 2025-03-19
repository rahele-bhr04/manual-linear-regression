import seaborn as sns
import math
import random
import numpy as np

df = sns.load_dataset("diamonds")

indexes = df.index
train_index = random.sample(range(0, len(indexes)), math.floor(0.80 * len(indexes)))  #80 percent train
test_index = list(filter(lambda x: x not in train_index, indexes))

X_train = np.array(df.iloc[train_index].carat)
X_test = np.array(df.iloc[test_index].carat)
y_train = np.array(df.iloc[train_index].price).flatten()
y_test = np.array(df.iloc[test_index].price).flatten()

