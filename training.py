import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import zip_longest

# Load data from pickle file
with open("./data.pickle", "rb") as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Convert labels to a NumPy array
labels = np.array(labels)

# Find the maximum length of the sequences
max_length = max(len(seq) for seq in data)

# Pad the sequences to the maximum length
padded_data = []
for seq in data:
    padded_seq = seq + [0] * (max_length - len(seq))
    padded_data.append(padded_seq)

# Convert padded_data to a NumPy array
data_flat = np.array(padded_data)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data_flat, labels, test_size=0.2, shuffle=True, stratify=labels)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("Sample X_train values:", X_train[0])
print("Sample Y_train value:", Y_train[0])

# Create and fit the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Predict on the test set
Y_predict = model.predict(X_test)

# Calculate accuracy score
score = accuracy_score(Y_predict, Y_test)
print("Accuracy Score: {}".format(score*100))
f=open('model.p',"wb")
pickle.dump({"model":model},f)
f.close()
