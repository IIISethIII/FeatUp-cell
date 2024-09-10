import pandas as pd

# Load the CSV file
metadata = pd.read_csv('/ictstr01/groups/labs/marr/qscd01/workspace/rao.umer/pengmarr_hackathon_2023/datasets/metadata_kf5.csv')

# Count the number of train and test samples for fold 0
train_count = (metadata['set0'] == 'train').sum()
test_count = (metadata['set0'] == 'test').sum()

# Calculate percentages
total_samples = len(metadata)
train_percentage = (train_count / total_samples) * 100
test_percentage = (test_count / total_samples) * 100

print(f"For fold 0:")
print(f"Training samples: {train_count} ({train_percentage:.2f}%)")
print(f"Test samples: {test_count} ({test_percentage:.2f}%)")

# Optionally, check the distribution for each class
class_distribution = metadata[metadata['set0'] == 'train']['label'].value_counts()
print("\nClass distribution in training set:")
print(class_distribution)

import matplotlib.pyplot as plt

train_dist = metadata[metadata['set0'] == 'train']['label'].value_counts()
test_dist = metadata[metadata['set0'] == 'test']['label'].value_counts()

fig, ax = plt.subplots(figsize=(12, 6))
train_dist.plot(kind='bar', ax=ax, position=0, width=0.4, label='Train')
test_dist.plot(kind='bar', ax=ax, position=1, width=0.4, label='Test')

ax.set_xlabel('Classes')
ax.set_ylabel('Number of samples')
ax.set_title('Train/Test split for fold 0')
ax.legend()
plt.tight_layout()
plt.show()