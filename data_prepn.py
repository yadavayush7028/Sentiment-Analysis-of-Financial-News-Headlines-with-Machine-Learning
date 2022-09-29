import pandas as pd
import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

f_data = open('./project2_training_data.txt', 'r', encoding='utf8')
f_lbls = open('./project2_training_data_labels.txt', 'r', encoding='utf8')

lines = f_data.readlines()
lbls = f_lbls.readlines()

fields = ['text', 'sntmt']

rows = []
for (line, lbl) in zip(lines, lbls):

    row = [line]
    row.append(lbl)
    rows.append(row)

with open('combined.csv', 'w') as f:

    writer = csv.writer(f)

    writer.writerow(fields)
    writer.writerows(rows)

df = pd.read_csv('./combined.csv', encoding='latin-1')

df['sntmt'].replace('\r', '', regex=True, inplace=True)
df['text'].replace('\r', '', regex=True, inplace=True)

df['sntmt'].replace('\n', '', regex=True, inplace=True)
df['text'].replace('\n', '', regex=True, inplace=True)

df.to_csv('./combined_data.csv', index=None)

############################################################

data = pd.read_csv('combined_data.csv')

# Vocabulary Size

count = 0
file = open("./project2_training_data.txt", "r", encoding='utf8')
read_data = file.read()
words = set(read_data.split())
for word in words:
    count += 1
print('Total Unique Words:', count)

count1 = 0
file = open("./project2_training_data.txt", "r", encoding='utf8')
read_data = file.read()
words = set(read_data.split())
for word in words:
    if word in stopwords.words('english'):
        continue
    count1+=1

print('Total Unique Words without stopwords:', count1)

# Dataset Info

print("*************************************")
print(data.isnull().sum())
print("*************************************")
print(data.info())
print("*************************************")

# Sentiments Count Plot

cnts = data.sntmt.value_counts()

plt.figure(figsize=(10,10))
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count Plot for Sentiments')

cnts.plot(kind='bar', color=['#6fdc6f','#66e0ff','#ff4d4d'])