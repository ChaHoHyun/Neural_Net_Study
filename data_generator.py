import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(9, 7))

# generate data ( -30 to 30, numbers = 100)

data_list = []

for i in range(100):
    data_x = np.random.uniform(low=-30, high=30)
    data_y = np.random.uniform(low=-30, high=30)
    data_dict = {
        "data_x": data_x,
        "data_y": data_y
    }

    data_list.append(data_dict)

# Make the data by csv file

raw_data = pd.DataFrame.from_dict(data_list)

# Set data by dividing "Group A", "Group_B" + Label

condition1 = raw_data["data_x"] > -5
condition2 = raw_data["data_y"] > -3

raw_data["hue"] = "GroupB"
raw_data.loc[condition1 & condition2, "hue"] = "GroupA"
raw_data['Label'] = 1
raw_data.loc[raw_data['hue'] == 'GroupB', 'Label'] = 0

# Visualization

plt.figure(figsize=(10, 8))
plt.title("Classification")
sns.scatterplot(data=raw_data, x="data_x", y="data_y",
                hue="hue", hue_order=["GroupA", "GroupB"], style="hue")
plt.title('DataSet')
plt.show()
plt.savefig('./dataset_R1.png', dpi=300)

# Save the file by CSV

raw_data.to_csv("dataset.csv", mode="w")
