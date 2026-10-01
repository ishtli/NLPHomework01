from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

root = Path(__file__).resolve().parent
source = root / "data" / "NYT" / "nyt-preprocessed_news_256000.csv"
output = root / "data" / "NYTprocessed"
output.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)

df["text"] = df["text"].str.strip()
df["label"] = df["topic_id"].astype(int)
df = df[["text", "label"]].reset_index(drop=True)


'''                         
                         topic_name  topic_id
0                            Sports       0.0
1  Arts, Culture, and Entertainment       1.0
2              Business and Finance       2.0
3               Health and Wellness       3.0
4             Lifestyle and Fashion       4.0
5            Science and Technology       5.0
6                          Politics       6.0
7                             Crime       7.0

topic_id: 浮点0.0-7.0，和topic_name对应，转成整型作为label新增一column
最后得到
Index(['text', 'label'], dtype='str')
'''


'''
Length: 256000, dtype: str
Length: 253450, dtype: str
有重复文本
分group确保重复文本在同一组，同时Stratified保证类别均匀
'''
splitter = StratifiedGroupKFold(
    n_splits=10, shuffle=True, random_state=42
)
df["fold_id"]=-1


for fold_idx, (_, test_idxes) in enumerate(
    splitter.split(df, y=df["label"], groups=df["text"])
):
    df.loc[test_idxes, "fold_id"] = fold_idx

print(df["fold_id"].unique())

train = df[["text","label"]].loc[df["fold_id"] >= 2]
val = df[["text","label"]].loc[df["fold_id"]  == 1]
test = df[["text","label"]].loc[df["fold_id"] == 0]

print(train)
print(test)
print(val)

assert len(train) + len(val) + len(test) == len(df)

for name, dataset in [("train", train), ("val", val), ("test", test)]:
    dataset.to_csv(output / f"{name}.csv", index=False, encoding="utf-8")
    print(name, len(dataset), dataset["label"].value_counts().sort_index().to_dict())

'''
[5 7 6 1 4 2 0 8 9 3]
                                                     text  label
0       Dana Kinker, Keegan O’Brien. Weddings and Enga...      4
1       Dress Code Snaps Back. Customs, Etiquette and ...      4
2       Virginia Johnson Steps Down From Dance Theater...      1
5       Disney Reorganization Anticipates 21st Century...      2
7       Dave Chappelle Stumbles Into the #MeToo Moment...      1
...                                                   ...    ...
255993  First Comes Sex Talk With These Renegades of C...      4
255995  Glamour Salutes Its Heroines. Fashion and Appa...      4
255996  Study Finds Prior Trauma Raised Children’s 9/1...      3
255997  Review: ‘Prevenge,’ Orchestrated by a Fiendish...      1
255999  Walmart’s E-Commerce Results and Japan’s Silve...      2

[204800 rows x 2 columns]
204800
                                                     text  label
9       Walmart Adjusts the Thermostat to Warm Worker ...      2
18      How Mary Tyler Moore Changed Television. Moore...      1
30      Biden Defends Son Hunter at Debate, Saying Foc...      6
49      Heather McGhee, Cassim Shepard. Weddings and E...      4
54      Allen Weisselberg, Top Trump Organization Offi...      6
...                                                   ...    ...
255971  Cory Booker on Gun Control. Gun Control, Booke...      6
255986  Meet an Ecologist Who Works for God (and Again...      7
255989  New Questions Over Actions of State Police in ...      7
255991  Kindle Users to Be Able to Borrow Library E-Bo...      5
255994  Fears of a ‘Twindemic’ Recede as Flu Lies Low....      3

[25600 rows x 2 columns]
25600
                                                     text  label
3       A Trailblazing Female Conductor Is Still Alone...      1
4       Police Officer Who Fatally Shot 15-Year-Old Te...      7
6       Alexandra Nemeth, Dmitriy Slavin. Weddings and...      4
12      We Went to the Grace Hopper Celebration. Here’...      5
15      Ariana Koblitz, Robert Nicolais. Weddings and ...      4
...                                                   ...    ...
255941  ‘Biophilia’ Celebrates Colorful Creatures, Ick...      5
255944  Colson Whitehead Wins National Book Award for ...      2
255979  Behind Volkswagen Settlement, Speed and Compro...      2
255980  The Best Wardrobe Investments, Courtesy of the...      4
255998  Pearson, Ex-Owner of Financial Times, to Shed ...      2

[25600 rows x 2 columns]
25600
train 204800 {0: 25600, 1: 25600, 2: 25600, 3: 25600, 4: 25600, 5: 25600, 6: 25600, 7: 25600}
val 25600 {0: 3200, 1: 3200, 2: 3200, 3: 3200, 4: 3200, 5: 3200, 6: 3200, 7: 3200}
test 25600 {0: 3200, 1: 3200, 2: 3200, 3: 3200, 4: 3200, 5: 3200, 6: 3200, 7: 3200}
'''