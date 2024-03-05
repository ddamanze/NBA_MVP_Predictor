# -*- coding: utf-8 -*-
"""NBA_MVP_Predictor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G7_nE_LLlTEk-fwuTLFXzPbqgVnf_KlK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

"""Kaggle dataset: https://www.kaggle.com/datasets/robertsunderhaft/nba-player-season-statistics-with-mvp-win-share/data

Import Data
"""

data = pd.read_csv("https://raw.githubusercontent.com/ddamanze/NBA_MVP_Predictor/main/NBA__MVP_Dataset.csv")
data

data.info()

"""Check for null values"""

data.isnull().sum()

"""Fill all null values with 0"""

data = data.fillna(0)

"""Create a variable with every MVP winner to add a column of who the MVP was from every year"""

mvp = data.groupby(by = "season").max('award_share')
mvp["was_mvp"] = True

"""Merge was_mvp column into original data set. Any row not in the MVP dataset will have a value of false under "was_mvp"
"""

data = data.merge(mvp[["award_share", "was_mvp"]], on = ["season", "award_share"], how = "left")
data["was_mvp"] = data["was_mvp"].fillna(value = False)

"""Fg per game seems repetitive when we are given fg percentage and attempts. Check to make sure they have a direct correlation"""

plt.scatter(data['fg_per_g'], data['fga_per_g']* data['fg_pct'])
plt.xlabel("fg_per_g")
plt.ylabel("fga * pct")
plt.show()
np.corrcoef(data["fg_per_g"], data["fga_per_g"]* data["fg_pct"])

"""They do, drop all columns that gives us makes per game"""

data = data.drop(columns=['fg_per_g', 'fg3_per_g', 'fg2_per_g', 'ft_per_g'])

"""Create two separate datasets for players who received MVP votes and those who did not. This well help us see the cutoffs to clean up the data and include rows that meet minimal criteria"""

mvp_votes = data[data['award_share'] > 0]
no_mvp_votes = data[data['award_share'] == 0]

plt.hist(mvp_votes["gs"], bins = 20, density = True, label = "MVP votes", alpha = 0.7)
plt.hist(no_mvp_votes["gs"], bins = 20, density = True, label = "No MVP votes", alpha = 0.7)
plt.xlabel("Games Started")
plt.ylabel("Proportion")
plt.title("Games started for MVP votes vs Non MVP votes")
plt.legend()
plt.show()

games_cutoff = mvp_votes['gs'].mean() - (mvp_votes['gs'].std() * 3)
print(f"The cutoff for number of games started to get MVP votes is {games_cutoff:.0f}")

mvp_votes[mvp_votes["gs"] < 36][["season","player","pts_per_g","gs","award_share"]].reset_index(drop = True)

"""Drop any players who did not start the GS cutoff of 36"""

df = data.drop(data[data["gs"] < games_cutoff].index)

plt.hist(mvp_votes["mp_per_g"], bins = 20, density = True, label = "MVP votes", alpha = 0.7)
plt.hist(no_mvp_votes["mp_per_g"], bins = 20, density = True, label = "No MVP votes", alpha = 0.7)
plt.xlabel("Minutes Played")
plt.ylabel("Proportion")
plt.title("Minutes played for MVP votes vs Non MVP votes")
plt.legend()
plt.show()

mp_cutoff = mvp_votes["mp_per_g"].mean() - (mvp_votes["mp_per_g"].std() * 3)
print(f"The cutoff for number of minutes played to get MVP votes is {mp_cutoff:.0f}")

mvp_votes[mvp_votes["mp_per_g"] < mp_cutoff][["season","player","pts_per_g","mp_per_g","award_share"]].reset_index(drop = True)

df = df.drop(df[df["mp_per_g"] < 28].index)

plt.hist(mvp_votes["pts_per_g"], bins =20, density = True, label = "MVP votes", alpha = 0.7)
plt.hist(no_mvp_votes["pts_per_g"], bins = 20, density = True, label = "No MVP votes", alpha = 0.7)
plt.xlabel("Pts Per Game")
plt.ylabel("Proportion")
plt.title("Pts Per Game MVP votes vs No MVP votes")
plt.legend()
plt.show()
pts_cutoff = mvp_votes["pts_per_g"].mean() - (mvp_votes["pts_per_g"].std() * 3)
print(f"The cutoff for pts per game is {pts_cutoff:.1f}")

mvp_votes[mvp_votes["pts_per_g"] < 6.9][["season", "player", "pts_per_g"]]

df = df.drop(df[df["pts_per_g"] < pts_cutoff].index)

plt.hist(mvp_votes["win_loss_pct"], bins = 20, density = True, label = "MVP votes", alpha = 0.7)
plt.hist(no_mvp_votes["win_loss_pct"], bins = 20, density = True, label = "No MVP votes", alpha = 0.7)
plt.xlabel("Win Loss Pct")
plt.ylabel("Proportion")
plt.title("Win Loss Pct MVP votes vs No MVP votes")
plt.legend()
plt.show()

WL_cutoff = mvp_votes["win_loss_pct"].mean() - (mvp_votes["win_loss_pct"].std()*3)
print(f"Win Loss Cutoff is {WL_cutoff:.3f}")

mvp_votes[mvp_votes["win_loss_pct"]<0.314][["season", "player", "pts_per_g", "win_loss_pct"]]

df = df.drop(df[df["win_loss_pct"]<WL_cutoff].index)

received_votes = df["award_share"] > 0

print(received_votes.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(received_votes.value_counts(), autopct='%.2f')

columns = df.columns.tolist()
columns = [c for c in columns if c not in ["player", "pos", "team_id", "was_mvp"]]
target = "award_share"
X = df[columns]
Y = df[target] > 0

UnSmote = len(columns)

"""Resample data using SMOTE"""

from collections import Counter
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42, sampling_strategy=1)
x_res, y_res = sm.fit_resample(X,Y)

fig1, ax1 = plt.subplots()
ax1.pie(y_res.value_counts(), autopct='%.2f')

Unsmote = len(df)

x_res

smote_df = x_res.merge(df, how='left')
smote_df.shape

smote_df["is_smote"] = smote_df.index >= Unsmote
smote_df

"""## Now find the important features"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

smote_df["was_mvp"] = smote_df["was_mvp"].fillna(value = False)

smote_df.sort_values(by="was_mvp", ascending=False)

col = [c for c in columns if c not in ["was_mvp", "award_share"]]
X = smote_df[col]
y = smote_df["was_mvp"]

"""Use random forest classifier to find the most important features"""

clf = RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X, y)

plt.bar(range(X.shape[1]), clf.feature_importances_)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show

feature_importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': range(X.shape[1]), 'Importance': feature_importances})
feature_importance_df.sort_values(by=["Importance"], ascending=False).head(10)

all_columns = pd.DataFrame(smote_df.columns.tolist())
all_columns['Feature'] = all_columns.index
all_columns.head()

important_columns = feature_importance_df.merge(all_columns, how = "left")
important_columns.rename(columns={0: "Metric"}).sort_values(by="Importance", ascending=False).head(5)

"""Top 5 metrics to determine MVP are: <br> 1. Win shares per 48<br> 2. Win shares <br> 3. Box plus minus <br> 4. Offensive win shares <br> 5. Margin of victory (adjusted)"""

smote_df

columns = ['age', 'g', 'gs', 'mp_per_g', 'fga_per_g', 'fg_pct',
        'fg2a_per_g', 'fg2_pct', 'efg_pct', 'fta_per_g', 'ft_pct', 'drb_per_g',
        'ast_per_g', 'stl_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp',
        'per', 'ts_pct', 'fta_per_fga_pct', 'drb_pct', 'trb_pct', 'ast_pct',
        'stl_pct', 'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_48',
        'obpm', 'dbpm', 'bpm', 'vorp','mov', 'mov_adj','win_loss_pct']

train_info = smote_df[["season", "player", "pos", "team_id", "award_share","is_smote", "was_mvp"]]
train_data = smote_df[columns]
target = smote_df[["award_share"]]

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

seasons = smote_df["season"].unique()
szn = smote_df["season"].head(1)
results_table = pd.DataFrame(columns=["Season", "RF MAE", "XGBoost MAE"])
for season in seasons:
  print(season)
  tr = train_data[train_info["season"] != season]
  targ = target[train_info["season"] != season]
  rf_regressor = RandomForestRegressor(max_depth=7, random_state=0)
  rf_regressor.fit(tr,targ.values.ravel())
  xgb = XGBRegressor()
  xgb.fit(tr, targ.values.ravel())
  test_x = train_data[(train_info["season"] == season) & (~train_info["is_smote"])]
  test_y = target[(train_info["season"] == season) & (~train_info["is_smote"])]
  rf_y_pred = rf_regressor.predict(test_x)
  xgb_y_pred = xgb.predict(test_x)
  rf_mae = np.mean(np.absolute(rf_y_pred - test_y.to_numpy()[:,0]))
  xgb_mae = np.mean(np.absolute(xgb_y_pred - test_y.to_numpy()[:,0]))
  baseline_results = pd.DataFrame({"Season": season, "RF MAE": rf_mae, "XGBoost MAE": xgb_mae}, index=[0])
  results_table = pd.concat([results_table, baseline_results], ignore_index=True)
  print(f"Random Forest MAE: {rf_mae}")
  print(f"XGBoost MAE: {xgb_mae}")
  baseline_results.append([season, rf_mae, xgb_mae])
results_table.loc[:, "Best MAE"] = results_table.apply(lambda x: "Random Forest" if x["RF MAE"] < x["XGBoost MAE"] else "XGBoost", axis=1)
print(results_table)
print("*"*80)
print("Model with the most accuracy")
print(results_table["Best MAE"].value_counts())

"""XGBoost has better accuracy than Random Forest"""

seasons = smote_df["season"].unique()
szn = smote_df["season"].head(1)
accuracy_list = []
for season in seasons:
  print(season)
  tr = train_data[train_info["season"] != season]
  targ = target[train_info["season"] != season]
  xgb = XGBRegressor()
  xgb.fit(tr, targ.values.ravel())
  test_x = train_data[(train_info["season"] == season) & (~train_info["is_smote"])]
  test_y = target[(train_info["season"] == season) & (~train_info["is_smote"])]
  xgb_y_pred = xgb.predict(test_x)
  xgb_mae = np.mean(np.absolute(xgb_y_pred - test_y.to_numpy()[:,0]))
  top_two = train_info.iloc[np.argsort(xgb_y_pred)[-2:]]
  mvp_winner_pred = train_info[(train_info["season"] == season) & (~train_info["is_smote"])].iloc[np.argsort(xgb_y_pred)[-1:]]
  actual_mvp_winner = train_info[(train_info["season"] == season) & (~train_info["is_smote"])].sort_values("award_share", ascending=False).head(1)
  print(f"MAE: {xgb_mae}")
  pred_player = mvp_winner_pred["player"].to_string(index=False)
  actual_player = actual_mvp_winner["player"].to_string(index=False)
  predicted_share = mvp_winner_pred["award_share"].to_string(index=False)
  T_F = pred_player == actual_player
  print(f"Predicted MVP Winner: {pred_player}")
  print(f"Predicted Award Share: {predicted_share}")
  print()
  print("Actual Top Two Vote Getters: ")
  print(train_info[(train_info["season"] == season) & (~train_info["is_smote"])].sort_values("award_share", ascending=False).head(2))
  print()
  print(f"Actual MVP Winner: {actual_player}")
  print(f"MVP prediction was correct?: {T_F}")
  print("*"*80)
  accuracy_list.append(T_F)
results_series = pd.Series(accuracy_list)
true_percentage = (results_series.sum() / len(results_series)) * 100
false_percentage = 100 - true_percentage
print("Model Accuracy: ")
print(f"True percentage: {true_percentage: .2f}%")
print(f"False percentage: {false_percentage: .2f}%")

Import 2023-2024 Data from Basketball Reference
"""

import requests
from bs4 import BeautifulSoup

url = "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html"

response = requests.get(url)

player_data_list = []
if response.status_code == 200:
  # Parse the HTML content
  soup = BeautifulSoup(response.content, "html.parser")

  # Find the HTML elements containing the player stats
  player_stats_table = soup.find("table", {"id": "per_game_stats"})

  # Extract the data from the table
  for row in player_stats_table.find("tbody").find_all("tr"):
    columns = row.find_all(["th","td"])
    player_name = columns[1].text.strip()
    other_columns = [col.text.strip() for col in columns[2:30]]
    player_dict = {"Player": player_name}
    for i, col_text in enumerate(other_columns):
      player_dict[f"Column_{i+1}"] = col_text
    player_data_list.append(player_dict)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
column_names = ["player", "pos", "age", "team_id", "g", "gs","mp_per_g", "fg_per_g", "fga_per_g", "fg_pct", "fg3_per_g", "fg3a_per_g", "fg3_pct",
                "fg2_per_g", "fg2a_per_g", "fg2_pct", "efg_pct", "ft_per_g", "fta_per_g", "ft_pct", "orb_per_g", "drb_per_g", "tr_per_g", "ast_per_g", "stl_per_g", "blk_per_g", #change to trb
                "tov_per_g", "pf_per_g", "pts_per_g"]
player_stats_df = pd.DataFrame(player_data_list)
player_stats_df.columns = column_names
print(player_stats_df)

url_advanced_stats = "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html"

response = requests.get(url_advanced_stats)

player_advanced_list = []
if response.status_code == 200:
  # Parse the HTML content
  soup = BeautifulSoup(response.content, "html.parser")

  # Find the HTML elements containing the player stats
  player_stats_table = soup.find("table", {"id": "advanced_stats"})

  # Extract the data from the table
  for row in player_stats_table.find("tbody").find_all("tr"):
    columns = row.find_all(["th","td"])
    player_name = columns[1].text.strip()
    other_columns = [col.text.strip() for col in columns[6:29]]
    player_dict = {"Player": player_name}
    for i, col_text in enumerate(other_columns):
      player_dict[f"Column_{i+1}"] = col_text
    player_advanced_list.append(player_dict)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
column_names = ["player", "mp","per", "ts_pct", "fg3a_per_fga_pct", "fta_per_fga_pct", "orb_pct", "drb_pct", "trb_pct",
                "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct", "","ows", "dws", "ws", "ws_per_48", "", "obpm", "dbpm", "bpm", "vorp"]
player_advanced_df = pd.DataFrame(player_advanced_list)
player_advanced_df.columns = column_names
print(player_advanced_df)

player_advanced_df = player_advanced_df.drop(columns = ["", ""])

nba_team_stats = "https://www.basketball-reference.com/leagues/NBA_2024.html"

response = requests.get(nba_team_stats)

nba_advanced_list = []
if response.status_code == 200:
  # Parse the HTML content
  soup = BeautifulSoup(response.content, "html.parser")

  # Find the HTML elements containing the player stats
  nba_advanced_table = soup.find("table", {"id": "advanced-team"})

  # Extract the data from the table (you'll need to adapt this based on the table structure)
  for row in nba_advanced_table.find("tbody").find_all("tr"):
    columns = row.find_all(["th","td"])
    team_name = columns[1].text.strip()
    other_columns = [col.text.strip() for col in columns[3:5]+columns[7:9]]
    team_dict = {"Team": team_name}
    for i, col_text in enumerate(other_columns):
      team_dict[f"Column_{i+1}"] = col_text
    nba_advanced_list.append(team_dict)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
column_names = ["team_id", "win","loss","mov","sos"]
nba_advanced_df = pd.DataFrame(nba_advanced_list)
nba_advanced_df.columns = column_names
print(nba_advanced_df)

tm_name = nba_advanced_df["team_id"]
tm_abbrev = pd.DataFrame({'tm_abbreviation': ['BOS', 'OKC', 'LAC', 'MIN', 'NYK', 'PHI', 'CLE', 'DEN', 'MIL', 'NOP', 'PHX', 'IND', 'SAC', 'HOU', 'ORL', 'GSW', 'DAL', 'BKN', 'LAL',
                                  'MIA', 'CHI', 'UTA', 'ATL', 'TOR', 'MEM', 'POR', 'SAS', 'WSH', 'DET', 'CHA']})

nba_advanced_df["mov"] = nba_advanced_df["mov"].astype(float)
nba_advanced_df["sos"] = nba_advanced_df["sos"].astype(float)
nba_advanced_df["win"] = nba_advanced_df["win"].astype(int)
nba_advanced_df["loss"] = nba_advanced_df["loss"].astype(int)

for i in tm_name.index:
  nba_advanced_df = nba_advanced_df.replace([tm_name.loc[i]], tm_abbrev.loc[i])
nba_advanced_df["mov_adj"] = nba_advanced_df[["mov","sos"]].sum(axis=1)
nba_advanced_df["win_loss_pct"] = nba_advanced_df.apply(lambda row: row["win"]/(row["win"] + row["loss"]), axis=1)
nba_advanced_df

nba_advanced_df_columns = ["team_id", "mov", "mov_adj", "win_loss_pct"]

nba_tm_advanced_df = nba_advanced_df[nba_advanced_df_columns]
nba_tm_advanced_df

merged_df = pd.merge(player_stats_df, player_advanced_df, on='player', how='inner')

merged_df = pd.merge(merged_df, nba_tm_advanced_df, on='team_id', how='inner')
merged_df

merged_df = merged_df.drop(columns=['fg_per_g', 'fg3_per_g', 'fg2_per_g', 'ft_per_g'])

columns_ty = ['age', 'g', 'gs', 'mp_per_g', 'fga_per_g', 'fg_pct',
        'fg2a_per_g', 'fg2_pct', 'efg_pct', 'fta_per_g', 'ft_pct', 'drb_per_g',
        'ast_per_g', 'stl_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp',
        'per', 'ts_pct', 'fta_per_fga_pct', 'drb_pct', 'trb_pct', 'ast_pct',
        'stl_pct', 'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_48',
        'obpm', 'dbpm', 'bpm', 'vorp', 'mov', 'mov_adj', 'win_loss_pct']

merged_df['age'] = merged_df['age'].replace('Age', np.nan)

#Check for duplicates, some players play for multiple teams during the season via trade or free agency
merged_df.player.duplicated().sum()

merged_df = merged_df[merged_df['team_id'] != 'TOT'].drop_duplicates(subset=['player'])

merged_df

merged_df.isnull().sum()

merged_df = merged_df.replace('', np.nan)

columns_to_int = ['age', 'g', 'gs', 'mp']
columns_to_float = ['mp_per_g', 'fga_per_g', 'fg_pct', 'fg3a_per_g', 'fg3_pct',
        'fg2a_per_g', 'fg2_pct', 'efg_pct', 'fta_per_g', 'ft_pct', 'drb_per_g',
        'ast_per_g', 'stl_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g',
        'per', 'ts_pct', 'fg3a_per_fga_pct','fta_per_fga_pct', 'orb_pct','drb_pct','trb_pct','ast_pct',
        'stl_pct', 'blk_pct','tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_48',
        'obpm', 'dbpm', 'bpm', 'vorp', 'mov', 'mov_adj', 'win_loss_pct']
merged_df[columns_to_int] = merged_df[columns_to_int].astype(int)
merged_df[columns_to_float] = merged_df[columns_to_float].astype(float)

merged_df.info()

#Add games_cutoff for players ineligible for MVP
merged_df = merged_df.drop(merged_df[merged_df["gs"] < games_cutoff].index)

ty_test = merged_df[columns_ty]

test_info = ["player", "gs","pos", "age", "team_id", "pts_per_g", "tr_per_g", "ast_per_g", "award_share"]

xgb_ty_pred = xgb.predict(ty_test)
merged_df["award_share"] = xgb_ty_pred
mvp_winner_pred = merged_df.iloc[np.argsort(xgb_ty_pred)[-1:]]
top_two = merged_df.iloc[np.argsort(xgb_ty_pred)[-2:]]
pred_player = mvp_winner_pred["player"].to_string(index=False)
top_two_info = top_two[test_info].sort_values(by='award_share', ascending=False)
predicted_share = mvp_winner_pred["award_share"].to_string(index=False)
print(f"Predicted MVP Winner: {pred_player}")
print(f"Predicted Award Share: {predicted_share}")
print()
print(f"Top Two: ")
print(top_two_info)
