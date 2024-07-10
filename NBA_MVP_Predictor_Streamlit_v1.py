#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
sns.set()


# Kaggle dataset: https://www.kaggle.com/datasets/robertsunderhaft/nba-player-season-statistics-with-mvp-win-share/data

# Import Data

# In[2]:


data = pd.read_csv("https://raw.githubusercontent.com/ddamanze/NBA_MVP_Predictor/main/NBA__MVP_Dataset.csv")
data['season'] = data['season'].apply(lambda x: f"{x:.0f}")


# In[3]:
st.title("Data Overview - Player Stats 1982-2022")
#data['season'] = data['season'].apply(lambda x: '{:.0f}'.format(x))
selected_year = st.selectbox('Select a season', options = data['season'].unique())
filtered_data = data[data['season'] == selected_year]
filtered_data = filtered_data.sort_values(by=["award_share"], ascending=False)
st.write(f'Data for the year {selected_year}:')
st.write(filtered_data)
st.write(f'{selected_year} Summary Statistics:')
st.write(filtered_data.describe())

# In[4]:
with st.expander("Player Search"):
    selected_player = st.text_input('Type a player to see their career stats')
    if selected_player:
        filtered_player = data[data['player'].str.contains(selected_player, case=False, na=False)]
        if not filtered_player.empty:
            st.write(filtered_player.groupby(by=["player"]))
        else:
            st.write("No player found with the name:", selected_player)
    else:
        st.write(data)


# In[5]:


data.info()


# Check for null values

# In[6]:


data.isnull().sum()


# Fill all null values with 0

# In[7]:


data = data.fillna(0)


# Create a variable with every MVP winner to add a column of who the MVP was from every year

# In[8]:


mvp = data.groupby(by = "season").max('award_share')
mvp["was_mvp"] = True


# Merge was_mvp column into original data set. Any row not in the MVP dataset will have a value of false under "was_mvp"

# In[9]:


data = data.merge(mvp[["award_share", "was_mvp"]], on = ["season", "award_share"], how = "left")
data["was_mvp"] = data["was_mvp"].fillna(value = False)


# Fg per game seems repetitive when we are given fg percentage and attempts. Check to make sure they have a direct correlation

# In[10]:


#plt.scatter(data['fg_per_g'], data['fga_per_g']* data['fg_pct'])
#plt.xlabel("fg_per_g")
#plt.ylabel("fga * pct")
#plt.show()
np.corrcoef(data["fg_per_g"], data["fga_per_g"]* data["fg_pct"])


# They do, drop all columns that gives us makes per game

# In[11]:


data = data.drop(columns=['fg_per_g', 'fg3_per_g', 'fg2_per_g', 'ft_per_g'])


# In[12]:


data_num = data.select_dtypes(exclude = 'object')


# In[13]:


fig5, ax = plt.subplots(figsize=(15,10))
sns.heatmap(data_num.corr())


# Create two separate datasets for players who received MVP votes and those who did not. This well help us see the cutoffs to clean up the data and include rows that meet minimal criteria

# In[14]:


mvp_votes = data[data['award_share'] > 0]
no_mvp_votes = data[data['award_share'] == 0]


# In[15]:


#plt.hist(mvp_votes["gs"], bins = 20, density = True, label = "MVP votes", alpha = 0.7)
#plt.hist(no_mvp_votes["gs"], bins = 20, density = True, label = "No MVP votes", alpha = 0.7)
#plt.xlabel("Games Started")
#plt.ylabel("Proportion")
#plt.title("Games started for MVP votes vs Non MVP votes")
#plt.legend()
#plt.show()


# In[16]:


games_cutoff = mvp_votes['gs'].mean() - (mvp_votes['gs'].std() * 3)
print(f"The cutoff for number of games started to get MVP votes is {games_cutoff:.0f}")


# In[17]:


#mvp_votes[mvp_votes["gs"] < 36][["season","player","pts_per_g","gs","award_share"]].reset_index(drop = True)


# Drop any players who did not start the GS cutoff of 36

# In[18]:


df = data.drop(data[data["gs"] < games_cutoff].index)


# In[19]:


#plt.hist(mvp_votes["mp_per_g"], bins = 20, density = True, label = "MVP votes", alpha = 0.7)
#plt.hist(no_mvp_votes["mp_per_g"], bins = 20, density = True, label = "No MVP votes", alpha = 0.7)
#plt.xlabel("Minutes Played")
#plt.ylabel("Proportion")
#plt.title("Minutes played for MVP votes vs Non MVP votes")
#plt.legend()
#plt.show()

mp_cutoff = mvp_votes["mp_per_g"].mean() - (mvp_votes["mp_per_g"].std() * 3)
print(f"The cutoff for number of minutes played to get MVP votes is {mp_cutoff:.0f}")


# In[20]:


#mvp_votes[mvp_votes["mp_per_g"] < mp_cutoff][["season","player","pts_per_g","mp_per_g","award_share"]].reset_index(drop = True)


# In[21]:


df = df.drop(df[df["mp_per_g"] < 28].index)


# In[22]:

with st.expander("Data Visualization"):
    mvp_votes['Category'] = 'MVP votes'
    no_mvp_votes['Category'] = 'No MVP votes'
    combined_mvp_votes = pd.concat([mvp_votes, no_mvp_votes])
    
    fig2 = px.histogram(
        combined_mvp_votes,
        x='pts_per_g',
        color = 'Category',
        nbins=20,
        barmode='overlay',
        histnorm='probability density',
        opacity=0.7,
        labels={'pts_per_g': 'Pts Per Game'},
        title='Pts Per Game MVP votes vs No MVP votes'
    )

    # Customize the layout
    fig2.update_layout(
        xaxis_title="Pts Per Game",
        yaxis_title="Proportion"
    )
    st.plotly_chart(fig2)
    pts_cutoff = mvp_votes["pts_per_g"].mean() - (mvp_votes["pts_per_g"].std()*3)
    st.write("Most of the players that have received MVP votes over the past two decades have averaged 20 points per game. The average for all players within this time is 8.4.")

    
    fig3 = px.histogram(
        combined_mvp_votes,
        x='win_loss_pct',
        color = 'Category',
        nbins=20,
        barmode='overlay',
        histnorm='probability density',
        opacity=0.7,
        labels={'win_loss_pct': 'Win Loss Pct'},
        title='Win Loss Pct MVP votes vs No MVP votes'
    )

    # Customize the layout
    fig3.update_layout(
        xaxis_title="Win Loss Pct",
        yaxis_title="Proportion"
    )
    st.plotly_chart(fig3)
    
    WL_cutoff = mvp_votes["win_loss_pct"].mean() - (mvp_votes["win_loss_pct"].std()*3)
    #st.write(f"Win Loss Cutoff to win MVP is {WL_cutoff:.3f}")
    st.write("Most players with MVP votes have a team win loss pct over 0.6. The average is 0.5")


# In[23]:


#mvp_votes[mvp_votes["pts_per_g"] < 6.9][["season", "player", "pts_per_g"]]


# In[24]:


df = df.drop(df[df["pts_per_g"] < pts_cutoff].index)


# In[ ]:





# In[25]:


#mvp_votes[mvp_votes["win_loss_pct"]<0.314][["season", "player", "pts_per_g", "win_loss_pct"]]


# In[26]:


df = df.drop(df[df["win_loss_pct"]<WL_cutoff].index)


# In[27]:


received_votes = df["award_share"] > 0


# In[28]:


print(received_votes.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(received_votes.value_counts(), autopct='%.2f')


# In[29]:


columns = df.columns.tolist()
columns = [c for c in columns if c not in ["player", "pos", "team_id", "was_mvp"]]
target = "award_share"
X = df[columns]
Y = df[target] > 0


# In[30]:


UnSmote = len(columns)


# Resample data using SMOTE

# In[31]:


from collections import Counter
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42, sampling_strategy=1)
x_res, y_res = sm.fit_resample(X,Y)


# In[32]:


fig1, ax1 = plt.subplots()
ax1.pie(y_res.value_counts(), autopct='%.2f')


# In[33]:


Unsmote = len(df)


# In[34]:


#x_res


# In[35]:


smote_df = x_res.merge(df, how='left')
#smote_df.shape


# In[36]:


smote_df["is_smote"] = smote_df.index >= Unsmote
#smote_df


# ## Now find the important features

# In[37]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[38]:


smote_df["was_mvp"] = smote_df["was_mvp"].fillna(value = False)


# In[39]:


smote_df.sort_values(by="was_mvp", ascending=False)


# In[40]:


col = [c for c in columns if c not in ["was_mvp", "award_share"]]
X = smote_df[col]
y = smote_df["was_mvp"]


# Use random forest classifier to find the most important features

# In[41]:


clf = RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X, y)


# In[42]:


feature_importances = clf.feature_importances_
features = X.columns

with st.expander("Key Features in Determining MVP"):
    # Create a dataframe for plotting
    importance_df = pd.DataFrame({
        'Metric': features,
        'Importance': feature_importances
    })
    
    fig4 = px.bar(
        importance_df,
        x='Metric',
        y='Importance',
        title='Random Forest Features Classifier',
        labels={'Importance': 'Importance', 'Metric': 'Features'},
        hover_data={'Metric': True, 'Importance': True}
    )
    
    # Customize the hover information
    fig4.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>" +
                      "Importance: %{y}<br>" +
                      "<extra></extra>",
        hovertext=importance_df['Metric']
    )

    # Display the Plotly figure in Streamlit
    st.write('What statistical metrics have the most importance for who wins MVP?')
    st.plotly_chart(fig4)
    
#    feature_importances = clf.feature_importances_
#    feature_importance_df = pd.DataFrame({'Feature': range(X.shape[1]), 'Importance': feature_importances})
#    feature_importance_df.sort_values(by=["Importance"], ascending=False).head(10)

#    all_columns = pd.DataFrame(smote_df.columns.tolist())
#    all_columns['Feature'] = all_columns.index
#    all_columns.head()

#    important_columns = feature_importance_df.merge(all_columns, how = "left")
#    important_columns = important_columns.rename(columns={0: "Metric"}).sort_values(by="Importance", ascending=False).head(5)
#    st.write(important_columns)
    st.markdown("""
    ### Top 5 metrics to determine MVP:
    1. Win shares per 48
    2. Win shares
    3. Box Plus Minus
    4. Offensive win shares
    5. Margin of victory (adjusted)
    """)


# In[43]:


all_columns = pd.DataFrame(smote_df.columns.tolist())
all_columns['Feature'] = all_columns.index
all_columns.head()


# Top 5 metrics to determine MVP are: <br> 1. Win shares per 48<br> 2. Win shares <br> 3. Box plus minus <br> 4. Offensive win shares <br> 5. Margin of victory (adjusted)

# In[44]:


#smote_df


# In[45]:


columns = ['age', 'g', 'gs', 'mp_per_g', 'fga_per_g', 'fg_pct',
        'fg2a_per_g', 'fg2_pct', 'efg_pct', 'fta_per_g', 'ft_pct', 'drb_per_g',
        'ast_per_g', 'stl_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp',
        'per', 'ts_pct', 'fta_per_fga_pct', 'drb_pct', 'trb_pct', 'ast_pct',
        'stl_pct', 'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_48',
        'obpm', 'dbpm', 'bpm', 'vorp','mov', 'mov_adj','win_loss_pct']


# In[46]:


train_info = smote_df[["season", "player", "pos", "team_id", "award_share","is_smote", "was_mvp"]]
train_data = smote_df[columns]
target = smote_df[["award_share"]]


# In[47]:



# Import 2023-2024 Data from Basketball Reference

# In[ ]:


import requests
from bs4 import BeautifulSoup


# In[ ]:


url = "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html"


# In[ ]:


response = requests.get(url)


# In[ ]:


player_data_list = []
if response.status_code == 200:
  # Parse the HTML content
  soup = BeautifulSoup(response.content, "html.parser")

  # Find the HTML elements containing the player stats
  player_stats_table = soup.find("table", {"id": "per_game_stats"})

  # Extract the data from the table (you'll need to adapt this based on the table structure)
  for row in player_stats_table.find("tbody").find_all("tr"):
    columns = row.find_all(["th","td"])
    player_name = columns[1].text.strip()
    other_columns = [col.text.strip() for col in columns[2:30]]
    #points_per_game = columns[28].text.strip()
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
#print(player_stats_df)


# In[ ]:


url_advanced_stats = "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html"


# In[ ]:


response = requests.get(url_advanced_stats)


# In[ ]:


player_advanced_list = []
if response.status_code == 200:
  # Parse the HTML content
  soup = BeautifulSoup(response.content, "html.parser")

  # Find the HTML elements containing the player stats
  player_stats_table = soup.find("table", {"id": "advanced_stats"})

  # Extract the data from the table (you'll need to adapt this based on the table structure)
  for row in player_stats_table.find("tbody").find_all("tr"):
    columns = row.find_all(["th","td"])
    player_name = columns[1].text.strip()
    other_columns = [col.text.strip() for col in columns[6:29]]
    #points_per_game = columns[28].text.strip()
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
#print(player_advanced_df)


# In[ ]:


player_advanced_df = player_advanced_df.drop(columns = ["", ""])


# In[ ]:


nba_team_stats = "https://www.basketball-reference.com/leagues/NBA_2024.html"


# In[ ]:


response = requests.get(nba_team_stats)


# In[ ]:


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
    other_columns = [col.text.strip() for col in columns[3:5]+columns[7:9]+columns[10:12]]
    #points_per_game = columns[28].text.strip()
    team_dict = {"Team": team_name}
    for i, col_text in enumerate(other_columns):
      team_dict[f"Column_{i+1}"] = col_text
    nba_advanced_list.append(team_dict)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
column_names = ["team_id", "win","loss","mov","sos","ortg", "drtg"]
nba_advanced_df = pd.DataFrame(nba_advanced_list)
nba_advanced_df.columns = column_names
#print(nba_advanced_df)


# In[ ]:


tm_name = nba_advanced_df["team_id"]
tm_abbrev = pd.DataFrame({'tm_abbreviation': ['BOS', 'OKC', 'LAC', 'MIN', 'NYK', 'PHI', 'CLE', 'DEN', 'MIL', 'NOP', 'PHX', 'IND', 'SAC', 'HOU', 'ORL', 'GSW', 'DAL', 'BKN', 'LAL',
                                  'MIA', 'CHI', 'UTA', 'ATL', 'TOR', 'MEM', 'POR', 'SAS', 'WAS', 'DET', 'CHA']})


# In[ ]:


nba_advanced_df["mov"] = nba_advanced_df["mov"].astype(float)
nba_advanced_df["sos"] = nba_advanced_df["sos"].astype(float)
nba_advanced_df["win"] = nba_advanced_df["win"].astype(int)
nba_advanced_df["loss"] = nba_advanced_df["loss"].astype(int)
nba_advanced_df["ortg"] = nba_advanced_df["ortg"].astype(float)
nba_advanced_df["drtg"] = nba_advanced_df["drtg"].astype(float)


# In[ ]:


for i in tm_name.index:
  nba_advanced_df = nba_advanced_df.replace([tm_name.loc[i]], tm_abbrev.loc[i])
nba_advanced_df["mov_adj"] = nba_advanced_df[["mov","sos"]].sum(axis=1)
nba_advanced_df["win_loss_pct"] = nba_advanced_df.apply(lambda row: row["win"]/(row["win"] + row["loss"]), axis=1)
#nba_advanced_df


# In[ ]:


nba_advanced_df_columns = ["team_id", "mov", "mov_adj", "win_loss_pct"]


# In[ ]:


nba_tm_advanced_df = nba_advanced_df[nba_advanced_df_columns]
#nba_tm_advanced_df


# In[ ]:

import os


# In[ ]:


from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# In[ ]:

with st.expander("2023-2024 Team Performance"):
    x = nba_advanced_df["ortg"]
    y = nba_advanced_df["drtg"]

    fig6 = px.scatter(
        nba_advanced_df,
        x = "ortg",
        y = "drtg",
        color = np.random.rand(len(nba_advanced_df)),
        hover_data = {"ortg": True, "drtg": True, "team_id": True},
        labels = {'ortg': 'ORTG', 'drtg': 'DRTG'}
    )

    # Customize hover info
    fig6.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>" +
                      "ORTG: %{x}<br>" +
                      "DRTG: %{y}<br>" +
                      "<extra></extra>",
        hovertext=nba_advanced_df["team_id"]
    )

    # Customize the layout
    fig6.update_layout(
        title = "Offensive and Defensive Team Ratings",
        xaxis_title="ORTG",
        yaxis_title="DRTG",
        yaxis=dict(autorange="reversed"), # Invert y-axis
        coloraxis_showscale=False # Hide color scale
    )

    st.plotly_chart(fig6)


# In[ ]:


merged_df = pd.merge(player_stats_df, player_advanced_df, on='player', how='inner')


# In[ ]:


merged_df = pd.merge(merged_df, nba_tm_advanced_df, on='team_id', how='inner')
#merged_df


# In[ ]:


merged_df = merged_df.drop(columns=['fg_per_g', 'fg3_per_g', 'fg2_per_g', 'ft_per_g'])


# In[ ]:


columns_ty = ['age', 'g', 'gs', 'mp_per_g', 'fga_per_g', 'fg_pct',
        'fg2a_per_g', 'fg2_pct', 'efg_pct', 'fta_per_g', 'ft_pct', 'drb_per_g',
        'ast_per_g', 'stl_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp',
        'per', 'ts_pct', 'fta_per_fga_pct', 'drb_pct', 'trb_pct', 'ast_pct',
        'stl_pct', 'tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_48',
        'obpm', 'dbpm', 'bpm', 'vorp', 'mov', 'mov_adj', 'win_loss_pct']


# In[ ]:


merged_df['age'] = merged_df['age'].replace('Age', np.nan)


# In[ ]:


#Check for duplicates, some players play for multiple teams during the season via trade or free agency
merged_df.player.duplicated().sum()


# In[ ]:


merged_df = merged_df[merged_df['team_id'] != 'TOT'].drop_duplicates(subset=['player'])


# In[ ]:


with st.expander("2023-2024 Player Stats"):
    st.write(merged_df)


# In[ ]:


merged_df.isnull().sum()


# In[ ]:


merged_df = merged_df.replace('', np.nan)


# In[ ]:


columns_to_int = ['age', 'g', 'gs', 'mp']
columns_to_float = ['mp_per_g', 'fga_per_g', 'fg_pct', 'fg3a_per_g', 'fg3_pct',
        'fg2a_per_g', 'fg2_pct', 'efg_pct', 'fta_per_g', 'ft_pct', 'drb_per_g',
        'ast_per_g', 'stl_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g',
        'per', 'ts_pct', 'fg3a_per_fga_pct','fta_per_fga_pct', 'orb_pct','drb_pct','trb_pct','ast_pct',
        'stl_pct', 'blk_pct','tov_pct', 'usg_pct', 'ows', 'dws', 'ws', 'ws_per_48',
        'obpm', 'dbpm', 'bpm', 'vorp', 'mov', 'mov_adj', 'win_loss_pct']
merged_df[columns_to_int] = merged_df[columns_to_int].astype(int)
merged_df[columns_to_float] = merged_df[columns_to_float].astype(float)


# In[ ]:


merged_df.info()


# In[ ]:


#Add games_cutoff for players ineligible for MVP. New rule requires players to play 65 games
merged_df = merged_df.drop(merged_df[merged_df["g"] < 64].index)


# In[ ]:


ty_test = merged_df[columns_ty]


# In[ ]:


test_info = ["player", "gs","pos", "age", "team_id", "pts_per_g", "tr_per_g", "ast_per_g", "award_share"]

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# In[ ]:


seasons = smote_df["season"].unique()
for season in seasons:
    tr = train_data[train_info["season"] != season]
    targ = target[train_info["season"] != season]

xgb = XGBRegressor()
xgb.fit(tr, targ.values.ravel())

# In[ ]:

# In[ ]:
    
xgb_ty_pred = xgb.predict(ty_test)
merged_df["award_share"] = xgb_ty_pred
mvp_winner_pred = merged_df.iloc[np.argsort(xgb_ty_pred)[-1:]]
top_two = merged_df.iloc[np.argsort(xgb_ty_pred)[-10:]]
pred_player = mvp_winner_pred["player"].to_string(index=False)
top_two_info = top_two[test_info].sort_values(by='award_share', ascending=False)
predicted_share = mvp_winner_pred["award_share"].to_string(index=False)
print(f"Predicted MVP Winner: {pred_player}")
print(f"Predicted Award Share: {predicted_share}")
print()
print(f"Top Two: ")
print(top_two_info)
#with st.expander("Section 7: 2023-2024 MVP Ladder"):
top_two_info = top_two_info.drop('award_share', axis=1)
st.title("2023-2024 MVP Ladder")
st.write(top_two_info)


# In[ ]:




