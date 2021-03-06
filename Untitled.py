{
 "cells": [
  {
   "cell_type": "raw",
   "id": "micro-element",
   "metadata": {},
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "\n",
    "import pickle\n",
    "with open('players.pkl', 'rb') as f:\n",
    "    new = pickle.load(f)\n",
    "\n",
    "all_games= pd.read_pickle('all_games.pkl')\n",
    "# team1 = st.selectbox(\"Home Team or Away Team: \",\n",
    "#                      ['Home', 'Away'])\n",
    "  \n",
    "# # print the selected hobby\n",
    "# st.write(\"Home Team: \", team1)\n",
    "\n",
    "ef user_input_features():\n",
    "    season = st.text_input(\"Season\", '2019/2020')\n",
    "    hteam = st.text_input(\"Home Team\", '2019/2020')\n",
    "#     hplayer1 = st.text_input(\"Left Back\", 'Xabi Alonso')\n",
    "#     hplayer2 = st.text_input(\"Left  Center Back\", 'Thiago Silva')\n",
    "#     hplayer3 = st.text_input(\"Right Center Back\", 'V. Kompany')\n",
    "#     hplayer4 = st.text_input(\"Right Back\", 'J. Terry')\n",
    "#     hplayer5 = st.text_input(\"Left Midfield\", 'J. Rodríguez')\n",
    "#     hplayer6 = st.text_input(\"Left Center Midfield\", 'David Silva')\n",
    "#     hplayer7 = st.text_input(\"Right Center Midfield\", 'L. Modrić')\n",
    "#     hplayer8 = st.text_input(\"Right Midfield\", 'E. Hazard')\n",
    "#     hplayer9 = st.text_input(\"Left Striker\", 'L. Suárez')\n",
    "#     hplayer10 = st.text_input(\"Right Striker\", 'Z. Ibrahimović')\n",
    "#     hplayer11 = st.text_input(\"Goalie\", 'M. Neuer')\n",
    "#     hteam = {'hplayer1': hplayer1\n",
    "#     aplayer1 = st.text_input(\"Left Back\", 'P. Pogba')\n",
    "#     aplayer2 = st.text_input(\"Left  Center Back\", 'Piqué')\n",
    "#     aplayer3 = st.text_input(\"Right Center Back\", 'Y. Touré')\n",
    "#     aplayer4 = st.text_input(\"Right Back\", 'Sergio Busquets')\n",
    "#     aplayer5 = st.text_input(\"Left Midfield\", 'G. Bale')\n",
    "#     aplayer6 = st.text_input(\"Left Center Midfield\", 'Neymar')\n",
    "#     aplayer7 = st.text_input(\"Right Center Midfield\", 'M. Özil')\n",
    "#     aplayer8 = st.text_input(\"Right Midfield\", 'T. Kroos')\n",
    "#     aplayer9 = st.text_input(\"Left Striker\", 'S. Agüero')\n",
    "#     aplayer10 = st.text_input(\"Right Striker\", 'T. Müller')\n",
    "#     aplayer11 = st.text_input(\"Goalie\", 'H. Lloris')\n",
    "        \n",
    "    \n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "wrapped-minutes",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "cosmetic-sailing",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import streamlit as st\n",
    "import pandas as pd\n",
    "\n",
    "import pickle\n",
    "win_loss_prediction_model = pickle.load(open('finalized_model_win_loss_prediction.sav', 'rb'))\n",
    "score_diff_prediction_model = pickle.load(open('finalized_model_score_prediction.sav', 'rb'))\n",
    "all_games= pd.read_pickle('model_table4.pkl')\n",
    "\n",
    "# ef user_input_features():\n",
    "#     Season = st.text_input(\"Season\", '2019/2020')\n",
    "#     HomeTeam = st.text_input(\"Home Team\", 'Arsenal')\n",
    "#     AwayTeam = st.text_input(\"Away Team\", 'Villarreal CF')\n",
    "      \n",
    "      \n",
    "\n",
    "    home_games = all_games.loc[(all_games['Season'] == Season) & (all_games['HomeTeam'] == HomeTeam)]\n",
    "\n",
    "    home_game = home_games.iloc[0][['Season','offense_mean_H', 'defense_mean_H', 'mentality_mean_H',\n",
    "           'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H']]\n",
    "    home_game\n",
    "    away_games = all_games.loc[(all_games['Season'] == Season) & (all_games['AwayTeam'] == AwayTeam)]\n",
    "    away_game = away_games.iloc[0][['Season', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',\n",
    "           'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']]\n",
    "\n",
    "    df = pd.concat([home_game, away_game], axis=0)\n",
    "    X_all = df[['Season','offense_mean_H', 'defense_mean_H', 'mentality_mean_H',\n",
    "           'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',\n",
    "           'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']].copy()\n",
    "    X_all['Season'] = 1\n",
    "\n",
    "    from sklearn.preprocessing import scale\n",
    "\n",
    "    cols = [['offense_mean_H', 'defense_mean_H', 'mentality_mean_H',\n",
    "           'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',\n",
    "           'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']]\n",
    "\n",
    "    for col in cols:\n",
    "        X_all[col] = scale(X_all[col])\n",
    "    X_all= pd.DataFrame(data=X_all)\n",
    "    X_all = X_all.T\n",
    "\n",
    "    X_all.drop(['Season'], axis = 1, inplace=True)\n",
    "    X_all\n",
    "    score_diff = score_diff_prediction_model.predict(X_all)\n",
    "    result = ''\n",
    "    if score_diff < -0.5:\n",
    "        result = \"Loss!\"\n",
    "    elif score_diff > 0.5:\n",
    "        result = \"Win!\"\n",
    "    else:\n",
    "        result = \"Tie\"\n",
    "    \n",
    "        \n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "protecting-medication",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.01064884], dtype=float32)"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Season = '2019/2020'\n",
    "HomeTeam = 'Arsenal'\n",
    "AwayTeam = 'Villarreal CF'\n",
    "home_games = all_games.loc[(all_games['Season'] == Season) & (all_games['HomeTeam'] == HomeTeam)]\n",
    "\n",
    "home_game = home_games.iloc[0][['Season','offense_mean_H', 'defense_mean_H', 'mentality_mean_H',\n",
    "       'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H']]\n",
    "home_game\n",
    "away_games = all_games.loc[(all_games['Season'] == Season) & (all_games['AwayTeam'] == AwayTeam)]\n",
    "away_game = away_games.iloc[0][['Season', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',\n",
    "       'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']]\n",
    "\n",
    "df = pd.concat([home_game, away_game], axis=0)\n",
    "X_all = df[['Season','offense_mean_H', 'defense_mean_H', 'mentality_mean_H',\n",
    "       'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',\n",
    "       'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']].copy()\n",
    "X_all['Season'] = 1\n",
    "\n",
    "from sklearn.preprocessing import scale\n",
    "\n",
    "cols = [['offense_mean_H', 'defense_mean_H', 'mentality_mean_H',\n",
    "       'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',\n",
    "       'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']]\n",
    "\n",
    "for col in cols:\n",
    "    X_all[col] = scale(X_all[col])\n",
    "X_all= pd.DataFrame(data=X_all)\n",
    "X_all = X_all.T\n",
    "\n",
    "X_all.drop(['Season'], axis = 1, inplace=True)\n",
    "X_all.dtypes\n",
    "X_all= X_all.astype('float')\n",
    "# X_all.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())\n",
    "\n",
    "score_diff_prediction_model.predict(X_all)\n",
    "score_diff = score_diff_prediction_model.predict(X_all)\n",
    "result = ''\n",
    "if score_diff < -0.1:\n",
    "    result = \"Loss!\"\n",
    "elif score_diff > 0.1:\n",
    "    result = \"Win!\"\n",
    "else:\n",
    "    result = \"Tie\"\n",
    "result\n",
    "score_diff"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "revolutionary-termination",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
