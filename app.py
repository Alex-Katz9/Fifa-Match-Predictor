import streamlit as st
import pandas as pd
import xgboost as xg
import pickle

win_loss_prediction_model = pickle.load(open('finalized_model_win_loss_prediction.sav', 'rb'))
score_diff_prediction_model = pickle.load(open('finalized_model_score_prediction.sav', 'rb'))
all_games = pd.read_pickle('model_table4.pkl')

st.markdown('# Fifa Match Predictor')

# st.sidebar.markdown('## Starting 11')
st.sidebar.markdown('## New Match')
st.sidebar.markdown(
    'To generate a new match, enter a soccer season in the same format as the example input, then enter two teams, home and away, then click, **Game On!**',
    unsafe_allow_html=True)
# st.sidebar.markdown(
#     'To generate a new team instance, just enter a soccer season in the same format as the example input, then enter starting 11 player lineups for both home and away teams, then click, **Game On!**',
#     unsafe_allow_html=True)

st.sidebar.markdown('## Notes')
st.sidebar.markdown(
    "<ul><li>It is important to note as a user that whichever team is selected as Home will inherantly be slightly more favored to win.</li><li>Displayed after user input will be the result of the match with respect to the home team entered.</li></ul>",
    unsafe_allow_html=True)

st.sidebar.markdown(
    '<img src="https://losangeles-mp7static.mlsdigital.net/styles/image_landscape/s3/images/USATSI_13131177.jpg?xjHzWNqdNtczSFx04lIcLKdWqI_IOAB4&itok=AFK5BOsk&c=bf236e045c21367079d12ba5a2e9386c" width="300"/> ',
    unsafe_allow_html=True)

###### Team Lineups Input ######
st.markdown('## Match Simulation')

# user input

Season = st.text_input("Season", '2019/2020')
HomeTeam = st.text_input("Home Team", 'Arsenal')
AwayTeam = st.text_input("Away Team", 'Villarreal CF')


# match button
match_button = st.button('Simulate Match')
if match_button:
    try:

        home_games = all_games.loc[(all_games['Season'] == Season) & (all_games['HomeTeam'] == HomeTeam)]

        home_game = home_games.iloc[0][['Season', 'offense_mean_H', 'defense_mean_H', 'mentality_mean_H',
                                        'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H']]

        away_games = all_games.loc[(all_games['Season'] == Season) & (all_games['AwayTeam'] == AwayTeam)]
        away_game = away_games.iloc[0][['Season', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',
                                        'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']]

        df = pd.concat([home_game, away_game], axis=0)
        X_all = df[['Season', 'offense_mean_H', 'defense_mean_H', 'mentality_mean_H',
                    'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H', 'offense_mean_A', 'defense_mean_A',
                    'mentality_mean_A',
                    'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']].copy()

        #         from sklearn.preprocessing import scale

        cols = [['offense_mean_H', 'defense_mean_H', 'mentality_mean_H',
                 'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H', 'offense_mean_A', 'defense_mean_A',
                 'mentality_mean_A',
                 'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']]

        X_all = pd.DataFrame(data=X_all)
        X_all = X_all.T
        X_all.drop(['Season'], axis=1, inplace=True)
        X_all = X_all.astype('float')



        score_diff = score_diff_prediction_model.predict(X_all)
        result = ''
        if score_diff < -0.5:
            result = "Loss!"
        elif score_diff > 0.5:
            result = "Win!"
        else:
            result = "Tie"

        st.write(result)


    except Exception as e:
        st.exception("Exception: %s\n" % e)



