import streamlit as st
import pandas as pd
import xgboost as xg
import pickle

win_loss_prediction_model = pickle.load(open('finalized_model_win_loss_prediction.sav', 'rb'))
score_diff_prediction_model = pickle.load(open('finalized_model_score_prediction.sav', 'rb'))
all_games= pd.read_pickle('all_games.pkl')
with open('players.pkl', 'rb') as f:
    new = pickle.load(f)
st.markdown('# Fifa Match Predictor')

st.sidebar.markdown('## Starting 11')

st.sidebar.markdown(
    'To generate a new team instance, just enter a soccer season in the same format as the example input, then enter starting 11 player lineups for both home and away teams, then click, **Game On!**',
    unsafe_allow_html=True)

st.sidebar.markdown('## Notes')
st.sidebar.markdown(
    "<ul><li>It is important to note as a user that whichever team is selected as Home will inherantly be slightly more favored to win.</li><li>When entering a players name, enter the initial for their first name or in some cases their entire first name and their last name together</li><li>Displayed after user input will be the result of the match with respect to the home team entered.</li></ul>",
    unsafe_allow_html=True)

st.sidebar.markdown(
    '<img src="https://losangeles-mp7static.mlsdigital.net/styles/image_landscape/s3/images/USATSI_13131177.jpg?xjHzWNqdNtczSFx04lIcLKdWqI_IOAB4&itok=AFK5BOsk&c=bf236e045c21367079d12ba5a2e9386c" width="300"/> ',
    unsafe_allow_html=True)

###### Team Lineups Input ######
st.markdown('## Match Simulation')
Season = st.text_input("Season", '2019/2020')
st.markdown('### Home Team Roster')

hplayer1 = st.text_input("Left Back", 'Xabi Alonso')
hplayer2 = st.text_input("Left  Center Back", 'Thiago Silva')
hplayer3 = st.text_input("Right Center Back", 'V. Kompany')
hplayer4 = st.text_input("Right Back", 'J. Terry')
hplayer5 = st.text_input("Left Midfield", 'J. Rodríguez')
hplayer6 = st.text_input("Left Center Midfield", 'David Silva')
hplayer7 = st.text_input("Right Center Midfield", 'L. Modrić')
hplayer8 = st.text_input("Right Midfield", 'E. Hazard')
hplayer9 = st.text_input("Left Striker", 'L. Suárez')
hplayer10 = st.text_input("Right Striker", 'Z. Ibrahimović')
hplayer11 = st.text_input("Goalie", 'M. Neuer')

st.markdown('### Away Team Roster')
aplayer1 = st.text_input("A-Left Back", 'P. Pogba')
aplayer2 = st.text_input("A-Left  Center Back", 'Piqué')
aplayer3 = st.text_input("A-Right Center Back", 'Y. Touré')
aplayer4 = st.text_input("A-Right Back", 'Sergio Busquets')
aplayer5 = st.text_input("A-Left Midfield", 'G. Bale')
aplayer6 = st.text_input("A-Left Center Midfield", 'Neymar')
aplayer7 = st.text_input("A-Right Center Midfield", 'M. Özil')
aplayer8 = st.text_input("A-Right Midfield", 'T. Kroos')
aplayer9 = st.text_input("A-Left Striker", 'S. Agüero')
aplayer10 = st.text_input("A-Right Striker", 'T. Müller')
aplayer11 = st.text_input("A-Goalie", 'H. Lloris')
hplayers = [hplayer1, hplayer2, hplayer3, hplayer4, hplayer5, hplayer6, hplayer7, hplayer8, hplayer9, hplayer10,
                    hplayer11]
aplayers = [aplayer1, aplayer2, aplayer3, aplayer4, aplayer5, aplayer6, aplayer7, aplayer8, aplayer9, aplayer10,
                    aplayer11]

# match button
match_button = st.button('Simulate Match', key='1')
if match_button:
    try:

        def create_players(data, hplayers, aplayers):
            result_list = []
            for season in data:
                dream_team1 = pd.DataFrame(season.loc[season.short_name.apply(lambda x: x in hplayers)])
                dream_team1['club_name'] = 'HomeTeam'

                dream_team2 = pd.DataFrame(season.loc[season.short_name.apply(lambda x: x in aplayers)])
                dream_team2['club_name'] = 'AwayTeam'
                season = pd.concat([season, dream_team1, dream_team2], axis=0)
                result_list.append(season)
            return (result_list)
        new = create_players(new, hplayers, aplayers)


        def defense_column(fifas):
            years = ['2014/2015', "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021"]
            updated_fifas = []
            for i, season in enumerate(fifas):
                subs = season[(season['team_position'] == 'SUB')].index
                season.drop(subs, inplace=True)
                defending = season.groupby('club_name').agg({'defending': ['mean']})
                defending['overall'] = season.groupby('club_name').agg({'overall': ['mean']})

                defending['Season'] = years[i]
                updated_fifas.append(defending)
            return updated_fifas


        fifa_ags = defense_column(new)
        fifa_ags_def = fifa_ags[0].append(fifa_ags[1])
        fifa_ags_def = fifa_ags_def.append(fifa_ags[2])
        fifa_ags_def = fifa_ags_def.append(fifa_ags[3])
        fifa_ags_def = fifa_ags_def.append(fifa_ags[4])
        fifa_ags_def = fifa_ags_def.append(fifa_ags[5])
        fifa_ags_def = fifa_ags_def.append(fifa_ags[6])
        fifa_ags_def.columns = ['defending_mean', 'overall', 'Season']
        fifa_ags_def.index.name = 'club_name'
        fifa_ags_def.reset_index(inplace=True)
        fifa_ags_def['defense_mean'] = fifa_ags_def.mean(axis=1)


        # offense
        def offense_column(fifas):
            years = ['2014/2015', "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021"]
            offense_columns = ['passing', 'dribbling', 'attacking_crossing', 'attacking_finishing',
                               'attacking_heading_accuracy',
                               'skill_long_passing', 'skill_ball_control']
            updated_fifas = []
            for i, season in enumerate(fifas):
                subs = season[(season['team_position'] == 'SUB')].index
                season.drop(subs, inplace=True)
                offense = season.groupby('club_name').agg({'passing': ['mean']})
                offense['dribbling mean'] = season.groupby('club_name').agg({'dribbling': ['mean']})
                offense['attacking_crossing mean'] = season.groupby('club_name').agg({'attacking_crossing': ['mean']})
                offense['attacking_finishing mean'] = season.groupby('club_name').agg({'attacking_finishing': ['mean']})
                offense['attacking_heading_accuracy mean'] = season.groupby('club_name').agg(
                    {'attacking_heading_accuracy': ['mean']})
                offense['skill_long_passing'] = season.groupby('club_name').agg({'skill_long_passing': ['mean']})
                offense['skill_ball_control'] = season.groupby('club_name').agg({'skill_ball_control': ['mean']})
                offense['Season'] = years[i]

                updated_fifas.append(offense)
            return updated_fifas


        fifa_ags = offense_column(new)
        fifa_ags_off = fifa_ags[0].append(fifa_ags[1])
        fifa_ags_off = fifa_ags_off.append(fifa_ags[2])
        fifa_ags_off = fifa_ags_off.append(fifa_ags[3])
        fifa_ags_off = fifa_ags_off.append(fifa_ags[4])
        fifa_ags_off = fifa_ags_off.append(fifa_ags[5])
        fifa_ags_off = fifa_ags_off.append(fifa_ags[6])
        # fifa_ags_off.columns = ['defending_mean', 'season']
        fifa_ags_off.index.name = 'club_name'
        fifa_ags_off.reset_index(inplace=True)
        fifa_ags_off['offense_mean'] = fifa_ags_off.mean(axis=1)


        # agility
        def agility_fitness_column(fifas):
            years = ['2014/2015', "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021"]

            updated_fifas = []
            for i, season in enumerate(fifas):
                subs = season[(season['team_position'] == 'SUB')].index
                season.drop(subs, inplace=True)
                ag_fit = season.groupby('club_name').agg({'pace': ['mean']})

                ag_fit['movement_acceleration mean'] = season.groupby('club_name').agg(
                    {'movement_acceleration': ['mean']})
                ag_fit['movement_sprint_speed mean'] = season.groupby('club_name').agg(
                    {'movement_sprint_speed': ['mean']})
                ag_fit['movement_agility mean'] = season.groupby('club_name').agg({'movement_agility': ['mean']})
                ag_fit['movement_reactions mean'] = season.groupby('club_name').agg({'movement_reactions': ['mean']})
                ag_fit['movement_balance mean'] = season.groupby('club_name').agg({'movement_balance': ['mean']})
                ag_fit['power_jumping mean'] = season.groupby('club_name').agg({'power_jumping': ['mean']})
                ag_fit['power_stamina mean'] = season.groupby('club_name').agg({'power_stamina': ['mean']})
                ag_fit['power_strength mean'] = season.groupby('club_name').agg({'power_strength': ['mean']})

                ag_fit['Season'] = years[i]

                updated_fifas.append(ag_fit)
            return updated_fifas


        fifa_ags = agility_fitness_column(new)

        fifa_ags_ag_fit = fifa_ags[0].append(fifa_ags[1])
        fifa_ags_ag_fit = fifa_ags_ag_fit.append(fifa_ags[2])
        fifa_ags_ag_fit = fifa_ags_ag_fit.append(fifa_ags[3])
        fifa_ags_ag_fit = fifa_ags_ag_fit.append(fifa_ags[4])
        fifa_ags_ag_fit = fifa_ags_ag_fit.append(fifa_ags[5])
        fifa_ags_ag_fit = fifa_ags_ag_fit.append(fifa_ags[6])

        fifa_ags_ag_fit.index.name = 'club_name'
        fifa_ags_ag_fit.reset_index(inplace=True)
        fifa_ags_ag_fit['agility_fitness_mean'] = fifa_ags_ag_fit.mean(axis=1)


        # scoring

        def scoring_column(fifas):
            years = ['2014/2015', "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021"]

            updated_fifas = []
            for i, season in enumerate(fifas):
                subs = season[(season['team_position'] == 'SUB')].index
                season.drop(subs, inplace=True)
                scoring = season.groupby('club_name').agg({'shooting': ['mean']})

                scoring['attacking_crossing mean'] = season.groupby('club_name').agg({'attacking_crossing': ['mean']})
                scoring['attacking_finishing mean'] = season.groupby('club_name').agg({'attacking_finishing': ['mean']})
                scoring['attacking_heading_accuracy mean'] = season.groupby('club_name').agg(
                    {'attacking_heading_accuracy': ['mean']})
                scoring['skill_fk_accuracy mean'] = season.groupby('club_name').agg({'skill_fk_accuracy': ['mean']})
                scoring['power_shot_power mean'] = season.groupby('club_name').agg({'power_shot_power': ['mean']})
                scoring['power_long_shots mean'] = season.groupby('club_name').agg({'power_long_shots': ['mean']})

                scoring['Season'] = years[i]

                updated_fifas.append(scoring)
            return updated_fifas


        fifa_ags = scoring_column(new)

        fifa_ags_scoring = fifa_ags[0].append(fifa_ags[1])
        fifa_ags_scoring = fifa_ags_scoring.append(fifa_ags[2])
        fifa_ags_scoring = fifa_ags_scoring.append(fifa_ags[3])
        fifa_ags_scoring = fifa_ags_scoring.append(fifa_ags[4])
        fifa_ags_scoring = fifa_ags_scoring.append(fifa_ags[5])
        fifa_ags_scoring = fifa_ags_scoring.append(fifa_ags[6])

        fifa_ags_scoring.index.name = 'club_name'
        fifa_ags_scoring.reset_index(inplace=True)
        fifa_ags_scoring['scoring_mean'] = fifa_ags_scoring.mean(axis=1)


        # mentality
        def mentality_column(fifas):
            years = ['2014/2015', "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021"]

            updated_fifas = []
            for i, season in enumerate(fifas):
                subs = season[(season['team_position'] == 'SUB')].index
                season.drop(subs, inplace=True)
                mentality = season.groupby('club_name').agg({'mentality_aggression': ['mean']})

                mentality['mentality_interceptions mean'] = season.groupby('club_name').agg(
                    {'mentality_interceptions': ['mean']})
                mentality['mentality_positioning mean'] = season.groupby('club_name').agg(
                    {'mentality_positioning': ['mean']})
                mentality['mentality_vision mean'] = season.groupby('club_name').agg({'mentality_vision': ['mean']})
                mentality['mentality_penalties mean'] = season.groupby('club_name').agg(
                    {'mentality_penalties': ['mean']})

                mentality['Season'] = years[i]

                updated_fifas.append(mentality)
            return updated_fifas


        fifa_ags = mentality_column(new)

        fifa_ags_mentality = fifa_ags[0].append(fifa_ags[1])
        fifa_ags_mentality = fifa_ags_mentality.append(fifa_ags[2])
        fifa_ags_mentality = fifa_ags_mentality.append(fifa_ags[3])
        fifa_ags_mentality = fifa_ags_mentality.append(fifa_ags[4])
        fifa_ags_mentality = fifa_ags_mentality.append(fifa_ags[5])
        fifa_ags_mentality = fifa_ags_mentality.append(fifa_ags[6])

        fifa_ags_mentality.index.name = 'club_name'
        fifa_ags_mentality.reset_index(inplace=True)
        fifa_ags_mentality['mentality_mean'] = fifa_ags_mentality.mean(axis=1)


        # goalkeeping
        def goalkeeping_column(fifas):
            years = ['2014/2015', "2015/2016", "2016/2017", "2017/2018", "2018/2019", "2019/2020", "2020/2021"]

            updated_fifas = []
            for i, season in enumerate(fifas):
                subs = season[(season['team_position'] == 'SUB')].index
                season.drop(subs, inplace=True)
                goalkeeping = season.groupby(['club_name', 'team_position']).agg({'gk_diving': ['mean']})

                #         goalkeeping['gk_handling mean'] = season.groupby('club_name').agg({'gk_handling': ['mean']})
                goalkeeping['gk_kicking mean'] = season.groupby(['club_name', 'team_position']).agg(
                    {'gk_kicking': ['mean']})
                goalkeeping['gk_reflexes mean'] = season.groupby(['club_name', 'team_position']).agg(
                    {'gk_reflexes': ['mean']})
                goalkeeping['gk_speed mean'] = season.groupby(['club_name', 'team_position']).agg(
                    {'gk_speed': ['mean']})
                #         goalkeeping['gk_positioning mean'] = season.groupby('club_name').agg({'gk_positioning': ['mean']})
                goalkeeping['goalkeeping_diving mean'] = season.groupby(['club_name', 'team_position']).agg(
                    {'goalkeeping_diving': ['mean']})
                #         goalkeeping['goalkeeping_handling mean'] = season.groupby(['club_name', 'team_position']).agg({'goalkeeping_handling': ['mean']})
                goalkeeping['goalkeeping_kicking mean'] = season.groupby(['club_name', 'team_position']).agg(
                    {'goalkeeping_kicking': ['mean']})
                #         goalkeeping['goalkeeping_positioning mean'] = season.groupby(['club_name', 'team_position']).agg({'goalkeeping_positioning': ['mean']})
                goalkeeping['goalkeeping_reflexes mean'] = season.groupby(['club_name', 'team_position']).agg(
                    {'goalkeeping_reflexes': ['mean']})

                goalkeeping['Season'] = years[i]

                updated_fifas.append(goalkeeping)
            return updated_fifas


        fifa_ags = goalkeeping_column(new)

        fifa_ags_goalkeeping = fifa_ags[0].append(fifa_ags[1])
        fifa_ags_goalkeeping = fifa_ags_goalkeeping.append(fifa_ags[2])
        fifa_ags_goalkeeping = fifa_ags_goalkeeping.append(fifa_ags[3])
        fifa_ags_goalkeeping = fifa_ags_goalkeeping.append(fifa_ags[4])
        fifa_ags_goalkeeping = fifa_ags_goalkeeping.append(fifa_ags[5])
        fifa_ags_goalkeeping = fifa_ags_goalkeeping.append(fifa_ags[6])

        fifa_ags_goalkeeping.index.name = 'club_name'
        fifa_ags_goalkeeping.reset_index(inplace=True)
        fifa_ags_goalkeeping = fifa_ags_goalkeeping.loc[fifa_ags_goalkeeping['team_position'] == 'GK']
        fifa_ags_goalkeeping['goalkeeping_mean'] = fifa_ags_goalkeeping.mean(axis=1)

        fifa_ags_goalkeeping1 = fifa_ags_goalkeeping[['Season', 'club_name', 'goalkeeping_mean']].copy()

        fifa_ags_mentality1 = fifa_ags_mentality[['Season', 'club_name', 'mentality_mean']].copy()

        fifa_ags_scoring1 = fifa_ags_scoring[['Season', 'club_name', 'scoring_mean']].copy()

        fifa_ags_ag_fit1 = fifa_ags_ag_fit[['Season', 'club_name', 'agility_fitness_mean']].copy()

        fifa_ags_off1 = fifa_ags_off[['Season', 'club_name', 'offense_mean']].copy()

        fifa_ags_def1 = fifa_ags_def[['Season', 'club_name', 'defense_mean', 'overall']].copy()

        fifa_agg_stats = pd.merge(fifa_ags_mentality1, fifa_ags_scoring1, how='left',
                                  on=['Season', 'club_name'])
        fifa_agg_stats1 = pd.merge(fifa_agg_stats, fifa_ags_ag_fit1, how='left',
                                   left_on=['Season', 'club_name'], right_on=['Season', 'club_name'])
        fifa_agg_stats2 = pd.merge(fifa_agg_stats1, fifa_ags_off1, how='left',
                                   left_on=['Season', 'club_name'], right_on=['Season', 'club_name'])
        fifa_agg_stats4 = pd.merge(fifa_agg_stats2, fifa_ags_def1, how='left', on=['Season', 'club_name'])
        # fifa_agg_stats3 = pd.merge(fifa_agg_stats2, fifa_ags_def1, how='left',
        #                           left_on=['Season', 'club_name'], right_on = ['Season', 'club_name'])
        fifa_agg_stats4.columns = ['Season', 'club_name', 'ss', 'cn', 'mentality_mean', 'scoring_mean',
                                   'agility_fitness_mean',
                                   'offense_mean', 'defense_mean', 'overall']
        # fifa_ags_def.columns = ['defending_mean', 'Season']

        fifa_agg_stats5 = fifa_agg_stats4[['Season', 'club_name', 'offense_mean', 'defense_mean', 'mentality_mean',
                                           'scoring_mean', 'agility_fitness_mean', 'overall']].copy()

        new_row = {'Season': Season, 'Datetime': 0, 'League': 0, 'HomeTeam': 'HomeTeam', 'AwayTeam': 'AwayTeam',
                   'FTHG': 0, 'FTAG': 0, 'Hwinodds': 0, 'Dodds': 0, 'Awinodds': 0, 'win3_tie1_loss0': 0}
        all_games = all_games.append(new_row, ignore_index=True)

        combo_final2 = pd.merge(all_games, fifa_agg_stats5, how='left', left_on=['Season', 'HomeTeam'],
                                right_on=['Season', 'club_name'])
        combo_final = pd.merge(combo_final2, fifa_agg_stats5, how='left', left_on=['Season', 'AwayTeam'],
                               right_on=['Season', 'club_name'], suffixes=('_H', '_A'))
        combo_final1 = combo_final.dropna(axis=0)
        all_games2 = combo_final1.iloc[::-1]
        home_games = all_games2.loc[(all_games['Season'] == Season) & (all_games2['HomeTeam'] == 'HomeTeam')]

        home_game = home_games.iloc[0][['Season', 'offense_mean_H', 'defense_mean_H', 'mentality_mean_H',
                                        'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H']]

        away_games = all_games2.loc[(all_games['Season'] == Season) & (all_games2['AwayTeam'] == 'AwayTeam')]
        away_game = away_games.iloc[0][['Season', 'offense_mean_A', 'defense_mean_A', 'mentality_mean_A',
                                        'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']]

        df = pd.concat([home_game, away_game], axis=0)
        X_all = df[['Season', 'offense_mean_H', 'defense_mean_H', 'mentality_mean_H',
                    'scoring_mean_H', 'agility_fitness_mean_H', 'overall_H', 'offense_mean_A', 'defense_mean_A',
                    'mentality_mean_A',
                    'scoring_mean_A', 'agility_fitness_mean_A', 'overall_A']].copy()

        # #         from sklearn.preprocessing import scale

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
        if score_diff < -2:
            result = "Loss!"
        elif score_diff > 0.5:
            result = "Win!"
        else:
            result = "Tie"

        st.write(result)


    except Exception as e:
        st.exception("Exception: %s\n" % e)




