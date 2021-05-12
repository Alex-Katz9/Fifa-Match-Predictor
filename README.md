# Fifa Match Predictor
 Utilizing player statistics in EA Sport's Fifa to predict European soccer match results. The dataset for player statistics was found on Sofifa.com and the soccer match data was captured from a SQL database on Kaggle.com. The match data set contained international soccer match data dating back to 2014. Here are the features included:
Season', 
'Datetime', 
'League', 
'HomeTeam', 
'AwayTeam', 
'FTHG', 
'FTAG',
'Hwinodds', 
'Dodds', 
'Awinodds'

The first step was to create our categorical target feature column in the match data set, win/tie/loss. Next I set about selecting relevant metrics from the players dataset and then grouping them into 6 categories.
Defense: defending, defensive marking, defending standing tackle, defending sliding tackle
Offense: passing, dribbling, attacking crossing, attacking finishing, attacking heading_accuracy, skill long passing, skill ball control
Agility/Fitness: pace, movement acceleration, movement sprint_speed, movement agility, movement reactions, movement balance, power jumping, power stamina, power strength
Goalkeeping: gk diving, gk handling, gk kicking, gk reflexes, gk speed, gk positioning, goalkeeping diving, goalkeeping handling, goalkeeping kicking, goalkeeping positioning, goalkeeping reflexes)
Scoring: shooting, attacking crossing, attacking finishing, attacking heading accuracy, skill fk accuracy, power shot power, power long_shots)
Mentality: mentality aggression, mentality interceptions, mentality positioning, mentality vision, mentality penalties

Each of these features were evaluated on a 0–100 scale (100 being best) and aggregated for each player in the dataset except for the Goalkeeping metric which was only calculated for players with goalkeeper as the listed position. Next the engineered features were averaged on a by team by season basis and then the two tables were merged.

In each match there are three possible outcomes: win or draw or lose. That makes for what’s called a multi-class classification problem.
It was important to note before I began that 45% of matches resulted in a home team win, 25% in draws and 30% in losses. The metric I decided to optimize for was the F1_score due to the importance it places on on false negative and false positive results allowing me to clearly see where the model was misclassifying and the fact that it balances precision and recall.
Precision = True Positives / All Positive Predictions
Recall = True Positives / Actual Positives

I tried logistic regression, SVC, and gradient boosting (XGBoost) algorithms to predict match results. However, none of the models were able to accurately capture ties. In order to work around this I established a machine learning pipeline using XGBoost Classification paired with XGBoost Regression whereby I first sought to distinguish ties from losses + wins and then wins from losses. This process dramatically improved the models performance with F1 score of 0.85 and 0.74 for the two measurements. After optimizing the models using GridSearch and comparing the predicted match results with the data I found where my predictions did fail, more often than not it was predicting a win when in actuality there was a tie or loss. Slight adjustments to regressor model (mainly adjusting the point-differential threshold declaring a game win/draw/loss) reduced these over-predictions by almost 50%.

The first app I created using Streamlit allows the user to input a season (2014-present) and then two teams in one of Europes top soccer leagues during that season even if the two teams did not actually play each other. ‘Simulating the match” outputs the models prediction of a home team win/draw/ loss.
The second app I created takes it a step further. Instead of two soccer club names entered the user may enter a custom 11 player starting lineup for each team. This allows the user to experiment with different lineups trying to achieve better results.
My system, as-is, can help fans, team owners/coaches and bettors:
Player/Lineup Insights: The player-centered version of the app would allow a team/owner or coach to identify shortcomings in their lineup and possibly encourage them subbing in another player or making a trade.
Proof of Concept: Player statistics (even one’s from a video game in this case) are a viable source for predicting real sports match results! Player statistics have been underutilized in most professional sports prediction models and could be greatly improved by incorporating something akin to this project.
