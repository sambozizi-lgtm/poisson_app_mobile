import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.title("Football Poisson Model - Batch Mode")

st.sidebar.header("Upload Match Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (matches)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    results = []

    for idx, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_attack = row['Home Avg Goals']
        home_defense = row['Home Avg Conceded']
        away_attack = row['Away Avg Goals']
        away_defense = row['Away Avg Conceded']
        league_avg = row['League Avg Goals']
        max_goals = 10

        home_exp = home_attack * away_defense / league_avg
        away_exp = away_attack * home_defense / league_avg

        home_probs = [poisson.pmf(i, home_exp) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_exp) for i in range(max_goals)]
        score_matrix = np.outer(home_probs, away_probs)

        home_win = np.sum(np.tril(score_matrix, -1))
        draw = np.sum(np.diag(score_matrix))
        away_win = np.sum(np.triu(score_matrix, 1))
        outcomes = {"Home Win": home_win, "Draw": draw, "Away Win": away_win}
        best_outcome = max(outcomes, key=outcomes.get)

        results.append({
            "Home Team": home_team,
            "Away Team": away_team,
            "Home Win %": round(home_win*100,1),
            "Draw %": round(draw*100,1),
            "Away Win %": round(away_win*100,1),
            "Direct Win Suggestion": best_outcome
        })

    result_df = pd.DataFrame(results)
    st.write("### Batch Match Predictions")
    st.dataframe(result_df)
else:
    st.info("Upload a CSV file with your matches to see predictions.")
