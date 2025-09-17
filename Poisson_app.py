import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson

st.title("Football Poisson Model - Bookmaker Style")

# ---------------------------
# Input team stats
# ---------------------------
st.sidebar.header("Enter Team Stats")
home_team = st.sidebar.text_input("Home Team", "Real Madrid")
away_team = st.sidebar.text_input("Away Team", "Barcelona")

home_attack = st.sidebar.number_input(f"{home_team} Avg Goals Scored", 0.0, 10.0, 2.3)
home_defense = st.sidebar.number_input(f"{home_team} Avg Goals Conceded", 0.0, 10.0, 0.9)
away_attack = st.sidebar.number_input(f"{away_team} Avg Goals Scored", 0.0, 10.0, 1.8)
away_defense = st.sidebar.number_input(f"{away_team} Avg Goals Conceded", 0.0, 10.0, 1.1)
league_avg_goals = st.sidebar.number_input("League Avg Goals/Game", 0.0, 10.0, 2.5)
max_goals = st.sidebar.number_input("Max Goals Considered", 1, 10, 10)

# ---------------------------
# Expected goals calculation
# ---------------------------
home_exp_goals = home_attack * away_defense / league_avg_goals
away_exp_goals = away_attack * home_defense / league_avg_goals
st.write(f"### Expected Goals")
st.write(f"{home_team}: {home_exp_goals:.2f}, {away_team}: {away_exp_goals:.2f}")

# ---------------------------
# Score probabilities
# ---------------------------
home_probs = [poisson.pmf(i, home_exp_goals) for i in range(max_goals)]
away_probs = [poisson.pmf(i, away_exp_goals) for i in range(max_goals)]
score_matrix = pd.DataFrame(np.outer(home_probs, away_probs),
                            index=[f"{i} goals" for i in range(max_goals)],
                            columns=[f"{i} goals" for i in range(max_goals)])

# ---------------------------
# Match outcome probabilities
# ---------------------------
home_win_prob = np.sum(np.tril(score_matrix.values, -1))
draw_prob = np.sum(np.diag(score_matrix.values))
away_win_prob = np.sum(np.triu(score_matrix.values, 1))

st.write("### Match Outcome Probabilities")
st.write(f"{home_team} Win: {home_win_prob*100:.1f}%")
st.write(f"Draw: {draw_prob*100:.1f}%")
st.write(f"{away_team} Win: {away_win_prob*100:.1f}%")

# ---------------------------
# Direct Win Suggestion
# ---------------------------
outcomes = {"Home Win": home_win_prob, "Draw": draw_prob, "Away Win": away_win_prob}
best_outcome = max(outcomes, key=outcomes.get)
st.write(f"### Suggested Direct Win Bet: **{best_outcome}**")

# ---------------------------
# Over/Under lines suggestion
# ---------------------------
total_goals_probs = np.zeros(2*max_goals)
for i in range(max_goals):
    for j in range(max_goals):
        total_goals_probs[i+j] += score_matrix.iloc[i,j]

st.write("### Suggested Over/Under Lines")
lines = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
for line in lines:
    over_prob = np.sum(total_goals_probs[int(line+0.5):])
    under_prob = np.sum(total_goals_probs[:int(line+0.5)])
    st.write(f"Over/Under {line} → Over: {over_prob*100:.1f}%, Under: {under_prob*100:.1f}%")

# Suggest the line closest to 50/50
best_line = min(lines, key=lambda l: abs(np.sum(total_goals_probs[int(l+0.5):]) - np.sum(total_goals_probs[:int(l+0.5)])))
st.write(f"**Optimal Over/Under Line:** {best_line} (closest to balanced probability)")

# Optional: Show full score matrix
if st.checkbox("Show Score Probability Matrix"):
    st.write(score_matrix)
