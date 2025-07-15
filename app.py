import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- DB connection ---
@st.cache_data(ttl=600)
def load_data():
    user = "root"
    password = "root"
    host = "localhost"
    database = "imdb2024"
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
    
    query = """
    SELECT title, genre, rating, votes, duration
    FROM movies_scrabbed
    """
    df = pd.read_sql(query, engine)
    
    # Clean genre column (strip whitespace)
    df['genre'] = df['genre'].str.strip()
    
    # Fill missing values to avoid errors in widgets
    df['votes'] = df['votes'].fillna(0)
    df['rating'] = df['rating'].fillna(df['rating'].min())
    df['duration'] = df['duration'].fillna(df['duration'].median())
    
    return df

df = load_data()

st.title("🎬 IMDb Movies Dashboard")

# --- Filtering sidebar ---
st.sidebar.header("Filter Movies")

# Duration filter in hours (assuming duration is in minutes)
duration_filter = st.sidebar.selectbox(
    "Duration (Hrs)",
    options=["All", "< 2 hrs", "2-3 hrs", "> 3 hrs"]
)

rating_filter = st.sidebar.slider(
    "Minimum Rating",
    min_value=float(df['rating'].min()),
    max_value=float(df['rating'].max()),
    value=8.0,
    step=0.1
)

max_votes = int(df['votes'].max())
default_votes = 10000

# If max_votes < default_votes, set default_votes to max_votes
if max_votes < default_votes:
    default_votes = max_votes

votes_filter = st.sidebar.number_input(
    "Minimum Votes",
    min_value=0,
    max_value=max_votes,
    value=default_votes,
    step=1000
)

# Genre filter (multiselect)
unique_genres = sorted(df['genre'].unique())
genre_filter = st.sidebar.multiselect(
    "Select Genres",
    options=unique_genres,
    default=unique_genres
)

# Apply filters
filtered = df.copy()

if duration_filter == "< 2 hrs":
    filtered = filtered[filtered['duration'] < 120]
elif duration_filter == "2-3 hrs":
    filtered = filtered[(filtered['duration'] >= 120) & (filtered['duration'] <= 180)]
elif duration_filter == "> 3 hrs":
    filtered = filtered[filtered['duration'] > 180]

filtered = filtered[filtered['rating'] >= rating_filter]
filtered = filtered[filtered['votes'] >= votes_filter]
filtered = filtered[filtered['genre'].isin(genre_filter)]

st.markdown(f"### Showing {len(filtered)} movies after filtering")

# --- Your visualization code continues here ---


# --- Visualizations ---

# ✅ Top 10 Movies by Rating and Votes
# ----------------------
st.subheader("📌 Top 10 Movies by Rating and Votes")
top10 = filtered.sort_values(['rating', 'votes'], ascending=False).head(10)
st.dataframe(top10[['title', 'genre', 'rating', 'votes', 'duration']])

# ----------------------
# ✅ Genre Distribution
# ----------------------
st.subheader("📊 Genre Distribution")
genre_counts = filtered['genre'].value_counts()
fig1, ax1 = plt.subplots()
sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax1, palette="viridis")
ax1.set_ylabel("Count")
ax1.set_xlabel("Genre")
ax1.set_title("Number of Movies per Genre")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig1)

# ----------------------
# ✅ Average Duration by Genre
# ----------------------
st.subheader("⏱️ Average Duration by Genre")
avg_duration = filtered.groupby('genre')['duration'].mean().sort_values()
fig2, ax2 = plt.subplots()
sns.barplot(x=avg_duration.values, y=avg_duration.index, ax=ax2, palette="magma")
ax2.set_xlabel("Average Duration (min)")
ax2.set_ylabel("Genre")
st.pyplot(fig2)

# ----------------------
# ✅ Average Votes by Genre
# ----------------------
st.subheader("✅ Average Votes by Genre")
avg_votes = filtered.groupby('genre')['votes'].mean().sort_values(ascending=False)

if not avg_votes.empty and avg_votes.sum() > 0:
    fig3, ax3 = plt.subplots()
    sns.barplot(x=avg_votes.values, y=avg_votes.index, ax=ax3, palette="cubehelix")
    ax3.set_xlabel("Average Votes")
    ax3.set_ylabel("Genre")
    st.pyplot(fig3)
else:
    st.warning("No valid vote data available for average vote chart.")

# ----------------------
# ✅ Rating Distribution
# ----------------------
st.subheader("⭐ Rating Distribution")
fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(filtered['rating'], bins=20, kde=True, ax=ax4, color='skyblue')
ax4.set_title("Histogram of Ratings")
sns.boxplot(x=filtered['rating'], ax=ax5, color='lightgreen')
ax5.set_title("Boxplot of Ratings")
st.pyplot(fig4)

# ----------------------visulazition
# ✅ Top 10 Movies by Rating and Votes
# ----------------------
st.subheader("📌 Top 10 Movies by Rating and Votes")
top10 = filtered.sort_values(['rating', 'votes'], ascending=False).head(10)
st.dataframe(top10[['title', 'genre', 'rating', 'votes', 'duration']])

# ----------------------
# ✅ Genre Distribution
# ----------------------
st.subheader("📊 Genre Distribution")
genre_counts = filtered['genre'].value_counts()
fig1, ax1 = plt.subplots()
sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax1, palette="viridis")
ax1.set_ylabel("Count")
ax1.set_xlabel("Genre")
ax1.set_title("Number of Movies per Genre")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
st.pyplot(fig1)

# ----------------------
# ✅ Average Duration by Genre
# ----------------------
st.subheader("⏱️ Average Duration by Genre")
avg_duration = filtered.groupby('genre')['duration'].mean().sort_values()
fig2, ax2 = plt.subplots()
sns.barplot(x=avg_duration.values, y=avg_duration.index, ax=ax2, palette="magma")
ax2.set_xlabel("Average Duration (min)")
ax2.set_ylabel("Genre")
st.pyplot(fig2)

# ----------------------
# ✅ Rating Distribution
# ----------------------
st.subheader("⭐ Rating Distribution")
fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(filtered['rating'], bins=20, kde=True, ax=ax4, color='skyblue')
ax4.set_title("Histogram of Ratings")
sns.boxplot(x=filtered['rating'], ax=ax5, color='lightgreen')
ax5.set_title("Boxplot of Ratings")
st.pyplot(fig4)

# ----------------------
# ✅ Top-Rated Movie per Genre
# ----------------------
st.subheader("🏆 Top-Rated Movie per Genre")
top_per_genre = filtered.loc[filtered.groupby('genre')['rating'].idxmax()]
st.dataframe(top_per_genre[['genre', 'title', 'rating', 'votes', 'duration']].sort_values('genre'))

# ----------------------
# ✅ Duration Extremes
# ----------------------
st.subheader("🎥 Duration Extremes")
if not filtered.empty:
    shortest = filtered.loc[filtered['duration'].idxmin()]
    longest = filtered.loc[filtered['duration'].idxmax()]
    st.markdown(f"- **Shortest Movie:** {shortest['title']} – {shortest['duration']} mins")
    st.markdown(f"- **Longest Movie:** {longest['title']} – {longest['duration']} mins")
else:
    st.warning("No data to show duration extremes.")

# ----------------------
# ✅ Average Ratings by Genre (Heatmap)
# ----------------------
st.subheader("🔥 Average Ratings by Genre (Heatmap)")
ratings_heat = filtered.pivot_table(index='genre', values='rating')
fig6, ax6 = plt.subplots(figsize=(6, max(4, len(ratings_heat)/2)))
sns.heatmap(ratings_heat, annot=True, cmap="YlOrBr", cbar=False, ax=ax6)
ax6.set_ylabel("Genre")
st.pyplot(fig6)

# ----------------------
# ✅ 🎯 Top 10 Genres by Average Rating (NEW)
# ----------------------
st.subheader("🎯 Top 10 Genres by Average Rating")
genre_rating = filtered.groupby('genre')['rating'].mean().sort_values(ascending=False).head(10)
fig8, ax8 = plt.subplots()
sns.barplot(x=genre_rating.values, y=genre_rating.index, ax=ax8, palette="coolwarm")
ax8.set_xlabel("Average Rating")
ax8.set_ylabel("Genre")
ax8.set_title("Top 10 Genres by Average Rating")
st.pyplot(fig8)

# ----------------------
# ✅ 📅 Yearly Average Rating Trend (NEW)
# ----------------------
if 'year' in filtered.columns:
    st.subheader("📅 Yearly Rating Trends")
    year_rating = filtered.groupby('year')['rating'].mean().sort_index()
    fig9 = px.line(
        x=year_rating.index,
        y=year_rating.values,
        labels={'x': 'Year', 'y': 'Average Rating'},
        title="Average Movie Rating Over Years"
    )
    st.plotly_chart(fig9)
else:
    st.info("No 'year' column found to plot yearly rating trend.")

# ----------------------
# ✅ Ratings vs Votes (Scatter Plot)
# ----------------------
st.subheader("📈 Ratings vs Votes (Scatter Plot)")
fig7 = px.scatter(
    filtered,
    x='votes',
    y='rating',
    hover_data=['title'],
    title="Correlation Between Votes and Ratings",
    labels={"votes": "Number of Votes", "rating": "Rating"},
    color='genre'
)
st.plotly_chart(fig7)

# ----------------------
# ✅ Show Final Filtered Data
# ----------------------
st.subheader("📂 Filtered Movie Data")
st.dataframe(filtered)