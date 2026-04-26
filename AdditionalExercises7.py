import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# STUDENT PARAMETERS

student_name = "Kurt Russel S. Belnas"
student_id = "TUPM-25-0291"
color_bar = "skyblue"
color_line = "red"
cmap_color = "viridis"


# LOAD DATASET

df = pd.read_csv("spotify_top_1000_tracks.csv")

# clean column names
df.columns = df.columns.str.strip()


# FIX: CREATE YEAR COLUMN

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["release_date"])  # remove invalid dates
df["year"] = df["release_date"].dt.year


# ENSURE NUMERIC TYPES

df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

df = df.dropna(subset=["duration_min", "popularity", "year"])



# 1. IDENTIFY LONGEST TRACKS


top10 = df.sort_values(by="duration_min", ascending=False).head(10)

print("\nTOP 10 LONGEST TRACKS")
print(top10[["track_name", "artist", "duration_min"]])

plt.figure(figsize=(10, 6))
plt.barh(top10["track_name"], top10["duration_min"], color=color_bar)
plt.xlabel("Duration (minutes)")
plt.ylabel("Track Name")
plt.title(f"Top 10 Longest Tracks - {student_name} ({student_id})")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



# 2. SONG RELEASE TREND


yearly_counts = df.groupby("year").size().sort_index()
cumulative = yearly_counts.cumsum()

plt.figure(figsize=(10, 6))

plt.bar(yearly_counts.index, yearly_counts.values,
        color=color_bar, alpha=0.6, label="Yearly Releases")

plt.plot(cumulative.index, cumulative.values,
         color=color_line, marker="o", label="Cumulative Releases")

plt.xlabel("Year")
plt.ylabel("Number of Songs")
plt.title(f"Song Release Trend - {student_name} ({student_id})")
plt.legend()
plt.tight_layout()
plt.show()



# 3. ANIMATED DENSITY MAP (HEXBIN)


years = sorted(df["year"].unique())

fig, ax = plt.subplots(figsize=(8, 6))

def update(year):
    ax.clear()
    subset = df[df["year"] == year]

    ax.hexbin(
        subset["duration_min"],
        subset["popularity"],
        gridsize=25,
        cmap=cmap_color,
        mincnt=1
    )

    ax.set_xlabel("Duration (min)")
    ax.set_ylabel("Popularity")
    ax.set_title(f"Duration vs Popularity ({year}) - {student_name} ({student_id})")

ani = animation.FuncAnimation(fig, update, frames=years, interval=800)

plt.show()