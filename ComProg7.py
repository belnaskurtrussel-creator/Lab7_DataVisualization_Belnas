import pandas as pd 
import os 
# Define the dataset path using your absolute file path 
dataset_path = r"C:\Users\Kurt Russel\OneDrive\Documents\Lab7_DataVisualization_Belnas\spotify_top_1000_tracks.csv"
# Load dataset 
df = pd.read_csv(dataset_path, encoding="utf-8") 
# Convert release_date and extract year 
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce') 
df['year'] = df['release_date'].dt.year 
# FIX: We REMOVE the line that tried to create 'duration_min'  
# because it already exists in the CSV file you loaded. 
# (The 'duration_min' column is ready for use!) 
print("   Dataset loaded and basic preprocessing complete!") 
print(df.head(3))



import numpy as np 
# Clean up text columns 
df['track_name'] = df['track_name'].str.strip() 
df['artist'] = df['artist'].str.strip() 
df['album'] = df['album'].str.strip() 
# Convert 'year' to integer 
df['year'] = df['year'].fillna(0).astype(int)

# Drop unnecessary columns 
cols_to_drop = ['spotify_url', 'id', 'release_date'] 
 
# Check for and add other common audio feature columns if they exist 
if 'time_signature' in df.columns: 
    cols_to_drop.append('time_signature') 
if 'key' in df.columns: 
    cols_to_drop.append('key') 
if 'mode' in df.columns: 
    cols_to_drop.append('mode') 
 
df = df.drop(columns=cols_to_drop, errors='ignore') 
 
# Feature Engineering: Tempo Category 
tempo_bins = [0, 100, 140, np.inf] 
tempo_labels = ['Slow', 'Medium', 'Fast'] 
 
if 'tempo' in df.columns: 
    df['tempo_category'] = pd.cut(  # Create tempo category column 
        df['tempo'], bins=tempo_bins,  
        labels=tempo_labels, right=False 
    ) 
    print("Feature 'tempo_category' created.") 
else: 
    print("Warning: 'tempo' column not found; skipping 'tempo_category' creation.") 
 
# Remove duplicates 
df = df.drop_duplicates(subset=['track_name', 'artist'], keep='first') 
 
print(f"   Data cleaning and feature engineering complete.") 
print(f"Final Row Count after deduplication: {len(df)}")



# Histogram

print("Histogram")

import matplotlib.pyplot as plt 
 
plt.figure(figsize=(10, 6)) 
plt.hist(df['popularity'], bins=40, color='indianred', 
edgecolor='darkred') 
plt.title('Distribution of Song Popularity', fontsize=14) 
plt.xlabel('Popularity Score (0-100)') 
plt.ylabel('Frequency') 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.show()



# Boxplot

print("Boxplot")

import matplotlib.pyplot as plt 
import seaborn as sns 
plt.figure(figsize=(12, 6)) 
sns.boxplot(x='year', y='popularity', data=df, palette='viridis') 
plt.title('Popularity Distribution by Release Year', fontsize=14) 
plt.xlabel('Release Year') 
plt.ylabel('Popularity Score') 
plt.xticks(rotation=45) 
plt.grid(True, linestyle='--', alpha=0.7) 
plt.show()



# Scatter Plot
print("Scatter Plot")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='duration_min',
    y='popularity',
    color='darkorange'
)

plt.title('Song Duration vs. Popularity', fontsize=14)
plt.xlabel('Duration (minutes)')
plt.ylabel('Popularity Score (0-100)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()



# Pair Plot
print("Pair Plot")

import matplotlib.pyplot as plt 
import seaborn as sns 
# Using only the confirmed numerical columns: 'duration_min' and 'popularity' 
key_features = ['duration_min', 'popularity'] 
# We use the simplified sns.pairplot function 
sns.pairplot( 
    df[key_features],  
    diag_kind='kde',  
    corner=True,  
    plot_kws={'alpha': 0.6, 'color': '#990000'} 
) 
plt.suptitle('Pair Plot of Duration and Popularity', y=1.02, fontsize=16) 
plt.show()



# Joint Plot
print("Joint Plot")

import matplotlib.pyplot as plt 
import seaborn as sns 
 
plt.figure(figsize=(8, 8)) 
sns.jointplot( 
    x='duration_min',  
    y='popularity',  
    data=df,  
    kind='scatter',  # Use 'scatter' for the central plot type 
    height=8,        # Controls the overall size of the plot 
    marginal_kws={'bins': 30, 'color': 'gray', 'edgecolor': 'black'}, 
    joint_kws={'alpha': 0.6, 'color': 'darkred'} 
) 
 
plt.suptitle('Joint Distribution of Duration and Popularity', y=1.02, 
fontsize=14) 
plt.show() 



# Animated Line Chart
print("Animated Line Chart")

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import webbrowser

# --- Step 1: Load Dataset ---
csv_files = r"C:\Users\Kurt Russel\OneDrive\Documents\Lab7_DataVisualization_Belnas\spotify_top_1000_tracks.csv"

df = pd.read_csv(csv_files)

# --- Step 2: Data Preparation ---
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['year'] = df['release_date'].dt.year

df = df.dropna(subset=['year', 'popularity'])

yearly_popularity = df.groupby('year')['popularity'].mean().reset_index()
yearly_popularity = yearly_popularity.sort_values('year')

# --- Step 3: Initialize Figure ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlim(yearly_popularity['year'].min(), yearly_popularity['year'].max())
ax.set_ylim(0, yearly_popularity['popularity'].max() * 1.1)

line, = ax.plot([], [], color='royalblue', linewidth=2.5)

ax.set_title("Evolution of Track Popularity Over Time", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Average Popularity")

plt.tight_layout()

# --- Step 4: Animation Function ---
def animate(i):
    if i == 0:
        return line,

    x = yearly_popularity['year'].iloc[:i]
    y = yearly_popularity['popularity'].iloc[:i]

    line.set_data(x, y)
    return line,

# --- Step 5: Create Animation ---
ani = FuncAnimation(
    fig,
    animate,
    frames=len(yearly_popularity),
    interval=100,
    repeat=False
)

# --- Step 6: Save GIF ---
gif_path = os.path.abspath("yearly_popularity_trend.gif")
ani.save(gif_path, writer=PillowWriter(fps=10))

print(f"GIF saved successfully at: {gif_path}")

# --- Step 7: Open GIF ---
webbrowser.open(f"file://{gif_path}")

plt.close(fig)