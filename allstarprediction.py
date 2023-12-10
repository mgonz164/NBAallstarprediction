# Mauricio Gonzalez
# Comp 541 Fall 2023 Professor Klotzman
# NBA All Star Prediction
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


df = pd.read_csv('all_seasons.csv')

# All these players are known all stars, here is where their name gets imported
known_all_stars = ['Stephen Curry', 'Kevin Durant', 'Fred VanVleet', 'Kemba Walker',
                   'Bradley Beal', 'Joakim Noah', 'Marc Gasol', 'Rudy Gobert',
                   'Jimmy Butler', 'Domantas Sabonis']

# The names imported earlier are then searched for within the all_seasons.csv file
# The player_name column in the file must match the String in the known_all_stars array
all_star_data = df[df['player_name'].isin(known_all_stars)].copy()
all_star_data['is_all_star'] = 1  # Label them as All-Stars

# This is where the names from the non-all-stars gets imported
other_players = ['Evan Mobley', 'Jalen Green', 'Cade Cunningham', 'Josh Giddey',
                 'Killian Hayes', 'Alperen Sengun', 'Deandre Ayton', 'Franz Wagner',
                 'Paolo Banchero', 'Keegan Murray', 'Jaden Ivey', 'Cam Thomas', 'Trey Murphy III']

other_players_data = df[df['player_name'].isin(other_players)].copy()
other_players_data['is_all_star'] = 0  # Label them as non-All-Stars

# Combine the datasets
combined_data = pd.concat([all_star_data, other_players_data])

# Features of data I want to use from the all_seasons.csv file
selected_columns = ['pts', 'reb', 'ast', 'net_rating', 'usg_pct', 'ts_pct', 'gp']
X = combined_data[selected_columns]
y = combined_data['is_all_star']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(random_state=42)


clf.fit(X_train, y_train)

# Make predictions on the entire dataset
all_players_pred = clf.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, all_players_pred)
conf_matrix = confusion_matrix(y, all_players_pred)
class_report = classification_report(y, all_players_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

print('For the Actual and Predicted Columns, a 1 means all-star status and 0 means non all-star status \n')
# Prints the name of the player next to their prediction
results = pd.DataFrame({'player_name': X.index.map(df['player_name']),
                        'Season': X.index.map(df['season']),
                        'Actual': y, 'Predicted': all_players_pred})
print(results)

# Output was too big so I needed to make .txt file for the whole output
output_file_path = 'allstarprediction.txt'

# Open the file and write the results
with open(output_file_path, 'w') as file:
    # Print the results to both the console and the file
    print(f'Accuracy: {accuracy:.2f}', file=file)
    print('Confusion Matrix:', file=file)
    print(conf_matrix, file=file)
    print('Classification Report:', file=file)
    print(class_report, file=file)
    print('\nFor the Actual and Predicted Columns, a 1 means all-star status and 0 means non all-star status\n',
          file=file)

    results = pd.DataFrame({'player_name': X.index.map(df['player_name']),
                            'Season': X.index.map(df['season']),
                            'Actual': y, 'Predicted': all_players_pred})
    results.to_string(file)


print(f'Output saved to: {output_file_path}')



#Plotting a visual rep. of the decision tree
plt.figure(figsize=(15, 10))

# Plot the decision tree
plot_tree(clf, filled=True, feature_names=selected_columns, class_names=['Not All-Star', 'All-Star'],
          rounded=True, fontsize=8)
plt.savefig('decision_tree_visualization.png')
plt.show()
