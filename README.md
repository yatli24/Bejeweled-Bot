# Bejeweled Bot
A bot that uses image recognition and a greedy algorithm to play the puzzle game Bejeweled.

Algorithm Priority:
1. Matches of 5 (L Shape)
2. Matches of 4
3. Vertical Matches (Theoretically better than horizontal matches)
4. Horizontal Matches
5. Matches of 3

# Notes
The code for the game board recognition region is hardcoded. Adjust according to your game version/setup.

The bot performs the best when color ranges are specifically defined.

A machine learning algorithm using scikit-learn was employed in version 1.0 with training and test sets using various game boards, but was found to be ineffective in recognizing certain gems.
