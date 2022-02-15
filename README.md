# Board game assistant

**Board game assistant** - your personal helper in board games, presented as a telegram bot.

## Main functionality (depends on game type):

* determining the type of board game
* analysis of the game  
* best move offer  
* scoring  
* determining the winner

## Installation

* ```git clone https://github.com/BoardGameAssistant/BoardGameAssistant.git```
* ```cd board-game-assistant/```
* Enter your bot token in the **config.json** file
* ```docker build -t board_game_assistant .```
* ```docker run --name boardGameAssistant --rm -itd -v CACHE_PATH:/output board_game_assistant  ```

## Example
For analysis, just send a photo to the bot **https://t.me/BoardGameAssistantBot**