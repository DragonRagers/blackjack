from blackjack import Game
import tensorflow as tf
import numpy as np

def printLine():
    print("----------------------------------------------------")

def hit(cardState):
    predictions = model.predict(np.array([cardState]))
    if np.argmax(predictions[0]) == 1:
        return True
    else:
        return False

def contGame():
    while True:
        game = Game()
        game.deal()

        printLine()
        game.printResult()
        printLine()
        while hit(game.state()):
            game.pPlay()
        game.dPlay()
        game.printResult()
        printLine()

        n = input("\n\nQuit?(y/n):")
        if n == "y":
            break

def finiteGames(n):
    games = [0,0,0]
    for i in range(n):
        game = Game()
        game.deal()

        while hit(game.state()):
            game.pPlay()
        game.dPlay()

        games[game.result()] += 1

    print(games)

model = tf.keras.models.load_model("blackjack.model")
finiteGames(10000)
