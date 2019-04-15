from blackjack import Game
import tensorflow as tf
import numpy as np

def printLine():
    print("----------------------------------------------------")

def hit(state, model):
    predictions = model.predict(np.array([state]))
    if np.argmax(predictions[0]) == 1:
        return True
    else:
        return False

# TODO: IMPLEMENT tf.predict() CORRECTLY TO DO IT IN PARALLEL
def simulateGames(n, model):
    games = [0,0,0]
    for i in range(n):
        game = Game()
        game.deal()

        while hit(game.state(), model):
            game.pPlay()
            if game.player.bust():
                break
        game.dPlay()

        games[game.result()] += 1
    print(games)


def contGame(model):
    while True:
        game = Game()
        game.deal()

        printLine()
        game.printResult()
        printLine()
        while hit(game.state(), model):
            game.pPlay()
        game.dPlay()
        game.printResult()
        printLine()

        n = input("\n\nQuit?(y/n):")
        if n == "y":
            break


def main():
    model = tf.keras.models.load_model("blackjack.model")
    simulateGames(10000, model)
    #contGame(model)

if __name__ == "__main__":
    main()
