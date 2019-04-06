from blackjack import Game
import tensorflow as tf
import numpy as np
import pickle
import time

states = []
hits = []

def hardValue(hand):
    a,b = hand.values()
    if not a == b:
        return False,max(a,b)
    else:
        return True,a


def strat(game):
    deal = game.dealer.value()
    hard,value = hardValue(game.player)
    if hard:
        if value <= 11:
            hit = 1
        elif value == 12:
            if 4 <= deal <= 6:
                hit = 0
            else:
                hit = 1
        elif 13 <= value <= 16:
            if 2 <= deal <= 6:
                hit = 0
            else:
                hit = 1
        elif value >= 17:
            hit = 0
    else:
        if value <= 17:
            hit = 1
        elif value == 18:
            if deal in [9,10,11]:
                hit = 1
            else:
                hit = 0
        else:
            hit = 0
    return [hard, value, deal], hit

def simulateGames():
    game = Game()
    game.deal()
    game.dPlay()
    while True:
        #cardState = game.state()
        ########################################################
        """
        hard, value = hardValue(game.player)
        deal = game.dealer.value()
        if hard:
            if value <= 11:
                hit = 1
            elif value == 12:
                if 4 <= deal <= 6:
                    hit = 0
                else:
                    hit = 1
            elif 13 <= value <= 16:
                if 2 <= deal <= 6:
                    hit = 0
                else:
                    hit = 1
            elif value >= 17:
                hit = 0
        else:
            if value <= 17:
                hit = 1
            elif value == 18:
                if deal in [9,10,11]:
                    hit = 1
                else:
                    hit = 0
            else:
                hit = 0
        """
        state, hit = strat(game)
        states.append(state)
        hits.append(hit)
        ########################################################
        if game.player.bust():
            break
        game.pPlay()


def makeData(n):
    for i in range(n):
        simulateGames()
    data = [np.array(states), np.array(hits)]
    save = open("data{}.pkl".format(time.time()),'wb')
    pickle.dump(data,save)
    save.close()


def makeModel():
    file = open("data.pkl", 'rb')
    input = pickle.load(file)
    file.close()
    x_train,y_train = input

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1)
    model.save("blackjack{}.model".format(time.time()))


def main():
    #makeData(1000000)
    makeModel()


if __name__ == "__main__":
    main()
