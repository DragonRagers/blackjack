from blackjack import Game
import tensorflow as tf
import numpy as np
import pickle
import time

states = []
hits = []

def simulateGames():
    game = Game()
    game.deal()
    cardState = game.state()
    while True:
        """
        cardState = [0]*12
        #cardState = np.array([0]*12)
        for i, card in enumerate(gameState[0]):
            cardState[i] = card.rank
        cardState[11] = gameState[1][0].rank
        """

        game.pPlay()
        if game.player.bust():
            hit = 0
        else:
            hit = 1

        states.append(cardState)
        hits.append(hit)

        if game.player.bust():
            break

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
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save("blackjack.model")

def testModel():
    model = tf.keras.models.load_model("blackjack.model")

    game = Game()
    game.deal()
    gameState = game.state()

    cardState = [0]*12
    #cardState = np.array([0]*12)
    for i, card in enumerate(gameState[0]):
        cardState[i] = card.rank
    cardState[11] = gameState[1][0].rank

    predictions = model.predict(np.array([cardState]))
    print(cardState)
    print(np.argmax(predictions[0]))

def main():
    #makeData(10000)
    makeModel()
    #testModle()


if __name__ == "__main__":
    main()
