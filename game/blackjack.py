import random

class Card:
    def __init__(self, suite, rank):
        self.suite = suite
        self.rank = rank

    def __str__(self):
        rank = 0
        if self.rank == 1:
            rank = "Ace"
        elif self.rank <= 10:
            rank = str(self.rank)
        elif self.rank == 11:
            rank = "Jack"
        elif self.rank == 12:
            rank = "Queen"
        elif self.rank == 13:
            rank = "King"
        return rank + " of " + self.suite

class Deck:
    suites = ["Clubs", "Diamonds", "Hearts", "Spades"]
    ranks = [1,2,3,4,5,6,7,8,9,10,11,12,13] #1 is ace

    def __init__(self):
        self.cards = []
        for suite in self.suites:
            for rank in self.ranks:
                self.cards.append(Card(suite,rank))

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        if self.cards:
            return self.cards.pop()

    def __str__(self):
        cards = []
        for card in self.cards:
            cards.append(card.__str__() + ", ")
        return ''.join(cards)


class Hand:
    def __init__(self, deck):
        self.deck = deck
        self.cards = []

    def add(self, card):
        self.cards.append(card)

    def hit(self):
        self.add(self.deck.draw())

    def values(self):
        v1 = 0
        v2 = 0
        for card in self.cards:
          if card.rank == 1:
            v1 += 1
            v2 += 11
          elif card.rank in [11,12,13]:
            v1 += 10
            v2 += 10
          else:
            v1 += card.rank
            v2 += card.rank
        return (v1,v2)

    def value(self):
        values = self.values()
        if max(values) > 21:
            return min(values)
        else:
            return max(values)

    def bust(self):
        return self.value() > 21

    def hardValue(self):
        a,b = self.values()
        if not a == b:
            return 0,max(a,b)
        else:
            return 1,a

    def __str__(self):
        cards = []
        for card in self.cards:
            cards.append(card.__str__() + ", ")
        return ''.join(cards)


class Dealer(Hand):
    def play(self):
        while self.value() < 17:
            self.hit()


class Game:
    def __init__(self):
        self.deck = Deck()
        self.deck.shuffle()

        self.dealer = Dealer(self.deck)
        self.player = Hand(self.deck)

    def deal(self):
        self.dealer.hit()
        self.player.hit()
        self.player.hit()


    def state(self):
        hard, value = self.player.hardValue()
        state = [hard, value, self.dealer.value()]
        return state

    def pPlay(self):
        self.player.hit()

    def dPlay(self):
        self.dealer.play()

    #0 - player win, 1 - dealer win, 2 - tie
    def result(self):
        if not self.player.bust() and not self.dealer.bust():
            if self.player.value() > self.dealer.value():
                return 0
            elif self.player.value() < self.dealer.value():
                return 1
            else:
                return 2
        elif not self.player.bust():
            return 0
        elif not self.dealer.bust():
            return 1
        else:
            return 2


    def printResult(self):
        print("Player: ", self.player, "\nValue: ", self.player.value(), "\nBust: ", self.player.bust())
        print()
        print("Dealer: ", self.dealer, "\nValue: ", self.dealer.value(), "\nBust: ", self.dealer.bust())
        result = self.result()
        print()
        if result == 0:
            print("Player Wins")
        elif result == 1:
            print("Dealer Wins")
        else:
            print("Tie")

def main():
    game = Game()
    game.deal()
    game.pPlay()
    game.dPlay()
    game.printResult()


if __name__ == "__main__":
    main()
