# 
# Baseline: CSS random choice model
#
import pandas as pd
import random

def randomizer(data):
    gambles = []
    for x in data.iterrows():
        gok = random.randint(1, 3)
        if gok == 1:
            gok = 'A'
        elif gok == 2:
            gok = 'B'
        elif gok == 3:
            gok = 'C'

        gambles.append(gok)
    return gambles

def multiple_checks(answers, data, iterations):
    avg = []
    
    for x in range(iterations):
        print("Processing iteration", x)
        result = randomizer(data)
        correct_counter = 0
        for gok, true in zip(result, answers):
            if gok == true:
                correct_counter += 1
        avg.append(correct_counter/ len(answers))
    return sum(avg) / len(avg)

if __name__ == "__main__":
    data = pd.read_csv("../common-sense/train_data.csv")
    answers = pd.read_csv("../common-sense/train_answers.csv")

    multiple_checks(answers['answer'].values, data, 100)