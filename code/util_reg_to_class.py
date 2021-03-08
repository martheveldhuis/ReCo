import numpy as np

def reg_to_class(model, x):
    result = []
    outcomes = model.get_prediction(x)
    results = np.empty((len(outcomes),5), dtype=object)
    i = 0
    for out in outcomes:
        prob_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        round_y = int(out.round())
        second_y = 0
        if round_y > out:
            second_y = int(round_y - 1)
        else:
            second_y = int(round_y + 1)
        prob_1 = 1 - abs(out-round_y)
        prob_2 = 1 - prob_1
        prob_array[round_y-1] = prob_1
        if second_y >= 1 and second_y <= 5:
            prob_array[second_y-1] = prob_2
        results[i] = prob_array
        i+=1
    return results