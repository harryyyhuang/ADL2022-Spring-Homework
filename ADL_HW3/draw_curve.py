import json
import numpy as np
import matplotlib.pyplot as plt

rouge1 = np.zeros(7)

rouge2 = np.zeros(7)
rougel = np.zeros(7)

for i in range(20):
    if(i %3 == 0):
        f = open(f"summary_beam3/all_results_{i}.json")

        data = json.load(f)

        rouge1[i//3] = data["eval_rouge1"]
        rouge2[i//3] = data["eval_rouge2"] 
        rougel[i//3] = data["eval_rougeL"]

        f.close()

print(rouge1)
epoches = range(0, 20, 3)


plt.plot(epoches, rouge1, "r", label="rouge-1")
plt.plot(epoches, rouge2, "g", label="rouge-2")
plt.plot(epoches, rougel, "b", label="rouge-l")

plt.title(f'Learning Curve')
plt.xlabel('Epochs')
plt.ylabel(f'Rouge Score')
plt.legend()
plt.savefig(f'LearningCurve.png')

