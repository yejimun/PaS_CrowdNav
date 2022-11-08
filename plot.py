import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


legends = ['train', 'val']
out_dir = "data/pasrl/"


# change the folder directories here!
# for holonomic weight
logs1 = pd.read_csv(out_dir + "progress.csv", error_bad_lines=False)
logs2 = pd.read_csv(out_dir + "val_progress.csv", error_bad_lines=False)


logDicts={1:logs1, 2:logs2} 
graphDicts={0:'eprewmean', 1:'loss/value_loss', 2: 'loss/policy_loss', 3:'loss/PaS_loss'}

legendList=[]
# summarize history for accuracy

# for each metric
for i in range(len(graphDicts)):
	plt.figure(i)
	plt.title(graphDicts[i])
	j = 0
	for key in logDicts:
		if graphDicts[i] not in logDicts[key]:
			continue
		else:
			plt.plot(logDicts[key]['misc/total_timesteps'],logDicts[key][graphDicts[i]])

			legendList.append(legends[j])
			print('avg', str(key), graphDicts[i], np.average(logDicts[key][graphDicts[i]]))
		j = j + 1
	print('------------------------')

	plt.xlabel('total_timesteps')
	plt.legend(legendList, loc='lower right')
	plt.savefig(out_dir + "training_curve"+str(i))
	legendList=[]

plt.show()