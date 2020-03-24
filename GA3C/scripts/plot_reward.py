import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

fig = plt.figure("roll reward", figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)


data = np.genfromtxt('/home/mfe/Downloads/run_network-tag-Roll_Reward (1).csv', delimiter=',', skip_header=10,
                     skip_footer=10, names=['wall time','step', 'value'])
data2 = np.genfromtxt('/home/mfe/Downloads/run_network-tag-Roll_Reward (2).csv', delimiter=',', skip_header=10,
                     skip_footer=10, names=['wall time','step', 'value'])


ind = np.argmin(abs(data2['step'] - data['step'][-1]))
steps_scale2 = 1.9e6 / data2['step'][-1]
scaled_steps2 = data2['step'][ind:]*steps_scale2
steps_scale1 = scaled_steps2[0] / data['step'][-1]
scaled_steps1 = data['step'] * steps_scale1

steps = np.hstack([scaled_steps1, scaled_steps2])
values = np.hstack([data['value'], data2['value'][ind:]])
# print data['step'][0:5]
# print data['step'][-5:]
# print data2['step'][0:5]

# Here is the label and arrow code of interest
# ax.annotate('SDL', xy=(0.5, 0.90), xytext=(0.8, 0.5), xycoords='axes fraction', \
#             fontsize=24, ha='center', va='bottom', \
#             bbox=dict(boxstyle='square', fc='white'), \
#             arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0))

ax.plot(steps, values, color='r')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.xlim([0,1e5])
plt.xlabel('Training Episode')
plt.ylabel('Rolling Reward')
plt.show()

# ax.plot(data['step'], data['value'], color='r', label='the data')
# ax.plot(data2['step'], data2['value'], color='b', label='the data')
# plt.show()


# import numpy as np

# import matplotlib.cbook as cbook

# fname = cbook.get_sample_data('/home/mfe/Downloads/run_network-tag-Roll_Reward.csv', asfileobj=False)
# # fname2 = cbook.get_sample_data('data_x_x2_x3.csv', asfileobj=False)

# # test 1; use ints
# plt.plotfile(fname, (0, 5, 6))


# import csv
# import matplotlib.pyplot as plt

# with open('/home/mfe/Downloads/run_network-tag-Roll_Reward.csv') as csvfile:
# 	reader = csv.DictReader(csvfile)
# 	for row in reader:
# 		steps = 
# 		print row.keys()




# # for row in reader:
# # print(row['first_name'], row['last_name'])

