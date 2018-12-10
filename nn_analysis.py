from matplotlib import pyplot as plt
import numpy as np


plt.plot(x, y_gibbs_sampling_time, color='green', label='gibbs_sampling ')
plt.plot(x, y_likelihood_weighting_time, color='red', label='likelihood_weighting ')
plt.plot(x, y_rejection_sampling_time,  color='skyblue', label='rejection_sample ')
# plt.plot(x, [real[0]]*len(x),  color='blue', label=' Exact Inference')
plt.xlim((100,100000))
# plt.ylim((0, 0.5))
# plt.title('error between Approximate Inference and Exact Inference')
# plt.title(' Approximate Inference performance')
plt.title('run time')
# plt.plot(x, y_rejection, color='blue', label='my rejection')
plt.legend() # 显示图例
plt.xlabel('sample number')
plt.ylabel('time')
plt.show()
