import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
print 'hello'
plt.plot([1,2,3,4])
plt.ylabel('some numbers')

plt.show(block=True)
