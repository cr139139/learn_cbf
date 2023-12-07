import matplotlib.pyplot as plt
plt.style.use('bmh')
from draw_tool import drawer_2d

filename = 'testing.npz'
drawer = drawer_2d(filename=filename)
drawer.show()