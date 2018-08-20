import matplotlib.pyplot as plt

res = {
1 : [0.131000, 0.091000, 0.247000],
10 : [0.137000, 0.099000, 0.259000],
100 : [0.188000, 0.284000, 0.647000],
1000 : [0.402000, 1.932000, 2.962000],
10000 : [1.964000, 18.656000, 26.506000],
100000 : [28.782000, 185.939000, 474.313000],
1000000 : [293.686000, 1861.201000, 5106.972000],
10000000 : [7486.378000, 19141.029000, 68144.381000],
}

xnordot = [ (k, res[k][0]) for k in res ]
blasdot = [ (k, res[k][1]) for k in res ]
booldot = [ (k, res[k][2]) for k in res ]

plt.plot(*zip(*xnordot[3:]), label='xnordot')
plt.plot(*zip(*blasdot[3:]), label='blasdot')
plt.plot(*zip(*booldot[3:]), label='booldot')

plt.xlabel('Input size')
plt.ylabel('Time elapsed')

plt.xscale('log', basex=10)
#  plt.yscale('log', basey=2)

plt.legend()

plt.show()
