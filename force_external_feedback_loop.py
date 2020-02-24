import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.stats as stats

# Set up network parameters
N = 1000
p = 0.1
g = 1.5        # g greater than 1 leads to chaoric networks
alpha = 1.0
nsecs = 1440
dt = 0.1
learn_every = 2
scale = 1.0/math.sqrt(p*N)

# Create a sparse randomly distributed network
rvs = stats.norm(loc=0, scale=1).rvs
M = sparse.random(m=N, n=N, density=p, data_rvs=rvs)*g*scale
M = M.todense()

nRec2Out = N
wo = np.zeros((nRec2Out, 1))
dw = np.zeros((nRec2Out, 1))
wf = 2.0 * (np.random.rand(N, 1) - 0.5)  # List of random numbers from -1 to 1

# Print summary
print("    N: {}".format(N))
print("    g: {}".format(g))
print("    p: {}".format(p))
print("    nRec2Out: {}".format(nRec2Out))
print("    alpha: {}".format(alpha))
print("    nsecs: {}".format(nsecs))
print("    learn_every: {}".format(learn_every))

simtime = np.linspace(start=0, stop=nsecs-dt, num=nsecs/dt)
simtime_len = len(simtime)
simtime2 = np.linspace(start=nsecs, stop=2*nsecs, num=nsecs/dt)

# Make sine waves
amp = 1.3
freq = 1/60
ft = (amp/1.0)*np.sin(1.0*(math.pi)*freq*simtime) + \
     (amp/2.0)*np.sin(2.0*(math.pi)*freq*simtime) + \
     (amp/6.0)*np.sin(3.0*(math.pi)*freq*simtime) + \
     (amp/3.0)*np.sin(4.0*(math.pi)*freq*simtime)
ft = ft/1.5

ft2 = (amp/1.0)*np.sin(1.0*(math.pi)*freq*simtime2) + \
      (amp/2.0)*np.sin(2.0*(math.pi)*freq*simtime2) + \
      (amp/6.0)*np.sin(3.0*(math.pi)*freq*simtime2) + \
      (amp/3.0)*np.sin(4.0*(math.pi)*freq*simtime2)
ft2 = ft2/1.5

wo_len = np.zeros(simtime_len)
zt = np.zeros(simtime_len)
zpt = np.zeros(simtime_len)

x0 = 0.5*np.random.normal(loc=0, scale=1, size=(N, 1))
z0 = 0.5*np.random.normal(loc=0, scale=1, size=(1, 1))

x = x0
r = np.tanh(x)
z = z0

ti = 0
P = (1.0/alpha) * np.eye(nRec2Out)

for t in simtime:

    if ti % (nsecs/2) == 0:
        print("time: {}.".format(np.round(t, 3)))
        # If realtime plotting is desired write it here.

    # sim, so x(t) and r(t) are created
    x = (1.0 - dt)*x + M*(r*dt) + wf*(z*dt)
    r = np.tanh(x)
    z = np.transpose(wo)*r
    if ti % learn_every == 0:
        # Update inverse correlation matrix
        k = P*r
        rPr = np.transpose(r) * k
        c = 1.0/(1.0 + rPr)
        P = P - k*(np.transpose(k*c))

        # Update the error for the linear readout
        e = z - ft[ti]
        # Update the output weights
        dw = -k * c * e
        wo = wo + dw
    # Store the output of the system
    zt[int(ti)] = z
    wo_len[int(ti)] = np.sqrt(np.transpose(wo)*wo)
    ti += 1            # Increment time step
# Calculate Mean absolute error
error_avg = np.sum(np.absolute(zt-ft))/simtime_len
print("Mean Absolute Error during training: {}.".format(np.round(error_avg,5)))

# Plot training performance
fig, axs = plt.subplots(2, figsize=(14,7))
fig.suptitle('Training')
plt.subplots_adjust(hspace=0.3)
axs[0].set_ylabel("f and z")
axs[0].set_xlabel("Time")
axs[0].plot(simtime, ft, color='r', label="f")
axs[0].plot(simtime, zt,color='b', label="z")
axs[0].legend()
axs[1].set_ylabel("|w|")
axs[1].set_xlabel("Time")
axs[1].plot(simtime, wo_len, 'o', label='outputweights')


# Time to test
print("Now testing... please wait.")

ti = 0
for t in simtime:
    # sim, so x(t) and r(t) are created
    # Not updating the weights wo, because we are no longer training
    x = (1.0 - dt)*x + M*(r*dt) + wf*(z*dt)
    r = np.tanh(x)
    z = np.transpose(wo)*r

    zpt[ti] = z
    ti += 1
error_avg = np.sum(np.absolute(zpt-ft2))/simtime_len
print("Mean Absolute Error during testing: {}.".format(np.round(error_avg,5)))

# Plot testing performance
fig, axs = plt.subplots(2, figsize=(14,7))
fig.suptitle('Testing')
plt.subplots_adjust(hspace=0.3)
axs[0].set_ylabel("f and z")
axs[0].set_xlabel("Time")
axs[0].plot(simtime, ft, color='r', label="f", linewidth=2)
axs[0].plot(simtime, zt,color='b', label="z", linewidth=2)
axs[0].legend()

axs[1].set_ylabel("f and z")
axs[1].set_xlabel("Time")
axs[1].plot(simtime2, ft2, 'o', label='ft2', linewidth=2)
axs[1].plot(simtime2, zpt, 'o', label='zpt', linewidth=2)
axs[1].legend()

# Show plots
plt.show()
