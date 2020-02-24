import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.stats as stats

# Set up network parameters
N = 2000
p = 0.1
g = 1.5        # g greater than 1 leads to chaoric networks
alpha = 1.0
nsecs = 1440
dt = 0.1
learn_every = 2
scale = 1.0/math.sqrt(p*N)

rvs = stats.norm(loc=0, scale=1).rvs
M = sparse.random(m=N, n=N, density=p, data_rvs=rvs)*g*scale
M = M.todense()

nRec2Out = int(N/2)
nRec2Control = int(N/2)

'''
Allow output and control units to start with different ICs.  If you set beta
greater than zero, then y will look different than z but still drive the
network with the appropriate frequency content (because it will be driven with
z).  A value of beta = 0 shows that the learning rules produce extremely
similar signals for both z(t) and y(t), despite having no common pre-synaptic
inputs.  Keep in mind that the vector norm of the output weights is 0.1-0.2
when finished, so if you make beta too big, things will eventually go crazy and
learning won't converge.
'''
# beta = 0.1

beta = 0.0
# Synaptic strengths from internal pool to output unit
wo = beta*np.random.normal(loc=0, scale=1, size=(nRec2Out, 1))/math.sqrt(N/2)
dwo = np.zeros((nRec2Out, 1))
wc = beta*np.random.normal(loc=0, scale=1, size=(nRec2Control, 1))/math.sqrt(N/2)
dwc = np.zeros((nRec2Control, 1))

# The feedback now comes from the control unit as opposed to the output
wf = 2.0 * (np.random.rand(N, 1) - 0.5)

# Deliberately set the presynaptic neurons to nonoverlapping between the output
# and the control units
zidxs = np.linspace(start=0, stop=round(N/2)-1, num=round(N/2), dtype=int)
yidxs = np.linspace(start=round(N/2), stop=N-1, num=round(N/2), dtype=int)

# Print summary
print("    N: {}".format(N))
print("    g: {}".format(g))
print("    p: {}".format(p))
print("    nRec2Out: {}".format(nRec2Out))
print("    nRec2Control: {}".format(nRec2Control))
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
wc_len = np.zeros(simtime_len)
zt = np.zeros(simtime_len)
yt = np.zeros(simtime_len)
zpt = np.zeros(simtime_len)
ypt = np.zeros(simtime_len)

x0 = 0.5*np.random.normal(loc=0, scale=1, size=(N, 1))
z0 = 0.5*np.random.normal(loc=0, scale=1, size=(1, 1))
y0 = 0.5*np.random.normal(loc=0, scale=1, size=(1, 1))

x = x0
r = np.tanh(x)
z = z0
y = y0

ti = 0
Pz = (1.0/alpha) * np.eye(nRec2Out)
Py = (1.0/alpha) * np.eye(nRec2Control)
rz = np.zeros((N//2))
ry = np.zeros((N//2))

for t in simtime:
    if ti % (nsecs/2) == 0:
        print("time: {}.".format(np.round(t, 3)))

    # sim, so x(t) and r(t) are created
    x = (1.0 - dt)*x + M*(r*dt) + wf*(y*dt)
    r = np.tanh(x)
    rz = r[zidxs]  # The neurons that project to the output
    ry = r[yidxs]  # The neurons that project to the control unit
    z = np.transpose(wo)*rz
    y = np.transpose(wc)*ry

    if ti % learn_every == 0:
        # Update inverse correlation matrix for the output unit
        kz = Pz * rz
        rPrz = np.transpose(rz) * kz
        cz = 1.0/(1.0 + rPrz)
        Pz = Pz - kz*(np.transpose(kz*cz))

        # Update the error for the linear readout
        e = z - ft[ti]
        # Update the output weights
        dwo = -kz * cz * e
        wo = wo + dwo

        # Update the inverse correlation matrix for the control unit
        ky = Py*ry
        rPry = np.transpose(ry) * ky
        cy = 1.0/(1.0 + rPry)
        Py = Py - ky * (np.transpose(ky * cy))

        # Update the output weights
        dwc = -ky * cy * e
        wc = wc + dwc
    # Store the output of the system
    zt[int(ti)] = z
    yt[int(ti)] = y
    wo_len[int(ti)] = np.sqrt(np.transpose(wo)*wo)
    wc_len[int(ti)] = np.sqrt(np.transpose(wc)*wc)
    ti += 1
# Calculate Mean absolute error
error_avg = np.sum(np.absolute(zt-ft))/simtime_len
print("Mean Absolute Error during training: {}.".format(np.round(error_avg, 5)))

# Plot training performance
fig, axs = plt.subplots(2, figsize=(14, 7))
fig.suptitle('Training')
plt.subplots_adjust(hspace=0.3)
axs[0].plot(simtime, ft, color='g', label="f")
axs[0].plot(simtime, zt, color='r', label="z")
axs[0].plot(simtime, yt, color='b', label="z")
axs[0].set_ylabel("f, z, and y")
axs[0].set_xlabel("Time")
axs[0].legend()

axs[1].plot(simtime, wo_len, 'b', label='outputweights')
axs[1].plot(simtime, wc_len, 'g', label='outputweights')
axs[1].set_ylabel("|w_o|, |w_c|")
axs[1].set_xlabel("Time")
axs[1].legend()

print("Now testing... please wait.")
# Now test

ti = 0
for t in simtime:
    # sim, so x(t) and r(t) are created
    # Not updating the weights wo, because we are no longer training
    x = (1.0 - dt)*x + M*(r*dt) + wf*(y*dt)  # Note the y here.
    r = np.tanh(x)
    rz = r[zidxs]
    ry = r[yidxs]
    z = np.transpose(wo)*rz
    y = np.transpose(wc)*ry
    zpt[ti] = z
    ypt[ti] = y
    ti += 1

error_avg = np.sum(np.absolute(zpt-ft2))/simtime_len
print("Mean Absolute Error during testing: {}.".format(np.round(error_avg, 5)))

# Plot testing performance
fig, axs = plt.subplots(2, figsize=(14, 7))
fig.suptitle('Testing')
plt.subplots_adjust(hspace=0.3)
axs[0].plot(simtime, ft, color='g', label="f", linewidth=2)
axs[0].plot(simtime, zt, color='r', label="z", linewidth=2)
axs[0].plot(simtime, yt, color='b', label="y", linewidth=2)
axs[0].set_ylabel("f, z, and t")
axs[0].set_xlabel("Time")
axs[0].legend()

axs[1].plot(simtime2, ft2, 'g', label='ft2', linewidth=2)
axs[1].plot(simtime2, zpt, 'r', label='zpt', linewidth=2)
axs[1].plot(simtime2, ypt, 'b', label='ypt', linewidth=2)
axs[1].set_ylabel("f, z, and y")
axs[1].set_xlabel("Time")
axs[1].legend()

plt.show()
