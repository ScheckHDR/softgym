from numpy import cos,sin,arccos
import numpy as np
import matplotlib.pyplot as plt

def parametric_circle(t,xc,yc,R):
    x = xc + R*cos(t)
    y = yc + R*sin(t)
    return x,y

def inv_parametric_circle(x,xc,R):
    t = arccos((x-xc)/R)
    return t

def get_arc(theta_start,theta_end,sign):
    if sign:
        s = theta_start
        e = theta_end
    else:
        s = min(theta_start,theta_end)
        e = max(theta_start,theta_end) - 2*np.pi
    theta = np.linspace(s,e,100)

    return np.vstack([np.cos(theta),np.sin(theta)]).T


if __name__ == '__main__':

    for theta_s in range(0,360,10):
        for theta_e in range(0,360,10):
            s = theta_s /180 * np.pi
            e = theta_e / 180 * np.pi
            left = get_arc(s,e,True)
            right = get_arc(s,e,False)

            plt.clf()
            plt.plot(left[:,0],left[:,1],'b-')
            plt.plot(right[:,0],right[:,1],'r-')

            plt.draw()
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.pause(0.05)


