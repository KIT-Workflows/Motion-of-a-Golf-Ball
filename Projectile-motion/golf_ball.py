import numpy as np
import sys, os, yaml, csv
from scipy.integrate import solve_ivp
from scipy import interpolate
from pylab import*
import matplotlib.pyplot as plt


def cdcl(v, nrpm):
    """Gives the value of the drag and lift coefficients as function of velocity and spin.
    Is only valid for velocities between 13.7 and 88.1 m/s, and spins between 2000 and 6000 rpm.
    The value is determined with linear interpolation of the data given by Bearman and Harvey, 
    Golf Ball Aerodynamics, volume 27, Aeronautival Quarterly, 1976."""

    if (v<13.7 or v>88.1):
        sys.exit('v is out of bounds. Must be between 13.7 and 88.41 m/s.')
        return

    if (nrpm<2000 or nrpm>6000):
        sys.exit('nrpm is out of bounds. Must be between 2000 and 6000 rpm.')
        return

    v0 = array([13.7, 21.6, 29.9, 38.4, 46.9, 55.2, 63.1, 71.9, 80.2, 88.1])
    nrpm0 = linspace(2000,6000,21)

    CD = array([[0.3624,0.2885,0.2765,0.2529,0.2472,0.2481,0.2467,0.2470,0.2470,0.2470],[0.3806,0.3102,0.2853,0.2590,0.2507,0.2498,0.2485,0.2486,0.2484,0.2484],[0.3954,0.3288,0.2937,0.2649,0.2543,0.2516,0.2504,0.2502,0.2497,0.2497],[0.4070,0.3443,0.3018,0.2708,0.2580,0.2535,0.2522,0.2518,0.2511,0.2511],[0.4153,0.3566,0.3095,0.2765,0.2617,0.2556,0.2541,0.2534,0.2524,0.2524],[0.4203,0.3658,0.3169,0.2822,0.2655,0.2578,0.2560,0.2550,0.2538,0.2538],[0.4120,0.3719,0.3240,0.2876,0.2693,0.2602,0.2579,0.2566,0.2551,0.2551],[0.3960,0.3749,0.3308,0.2930,0.2732,0.2627,0.2599,0.2582,0.2565,0.2565],[0.3876,0.3766,0.3372,0.2983,0.2772,0.2653,0.2619,0.2598,0.2578,0.2578],[0.4100,0.3854,0.3433,0.3034,0.2811,0.2681,0.2639,0.2614,0.2592,0.2592],[0.4288,0.4003,0.3490,0.3084,0.2852,0.2710,0.2659,0.2630,0.2605,0.2605],[0.4445,0.4082,0.3544,0.3133,0.2893,0.2741,0.2680,0.2646,0.2619,0.2619],[0.4575,0.4153,0.3595,0.3180,0.2934,0.2772,0.2701,0.2662,0.2632,0.2632],[0.4682,0.4215,0.3643,0.3227,0.2976,0.2806,0.2722,0.2678,0.2646,0.2646],[0.4772,0.4269,0.3687,0.3272,0.3019,0.2840,0.2743,0.2694,0.2659,0.2659],[0.4848,0.4314,0.3728,0.3316,0.3062,0.2876,0.2765,0.2710,0.2673,0.2673],[0.4914,0.4350,0.3765,0.3358,0.3105,0.2913,0.2787,0.2726,0.2686,0.2686],[0.4976,0.4377,0.3799,0.3400,0.3149,0.2952,0.2809,0.2742,0.2700,0.2700],[0.5039,0.4395,0.3830,0.3440,0.3194,0.2992,0.2831,0.2758,0.2713,0.2713],[0.5105,0.4405,0.3858,0.3479,0.3239,0.3034,0.2854,0.2774,0.2727,0.2727],[0.5180,0.4406,0.3882,0.3517,0.3285,0.3076,0.2877,0.2790,0.2740,0.2740]])
    CL = array([[0.1040,0.1846,0.2460,0.1984,0.1762,0.1538,0.1418,0.1360,0.1280,0.1276],[0.1936,0.2318,0.2590,0.2089,0.1824,0.1603,0.1476,0.1405,0.1321,0.1324],[0.2608,0.2694,0.2715,0.2191,0.1886,0.1668,0.1533,0.1450,0.1362,0.1367],[0.3090,0.2986,0.2835,0.2289,0.1949,0.1731,0.1589,0.1494,0.1403,0.1404],[0.3418,0.3205,0.2947,0.2384,0.2011,0.1794,0.1646,0.1539,0.1444,0.1436],[0.3624,0.3362,0.3049,0.2475,0.2073,0.1856,0.1702,0.1584,0.1485,0.1464],[0.3743,0.3470,0.3140,0.2562,0.2135,0.1916,0.1757,0.1629,0.1526,0.1488],[0.3808,0.3541,0.3217,0.2644,0.2197,0.1976,0.1813,0.1673,0.1567,0.1508],[0.3854,0.3584,0.3280,0.2722,0.2259,0.2035,0.1868,0.1718,0.1608,0.1524],[0.3915,0.3614,0.3325,0.2795,0.2322,0.2092,0.1923,0.1763,0.1649,0.1539],[0.4005,0.3640,0.3347,0.2862,0.2384,0.2149,0.1977,0.1807,0.1690,0.1582],[0.4010,0.3696,0.3380,0.2925,0.2446,0.2205,0.2032,0.1852,0.1731,0.1624],[0.4026,0.3748,0.3412,0.2982,0.2508,0.2259,0.2086,0.1897,0.1772,0.1666],[0.4050,0.3797,0.3440,0.3033,0.2570,0.2313,0.2139,0.1941,0.1813,0.1707],[0.4084,0.3845,0.3468,0.3078,0.2632,0.2366,0.2193,0.1986,0.1854,0.1749],[0.4130,0.3894,0.3496,0.3119,0.2695,0.2418,0.2246,0.2031,0.1895,0.1791],[0.4192,0.3947,0.3527,0.3149,0.2757,0.2469,0.2298,0.2076,0.1936,0.1833],[0.4271,0.4004,0.3563,0.3184,0.2819,0.2518,0.2351,0.2120,0.1977,0.1875],[0.4371,0.4069,0.3607,0.3233,0.2881,0.2567,0.2403,0.2165,0.2018,0.1916],[0.4494,0.4143,0.3660,0.3300,0.2943,0.2615,0.2455,0.2210,0.2059,0.1958],[0.4644,0.4227,0.3724,0.3393,0.3005,0.2662,0.2507,0.2254,0.2100,0.2000]])
    cd = interpolate.interp2d(v0, nrpm0, CD, kind='linear')
    cl = interpolate.interp2d(v0, nrpm0, CL, kind='linear')
    return cd(v, nrpm), cl(v, nrpm)


# Define the function cd_sphere
def cd_sphere(Re):
    "This function computes the drag coefficient of a sphere as a function of the Reynolds number Re."
    # Curve fitted after fig . A -56 in Evett and Liu: "Fluid Mechanics and Hydraulics"
#    from numpy import log10, array, polyval
    if Re <= 0.0:
        CD = 0.0
    elif Re > 8.0e6:
        CD = 0.2
    elif Re > 0.0 and Re <= 0.5:
        CD = 24.0/Re
    elif Re > 0.5 and Re <= 100.0:
        p = np.array([4.22, -14.05, 34.87, 0.658])
        CD = polyval(p, 1.0/Re)
    elif Re > 100.0 and Re <= 1.0e4:
        p = np.array([-30.41, 43.72, -17.08, 2.41])
        CD = polyval(p, 1.0/log10(Re))
    elif Re > 1.0e4 and Re <= 3.35e5:
        p = np.array([-0.1584, 2.031, -8.472, 11.932])
        CD = polyval(p, log10(Re))
    elif Re > 3.35e5 and Re <= 5.0e5:
        x1 = log10(Re/4.5e5)
        CD = 91.08*x1**4 + 0.0764
    else:
        p = np.array([-0.06338, 1.1905, -7.332, 14.93])
        CD = polyval(p, log10(Re))
    return CD


# smooth ball no drag
def f0(t, u):
    x, xdot, y, ydot = u
    speed = np.hypot(xdot, ydot)
    xdot = xdot - vfx
    ydot = ydot - vfy
    xdotdot = 0.0
    ydotdot = -g

    return xdot, xdotdot, ydot, ydotdot

# def deriv_add(t, u): 
#     x, xdot, y, ydot = u
#     speed = np.hypot(xdot, ydot)
#     xdotdot = -k/m * speed * xdot
#     ydotdot = -k/m * speed * ydot - g
#     return xdot, xdotdot, ydot, ydotdot

# smooth ball + drag
def f(t, u):
    """4x4 system for smooth sphere with drag in two directions."""
    x, xdot, y, ydot = u
    C = 3.0*rho_f/(4.0*rho_s*d)
    xdot = xdot - vfx
    ydot = ydot - vfy
    vr = np.sqrt(xdot**2 + ydot**2)
    Re = vr*d/nu
    CD = cd_sphere(Re) # using the already defined function
    xdotdot = -C*vr*(CD*xdot)
    ydotdot = C*vr*(-CD*ydot) - g

    return xdot, xdotdot, ydot, ydotdot

#  golf ball with drag and without lift
def f2(t, u):
    """4x4 system for golf ball with drag in two directions."""
    x, xdot, y, ydot = u
    C = 3.0*rho_f/(4.0*rho_s*d)
    xdot = xdot - vfx
    ydot = ydot - vfy
    vr = np.sqrt(xdot**2 + ydot**2)
    Re = vr*d/nu
    CD, CL = cdcl(vr, nrpm)
    xdotdot = -C*vr*(CD*xdot)
    ydotdot = C*vr*(-CD*ydot) - g

    return xdot, xdotdot, ydot, ydotdot

# golf ball with drag + lift
def f3(t,u):
    """4x4 system for golf ball with drag and lift in two directions."""
    x, xdot, y, ydot = u
    C = 3.0*rho_f/(4.0*rho_s*d)
    xdot = xdot - vfx
    ydot = ydot - vfy
    vr = np.sqrt(xdot**2 + ydot**2)
    Re = vr*d/nu
    CD, CL = cdcl(vr, nrpm)
    xdotdot = -C*vr*(CD*xdot + CL*ydot)
    ydotdot = C*vr*(CL*xdot - CD*ydot) - g
    
    return xdot, xdotdot, ydot, ydotdot


def hit_target(t, u):
    # We've hit the target if the y-coordinate is 0.
    return u[2]

def max_height(t, u):
    # The maximum height is obtained when the y-velocity is zero.
    return u[3]

def round_float_list(float_list, decimal_points):
    float_list = [round(float(item),decimal_points) for item in float_list]
    return float_list

if __name__ == '__main__':
    
    with open('rendered_wano.yml') as file:
        wano_file = yaml.full_load(file)
    
    
    # Input data
    g = 9.81      # Gravity [m/s^2]
    nu = 1.5e-5   # Kinematical viscosity [m^2/s]
    rho_f = 1.225  # Density of fluid [kg/m^3]
    rho_s = 1275  # Density of sphere [kg/m^3]
    #d = 41.0e-3   # Diameter of the sphere [m]
    nrpm = 3500   # no of rpm of golf ball
    # v0 = 50.0     # Initial velocity [m/s]
    

    no_plots = len(wano_file["Parameters"])
    PROJOUT = {}  # output file
    decimal_points = 6 # decimal points 

    system = []
    x0_data = []
    y0_data = []
    v0_data = []
    m_data = []
    phi0_data = []
    r_data = []
    vf_x = []
    vf_y = []
    labels = []

    for i in range(no_plots):
        system.append(wano_file["Parameters"][i]['System'])
        x0_data.append(wano_file["Parameters"][i]['x0 (m)'])
        y0_data.append(wano_file["Parameters"][i]['y0 (m)'])
        v0_data.append(wano_file["Parameters"][i]['v0 (m/s)'])
        m_data.append(wano_file["Parameters"][i]['Mass (kg)'])
        phi0_data.append(wano_file["Parameters"][i]['Angle (°)'])
        r_data.append(wano_file["Parameters"][i]['Radius (m)'])
        labels.append(wano_file["Parameters"][i]['label'])
        if "vf-x" in wano_file["Parameters"][0]:
            vf_x.append(wano_file["Parameters"][i]["vf-x"])
            print('Hi')
            
        if "vf-y" in wano_file["Parameters"][0]:
            vf_y.append(wano_file["Parameters"][i]["vf-y"])
            print('Hi')
            
    # Output_data
    Sol_dic = {}
    ii_list = []
    xmax_list = []
    ymax_list = []
    time_target = []
    time_highest = []
    Full_sol = []

    
    for ii in range(no_plots):
        if len(vf_x) == 0:
            vfx = 0.0  # x-component of fluid's velocity
            vfy = 0.0  # y-component of fluid's velocity
        else:     
            vfx = vf_x[ii]  # x-component of fluid's velocity
            vfy = vf_y[ii]  # y-component of fluid's velocity

        system_var = system[ii] # drag force True or False
        x0 = x0_data[ii]
        y0 = y0_data[ii]
        v0 = v0_data[ii] # 50 m/s
        m = m_data[ii]
        # Initial speed and launch angle (from the horizontal).
        phi0 = np.radians(phi0_data[ii])
        r = r_data[ii] #0.9
        d = 2.0*r_data[ii]
        # Initial conditions: x0, v0_x, y0, v0_y.
        u0 = x0, v0 * np.cos(phi0), y0, v0 * np.sin(phi0) 
        
        # Integrate up to tf unless we hit the target sooner.
        t0, tf = 0, 80

        # Stop the integration when we hit the target.
        hit_target.terminal = True
        # We must be moving downwards (don't stop before we begin moving upwards!)
        hit_target.direction = -1
        
        if system_var == "smooth ball + drag":
            soln = solve_ivp(f, (t0, tf), u0, dense_output=True, events=(hit_target, max_height))
            print('1')
        elif system_var == "golf ball + drag":    
            soln = solve_ivp(f2, (t0, tf), u0, dense_output=True, events=(hit_target, max_height))
            print('2')
        elif system_var == "golf ball + drag + lift":
            print('3')
            soln = solve_ivp(f3, (t0, tf), u0, dense_output=True, events=(hit_target, max_height))
        else:
            print('4') # smooth ball
            soln = solve_ivp(f0, (t0, tf), u0, dense_output=True, events=(hit_target, max_height))
            
        # A fine grid of time points from 0 until impact time.
        t = np.linspace(0, soln.t_events[0][0], 100)

        # Retrieve the solution for the time grid and plot the trajectory.
        sol = soln.sol(t)
        x = round_float_list(sol[0],decimal_points) 
        y = round_float_list(sol[2], decimal_points)
        vx = round_float_list(sol[1], decimal_points)
        vy = round_float_list(sol[3], decimal_points)
        
        Sol_dic["x"+str(ii)] = x
        Sol_dic["y"+str(ii)] = y
        Sol_dic["vx"+str(ii)] = vx
        Sol_dic["vy"+str(ii)] = vy

        ii_list.append(ii)
        xmax_list.append(float(round(x[-1],decimal_points)))
        ymax_list.append(float(round(max(y),decimal_points)))
        time_target.append(float(round(soln.t_events[0][0],decimal_points)))
        time_highest.append(float(round(soln.t_events[1][0],decimal_points)))

    PROJOUT["Step ii"] = ii_list
    PROJOUT["xmax"] = xmax_list
    PROJOUT["ymax"] = ymax_list
    PROJOUT["time to target"] = time_target
    PROJOUT["time to highest point"] = time_highest
    
    try:   
        with open("PROJOUT.yml",'w') as out:
            yaml.dump(PROJOUT, out,default_flow_style=False)
    
        with open("PROJDATA.yml",'w') as out:
            yaml.dump(Sol_dic, out,default_flow_style=False)
    except IOError:
        print("I/O error")
    