def rk4(f, t, y, dt, *args):
    k1 = f(t, y, *args)
    k2 = f(t+dt/2, y+dt*k1/2, *args)
    k3 = f(t+dt/2, y+dt*k2/2, *args)
    k4 = f(t+dt, y+dt*k3, *args)
    y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y

