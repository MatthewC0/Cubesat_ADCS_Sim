def euler(f, t, y, dt, *args):
    ydot = f(t, y, *args)
    y = y + ydot*dt
    return y
