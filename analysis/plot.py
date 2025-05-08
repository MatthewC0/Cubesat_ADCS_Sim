import matplotlib.pyplot as plt


def plot(data, time, datatype='none',show_plot=False):
    # Set up figure
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(data.shape[1]):
        ax1.plot(time, data[:, i])

    # Plot configuration
    plt.grid()
    if datatype == 'angle':
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.legend(['X', 'Y', 'Z'])

    if datatype == 'angular_velocity':
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity [deg/s]')
        plt.legend(['X', 'Y', 'Z'])

    if datatype == 'quaternion':
        plt.xlabel('Time [s]')
        plt.ylabel('Quaternion')
        plt.legend(['X', 'Y', 'Z', 'W'])

    if show_plot:
        plt.show()