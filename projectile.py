import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

class Projectile:
    
    def __init__(self):
        self.trajectory = []
        
    def add_trajectory_point(self, time, along_x, along_y):
        self.trajectory.append((time, along_x, along_y))
        
    def number_of_points(self):
        return len(self.trajectory)
    
    def get_initial_velocity_vector(self):
        t = np.array([np.array([time, time * time]) for (time, _, _) in self.trajectory])
        x = np.array([np.array([along_x]) for (_, along_x, _) in self.trajectory])
        y = np.array([np.array([along_y]) for (_, _, along_y) in self.trajectory])
        
        regr_x = linear_model.LinearRegression(normalize=True)
        regr_x.fit(t, x)
        coeff_x = regr_x.coef_
        vx = coeff_x[0][0]
        
        regr_y = linear_model.LinearRegression(normalize=True)
        regr_y.fit(t, y)
        coeff_y = regr_y.coef_
        vy = coeff_y[0][0]
        
        return vx, vy
        
    
    def plot_graph(self):
        x = [along_x for (_, along_x, _) in self.trajectory]
        y = [along_y for (_, _, along_y) in self.trajectory]
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        figure_fraction_of_image = .75;
        x_extend = (1.0 / figure_fraction_of_image - 1) / 2.0 * (max_x - min_x)
        y_extend = (1.0 / figure_fraction_of_image - 1) / 2.0 * (max_y - min_y)
        plt.plot(x, y, '-ro')
        plt.axis([min_x - x_extend, max_x + x_extend, min_y - y_extend, max_y + y_extend])
        plt.show()