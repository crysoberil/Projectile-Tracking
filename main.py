import csv
from projectile import Projectile
from projectiletracker import ProjectileTracker
from math import cos, pi, sin

# Reads the CSV file and returns a list of Projectile objects.
def read_projectile_trajectories(csv_file="Dataset.csv"):
    projectiles = []
    
    with open(csv_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        projectile = Projectile()
        first_iteration = True
        
        for row in reader:
            time, along_x, along_y = int(row[0]), float(row[1]), float(row[2])
            
            if not(first_iteration) and time == 0:
                projectiles.append(projectile)
                projectile = Projectile()
                projectile.add_trajectory_point(time, along_x, along_y)
                continue
            
            projectile.add_trajectory_point(time, along_x, along_y)
            first_iteration = False
        
        if projectile.trajectory_points_count() != 0:
            projectiles.append(projectile)
    
    return projectiles

# Returns the tuple (base_features, label_x, label_y).
# 'base_features': A feature matrix where each row is in the format [initial velocity along x, initial velocity along y, time].
#      One row for each reported trajectory points in the given dataset.
# 'label_x': A matrix where each row represents the position of the projectile along x axis corresponding to the 'base_features' row.
# 'label_y': A matrix where each row represents the position of the projectile along y axis corresponding to the 'base_features' row.
def build_training_sets(projectiles):
    base_features = []
    label_x = []
    label_y = []
    
    for projectile in projectiles:
        vx, vy = projectile.get_initial_velocity_vector()
        
        for time, along_x, along_y in projectile.trajectory:
            base_features.append([vx, vy, time])
            label_x.append([along_x])
            label_y.append([along_y])
    
    return base_features, label_x, label_y
            



projectiles = read_projectile_trajectories()
base_features, label_x, label_y = build_training_sets(projectiles)

projectile_tracker = ProjectileTracker()
# Build a model on how a projectile behaves.
projectile_tracker.fit(base_features, label_x, label_y)

# Create the test projectile.
test_projectile = Projectile()
test_thorow_vx = 1.0 * cos(pi / 4.0)
test_thorow_vy = 1.0 * sin(pi / 4.0)

# print "Test Projectile(thrown in 45 degree angle at velocity 10 m/s):"
for i in xrange(101):
    base_feat = [[test_thorow_vx, test_thorow_vy, i]]
    prediction = projectile_tracker.predict(base_feat)
    pred_x = prediction[0][0]
    pred_y = prediction[0][1]
    
    if i != 0 and pred_y < 0:
        break
    
    test_projectile.add_trajectory_point(i, pred_x, pred_y)
    print ', '.join([str(elm) for elm in [i, pred_x, pred_y]])

# test_projectile.plot_trajectory()
