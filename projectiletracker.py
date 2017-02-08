import numpy.random as rand
import numpy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# A linear regression model that learns to map the location of projectiles in x and y axis
# using features vector [throwing velocity x, throwing velocity y, time].
class ProjectileTracker:
    def __init__(self):
        self.model_x = None
        self.model_y = None
        self.x_projection_dimension = -1;
        self.y_projection_dimension = -1;
    
    # Returns a 4-tuple train_X, train_y, cv_X, cv_y, the dataset split in training and test sets according to the train fraction.
    # Note, the each element is randomly picked to the train or cross validation dataset. Hence after split the exact train_fraction may not hold.
    def _split_dateset_train_cv(self, X, y, train_fraction=0.9):
        train_X = []
        train_y = []
        cv_X = []
        cv_y = []
        
        for i in xrange(len(y)):
            if rand.uniform(0, 1) <= train_fraction:
                train_X.append(X[i])
                train_y.append(y[i])
            else:
                cv_X.append(X[i])
                cv_y.append(y[i])
        
        return train_X, train_y, cv_X, cv_y
    
    
    # Maps the given feature vector X to a higher dimension 'dimension'. This allows non-linear regression.
    def _map_to_dimension(self, X, dimension):
        npX = numpy.array(X)
        poly = PolynomialFeatures(dimension, include_bias=False)
        mapped = poly.fit_transform(npX)
        return mapped
    
    
    # Using a hold out cross validation, finds the best dimension the feature vector should be mapped to.
    def _get_best_polynomial_degree(self, X, y, min_degree=1, max_degree=3):
        train_X, train_y, cv_X, cv_y = self._split_dateset_train_cv(X, y)
        cverror_polypower_pairs = []
        
        for i in xrange(min_degree, max_degree + 1):
            train_X_mapped = self._map_to_dimension(train_X, i)
            cv_X_mapped = self._map_to_dimension(cv_X, i)
            
            regr = linear_model.LinearRegression(normalize=True)
            regr.fit(train_X_mapped, train_y)
            
            predicted_cv = regr.predict(cv_X_mapped)
            mean_sqr_err = mean_squared_error(cv_y, predicted_cv)
            cverror_polypower_pairs.append((mean_sqr_err, i))
        
        _, best_power = min(cverror_polypower_pairs)
        return best_power
        
            
    
    # Fits the linear regression model.
    def fit(self, base_features, label_x, label_y):
        self.x_projection_dimension = self._get_best_polynomial_degree(base_features, label_x, 1, 2)
        feature_mapped_x = self._map_to_dimension(base_features, self.x_projection_dimension)
        self.model_x = linear_model.LinearRegression(normalize=True)
        self.model_x.fit(feature_mapped_x, label_x)
        
        self.y_projection_dimension = self._get_best_polynomial_degree(base_features, label_y, 1, 2)
        feature_mapped_y = self._map_to_dimension(base_features, self.y_projection_dimension)
        self.model_y = linear_model.LinearRegression(normalize=True)
        self.model_y.fit(feature_mapped_y, label_y)
    
    # Given a feature matrix X, returns a numpy array where each row is in format [x, y], the location of the projectile for the given feature vector.
    def predict(self, X):
        feature_mapped_for_x = self._map_to_dimension(X, self.x_projection_dimension)
        feature_mapped_for_y = feature_mapped_for_x if (self.x_projection_dimension == self.y_projection_dimension) else self._map_to_dimension(X, self.x_projection_dimension)
        
        predicted_x = self.model_x.predict(feature_mapped_for_x)
        predicted_y = self.model_y.predict(feature_mapped_for_y)
        
        return numpy.array([[predicted_x[i][0], predicted_y[i][0]] for i in xrange(len(predicted_x))])
    
    
