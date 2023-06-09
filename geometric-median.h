#ifndef GEOMETRIC_MEDIAN_H
#define GEOMETRIC_MEDIAN_H

#include <Eigen/Dense>

//
// Methods for computing the weighted geometric mean using
// a simple version and a more robust modified version of
// the Weiszfeld's Method.
// points : dim x N matrix of anchor points.
// weights: vector of N weights
//

void simpleWeightedGeometricMedian(const Eigen::MatrixXd& points, const Eigen::VectorXd& weights,
                                   Eigen::VectorXd& median, double epsilon = 1e-5, size_t maxIters = 20);

void weightedGeometricMedian(const Eigen::MatrixXd& points, const Eigen::VectorXd& weights,
                             Eigen::VectorXd& median, double epsilon = 1e-5, size_t maxIters = 20);

#endif // GEOMETRIC_MEDIAN_H
