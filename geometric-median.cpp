#include <numeric>
#include <limits>
#include <cmath>
#include "geometric-median.h"

//
// Weighted Geometric Median using a simplistic update formula that
// may fail whenever x in one of the anchor points.
//
// points : dim x N matrix of anchor points.
// weights: vector of N weights
//
void simpleWeightedGeometricMedian(const Eigen::MatrixXd& points, const Eigen::VectorXd& weights,
                                   Eigen::VectorXd& median, double epsilon, size_t maxIters) {
    const auto N = points.cols();
    assert(weights.size() == N);
    const auto dim = points.rows();

    const Eigen::VectorXd w = weights / weights.sum();
    const Eigen::VectorXd mean = points * w;        // weighted geometric mean
    
    Eigen::VectorXd x = mean;
    for (size_t iter = 0; iter < maxIters; iter++) {
        const Eigen::MatrixXd V = points.colwise() - x;
        const Eigen::VectorXd d = V.colwise().norm();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(N);
        for (int i = 0; i < N; i++)
            u(i) = w(i)/std::max(epsilon,d(i));
        const Eigen::MatrixXd xnext = points * u / u.sum();
        const double err = (x - xnext).norm();
        x = xnext;
        if (err <= epsilon)
            break;
    }

    median = x;
}

//
// Weighted Geometric Median
// We use the modified Weizfeld's Method accoring to the paper
//   "Weiszfeld's Method: Old and New Results"
//   by Amir Beck and Shoham Sabach,
//   Journal of Optimization Theory and Applications
//   May 9, 2014
// https://ssabach.net.technion.ac.il/files/2015/12/BS2015.pdf
// points : dim x N matrix of anchor points.
// weights: vector of N weights
//
void weightedGeometricMedian(const Eigen::MatrixXd& points, const Eigen::VectorXd& weights,
                             Eigen::VectorXd& median, double epsilon, size_t maxIters) {
    const auto N = points.cols();
    assert(weights.size() == N);
    const auto dim = points.rows();

    const Eigen::VectorXd w = weights / weights.sum(); // make weights sum to 1

    //
    // Cache all distances between anchor points in symmetric array D.
    // Outside the diagonal w add a small value to avoid division by zero later.
    //
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(N,N);
    constexpr double tiny = 1e-5;  // used to avoid division by zero
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            D(i,j) = D(j,i) = (points.col(i) - points.col(j)).norm() + tiny;

    //
    // Choose p according to Eqn 20.
    // f(x) = \sum_i w_i * dist(x,a_i)  (FW = Femat-Weber problem)
    //
    const Eigen::VectorXd fa = D*w;  // f(a) for a in points
    Eigen::MatrixXd::Index minIndex;
    const double minF = fa.minCoeff(&minIndex);      // min f(a)
    const Eigen::VectorXd p = points.col(minIndex);  // p = argmin f(a)

    //
    // Compute R(a_i) = \sum_{i != j}  w_i (a_j - a_i) / d(a_j - a_i)  (see Section 7.1)
    //           L(a_i) = \sum_{i != j} w_j / d(a_j - a_i)             (eqn 19)
    //
    Eigen::VectorXd La = Eigen::VectorXd::Zero(N);
    Eigen::MatrixXd Ra = Eigen::MatrixXd::Zero(dim, N);
    for (int j = 0; j < N; j++) {
        double Lsum = 0.0;
        Eigen::VectorXd Rsum = Eigen::VectorXd::Zero(dim);
        for (int i = 0; i < N; i++)
            if (i != j) {
                const double c = w(i)/D(i,j);
                Lsum += c;
                Rsum += c * (points.col(j) - points.col(i));
            }
        La(j) = Lsum;
        Ra.col(j) = Rsum;
    }

    //
    // If || R(p = a_j) || < w_j, then p = a_j is median (see eqn 18)
    //
    const Eigen::VectorXd Rp = Ra.col(minIndex);
    const double wp = w(minIndex);
    const double Rp_norm = Rp.norm();
    if (Rp_norm < wp) {
        median = p;
        return;
    }

    //
    // Choose starting point S(p)   (see Section 7.1)
    //
    const double tp = (Rp_norm - wp)/La(minIndex);
    const Eigen::VectorXd dp = -Rp / Rp_norm;
    const Eigen::VectorXd ap = points.col(minIndex);
    const Eigen::VectorXd Sp = ap + tp * dp;

    //
    // Modified Weiszfeld's Methid  (see eqn 18)
    // We use the Vardi and Zhang update for L (eqn 19)
    // when at anchor point.
    //
    Eigen::VectorXd x = Sp;
    for (size_t iter = 0; iter < maxIters; iter++) {
        const Eigen::MatrixXd V = points.colwise() - x;
        const Eigen::VectorXd d = V.colwise().norm();
        Eigen::MatrixXd::Index index;
        const double mind = d.minCoeff(&index);
        Eigen::MatrixXd xnext;
        if (mind <= 0) {  // at anchor point = points(index)
            const Eigen::VectorXd Rx = Ra.col(index);
            const double Rx_norm = Rx.norm();
            const double wx = w(index);
            if (Rx_norm < wx)
                break;  // at anchor point that is actually the median
            const Eigen::VectorXd dx = -Rx / Rx_norm;
            const Eigen::VectorXd ax = points.col(index);
            const double tx = (Rx_norm - wx)/La(index);
            const Eigen::VectorXd Sx = ax + tx * dx;
            xnext = Sx;
        } else {
            const Eigen::VectorXd u = w.array() / d.array();
            xnext = points * u / u.sum();
        }
        const double err = (x - xnext).norm();
        x = xnext;
        if (err <= epsilon)
            break; // convergence
    }

    median = x;
}

#ifdef TEST_WEIGHTED_GEOMETRIC_MEAN

#include <iostream>
#include <random>

int main(int argc, char *argv[]) {

#ifdef SIMPLE_TEST_XXX

    constexpr int N = 10;
    Eigen::MatrixXd P = Eigen::Matrix<double,2,N>::Zero();
    P <<
        5, 10,  5, 10, 6, 7, 8, 9, 10, 500,   // replicated point + outlier
        5,  5, 10, 10, 9, 7, 6, 4, 5, 1000;

    Eigen::VectorXd w = Eigen::VectorXd::Constant(N,1,1.0);
    // w(1) = 2.001;

    Eigen::VectorXd median;
    simpleWeightedGeometricMedian(P, w, median);

    std::cout << "simple median:\n" << median << "\n";

    weightedGeometricMedian(P, w, median);
    std::cout << "optimal median:\n" << median << "\n";

#else

    constexpr int N = 1000;
    Eigen::MatrixXd P = Eigen::Matrix<double,2,N>::Zero();
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,25.0);
    constexpr double cx = 45, cy = 32;
    for (int i = 0; i < N; i++) {
        const double x = distribution(generator) + cx;
        const double y = distribution(generator) + cy;
        P(0,i) = x;
        P(1,i) = y;
    }

    // Add some outliers
    constexpr int M = 50;
    for (int i = 0; i < M; i++) {
        const double x = (i+1)*1000;
        const double y = (i+7)*10000;
        P(0,i) = x;
        P(1,i) = y;
    }

    // Add some duplicates
    constexpr int D = 50;
    for (int i = 0; i < M; i++)
        P.col(i + M) = P.col(i + 2*(M + D));    

    Eigen::VectorXd median;
    Eigen::VectorXd w = Eigen::VectorXd::Constant(N,1,1.0);

    simpleWeightedGeometricMedian(P, w, median);
    std::cout << "simple median:\n" << median << "\n";

    weightedGeometricMedian(P, w, median);
    std::cout << "optimal median:\n" << median << "\n";
#endif
    
    return 0;
}

#endif // TEST_WEIGHTED_GEOMETRIC_MEAN
