#include <iostream>
#include <vector>
#include <numeric>
#include <limits>
#include <cmath>
#include <Eigen/Dense>

//
// Weighted Geometric Median using a simplistic update formula that
// fails whenever x in one of the feature points.
//
// points : dim x N matrix of anchor points.
// weights: vector of N weights
//
void simpleWeightedGeometricMedian(const Eigen::MatrixXd& points, const Eigen::VectorXd& weights,
                                   Eigen::VectorXd& median, double epsilon = 1e-5, size_t maxIters = 20) {
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
// https://ssabach.net.technion.ac.il/files/2015/12/BS2015.pdf
//
void weightedGeometricMedian(const Eigen::MatrixXd& points, const Eigen::VectorXd& weights,
                             Eigen::VectorXd& median, double epsilon = 1e-5, size_t maxIters = 20) {
    const auto N = points.cols();
    assert(weights.size() == N);
    const auto dim = points.rows();

    const Eigen::VectorXd w = weights / weights.sum();

    constexpr double tiny = 1e-5;  // used to avoid division by zero

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(N,N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            D(i,j) = D(j,i) = (points.col(i) - points.col(j)).norm() + tiny;

    const Eigen::VectorXd fa = D*w;
    Eigen::MatrixXd::Index minIndex;
    const double minF = fa.minCoeff(&minIndex);
    const Eigen::VectorXd p = points.col(minIndex);

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

    const Eigen::VectorXd Rp = Ra.col(minIndex);
    const double wp = w(minIndex);
    const double Rp_norm = Rp.norm();
    if (Rp_norm < wp)
        return p;

    const double tp = (Rp_norm - wp)/La(minIndex);
    const Eigen::VectorXd dp = -Rp / Rp_norm;
    const Eigen::VectorXd ap = points.col(minIndex);
    const Eigen::VectorXd Sp = ap + tp * dp;

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
            break;
    }

    median = x;
}

int main(int argc, char *argv[]) {

    constexpr int N = 4;
    Eigen::MatrixXd P = Eigen::Matrix<double,2,N>::Zero();
    P <<
        5, 10,  5, 10,
        5,  5, 10, 10;

    Eigen::VectorXd w = Eigen::VectorXd::Constant(N,1,1.0);

    Eigen::VectorXd median;
    simpleWeightedGeometricMedian(P, w, median);

    std::cout << "simple median:\n" << median << "\n";

    weightedGeometricMedian(P, w, median);
    std::cout << "optimal median:\n" << median << "\n";
    
    return 0;
}
