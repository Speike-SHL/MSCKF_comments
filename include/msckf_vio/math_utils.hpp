/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_MATH_UTILS_HPP
#define MSCKF_VIO_MATH_UTILS_HPP

#include <cmath>
#include <Eigen/Dense>

namespace msckf_vio
{

/**
 *  @brief 反对称矩阵
 *  @note Performs the operation:
 *  w   ->  [  0 -w3  w2]
 *          [ w3   0 -w1]
 *          [-w2  w1   0]
 */
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d w_hat;
    w_hat(0, 0) = 0;
    w_hat(0, 1) = -w(2);
    w_hat(0, 2) = w(1);
    w_hat(1, 0) = w(2);
    w_hat(1, 1) = 0;
    w_hat(1, 2) = -w(0);
    w_hat(2, 0) = -w(1);
    w_hat(2, 1) = w(0);
    w_hat(2, 2) = 0;
    return w_hat;
}

/**
 * @brief 标准化四元数
 */
inline void quaternionNormalize(Eigen::Vector4d &q)
{
    double norm = q.norm();
    q = q / norm;
    return;
}

/**
 * @brief Perform q1 * q2
 * Indirect Kalman Filter for 3D Attitude Estimation 公式8
 * jpl四元数乘法
 */
inline Eigen::Vector4d quaternionMultiplication(
    const Eigen::Vector4d &q1,
    const Eigen::Vector4d &q2)
{
    Eigen::Matrix4d L;
    L(0, 0) = q1(3);
    L(0, 1) = q1(2);
    L(0, 2) = -q1(1);
    L(0, 3) = q1(0);
    L(1, 0) = -q1(2);
    L(1, 1) = q1(3);
    L(1, 2) = q1(0);
    L(1, 3) = q1(1);
    L(2, 0) = q1(1);
    L(2, 1) = -q1(0);
    L(2, 2) = q1(3);
    L(2, 3) = q1(2);
    L(3, 0) = -q1(0);
    L(3, 1) = -q1(1);
    L(3, 2) = -q1(2);
    L(3, 3) = q1(3);

    Eigen::Vector4d q = L * q2;
    quaternionNormalize(q);
    return q;
}

/**
 * @brief 李代数转四元数，小量
 * Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
inline Eigen::Vector4d smallAngleQuaternion(
    const Eigen::Vector3d &dtheta)
{
    // δq ~= (1/2δθ, 1)
    Eigen::Vector3d dq = dtheta / 2.0;
    Eigen::Vector4d q;
    double dq_square_norm = dq.squaredNorm();

    // 这么做就是为了符合四元数的定义
    // q(3)的平方+q.head<3>()的平方和的和是1
    if (dq_square_norm <= 1)
    {
        q.head<3>() = dq;
        q(3) = std::sqrt(1 - dq_square_norm);
    }
    else
    {
        q.head<3>() = dq;
        q(3) = 1;
        q = q / std::sqrt(1 + dq_square_norm);
    }

    return q;
}

/**
 * @brief 四元数转旋转矩阵 jpl
 * Convert a quaternion to the corresponding rotation matrix
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (62).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Matrix3d quaternionToRotation(
    const Eigen::Vector4d &q)
{
    const Eigen::Vector3d &q_vec = q.block(0, 0, 3, 1);
    const double &q4 = q(3);
    Eigen::Matrix3d R =
        (2 * q4 * q4 - 1) * Eigen::Matrix3d::Identity() -
        2 * q4 * skewSymmetric(q_vec) +
        2 * q_vec * q_vec.transpose();
    return R;
}

/**
 * @brief 旋转矩阵转四元数 没在论文里找到，这里不用看，直接用！
 * Convert a rotation matrix to a quaternion.
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Vector4d rotationToQuaternion(
    const Eigen::Matrix3d &R)
{
    Eigen::Vector4d score;
    score(0) = R(0, 0);
    score(1) = R(1, 1);
    score(2) = R(2, 2);
    score(3) = R.trace();

    int max_row = 0, max_col = 0;
    score.maxCoeff(&max_row, &max_col);

    Eigen::Vector4d q = Eigen::Vector4d::Zero();
    if (max_row == 0)
    {
        q(0) = std::sqrt(1 + 2 * R(0, 0) - R.trace()) / 2.0;
        q(1) = (R(0, 1) + R(1, 0)) / (4 * q(0));
        q(2) = (R(0, 2) + R(2, 0)) / (4 * q(0));
        q(3) = (R(1, 2) - R(2, 1)) / (4 * q(0));
    }
    else if (max_row == 1)
    {
        q(1) = std::sqrt(1 + 2 * R(1, 1) - R.trace()) / 2.0;
        q(0) = (R(0, 1) + R(1, 0)) / (4 * q(1));
        q(2) = (R(1, 2) + R(2, 1)) / (4 * q(1));
        q(3) = (R(2, 0) - R(0, 2)) / (4 * q(1));
    }
    else if (max_row == 2)
    {
        q(2) = std::sqrt(1 + 2 * R(2, 2) - R.trace()) / 2.0;
        q(0) = (R(0, 2) + R(2, 0)) / (4 * q(2));
        q(1) = (R(1, 2) + R(2, 1)) / (4 * q(2));
        q(3) = (R(0, 1) - R(1, 0)) / (4 * q(2));
    }
    else
    {
        q(3) = std::sqrt(1 + R.trace()) / 2.0;
        q(0) = (R(1, 2) - R(2, 1)) / (4 * q(3));
        q(1) = (R(2, 0) - R(0, 2)) / (4 * q(3));
        q(2) = (R(0, 1) - R(1, 0)) / (4 * q(3));
    }

    if (q(3) < 0)
        q = -q;
    quaternionNormalize(q);
    return q;
}

Eigen::MatrixXd Adjoint_SEK3(const Eigen::MatrixXd &X)
{
    int K = X.cols() - 3;
    Eigen::MatrixXd Adj = Eigen::MatrixXd::Zero(3 + 3 * K, 3 + 3 * K);
    Eigen::Matrix3d R = X.block<3, 3>(0, 0);
    Adj.block<3, 3>(0, 0) = R;
    for (int i = 0; i < K; ++i)
    {
        Adj.block<3, 3>(3 + 3 * i, 3 + 3 * i) = R;
        Adj.block<3, 3>(3 + 3 * i, 0) = skewSymmetric(X.block<3, 1>(0, 3 + i)) * R;
    }
    return Adj;
}

long int myfactorial(int n) { return (n == 1 || n == 0) ? 1 : myfactorial(n - 1) * n; }

Eigen::Matrix3d Gamma_SO3(const Eigen::Vector3d &w, int m) {
    // Computes mth integral of the exponential map: \Gamma_m =
    // \sum_{n=0}^{\infty} \dfrac{1}{(n+m)!} (w^\wedge)^n
    assert(m >= 0);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    double theta = w.norm();
    if (theta < 1e-10) {
        return (1.0 / myfactorial(m)) * I;
    }
    Eigen::Matrix3d A = skewSymmetric(w);
    double theta2 = theta * theta;

    // Closed form solution for the first 3 cases
    switch (m) {
        case 0:  // Exp map of SO(3)
            return I + (sin(theta) / theta) * A + ((1 - cos(theta)) / theta2) * A * A;

        case 1:  // Left Jacobian of SO(3)
            // eye(3) - A*(1/theta^2) * (R - eye(3) - A);
            // eye(3) + (1-cos(theta))/theta^2 * A + (theta-sin(theta))/theta^3 * A^2;
            return I + ((1 - cos(theta)) / theta2) * A + ((theta - sin(theta)) / (theta2 * theta)) * A * A;

        case 2:
            // 0.5*eye(3) - (1/theta^2) * (R - eye(3) - A - 0.5*A^2);
            // 0.5*eye(3) + (theta-sin(theta))/theta^3 * A + (2*(cos(theta)-1) +
            // theta^2)/(2*theta^4) * A^2
            return 0.5 * I + (theta - sin(theta)) / (theta2 * theta) * A +
                   (theta2 + 2 * cos(theta) - 2) / (2 * theta2 * theta2) * A * A;

        default:  // General case
            Eigen::Matrix3d R = I + (sin(theta) / theta) * A + ((1 - cos(theta)) / theta2) * A * A;
            Eigen::Matrix3d S = I;
            Eigen::Matrix3d Ak = I;
            long int kfactorial = 1;
            for (int k = 1; k <= m; ++k) {
                kfactorial = kfactorial * k;
                Ak = (Ak * A).eval();
                S = (S + (1.0 / kfactorial) * Ak).eval();
            }
            if (m == 0) {
                return R;
            } else if (m % 2) {  // odd
                return (1.0 / kfactorial) * I + (pow(-1, (m + 1) / 2) / pow(theta, m + 1)) * A * (R - S);
            } else {  // even
                return (1.0 / kfactorial) * I + (pow(-1, m / 2) / pow(theta, m)) * (R - S);
            }
    }
}

Eigen::MatrixXd Exp_SEK3(const Eigen::VectorXd &v)
{
    int K = (v.size() - 3) / 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Identity(3 + K, 3 + K);
    Eigen::Matrix3d R;
    Eigen::Matrix3d Jl;
    Eigen::Vector3d w = v.head(3);
    double theta = w.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < 1e-10)
    {
        R = I;
        Jl = I;
    }
    else
    {
        Eigen::Matrix3d A = skewSymmetric(w);
        double theta2 = theta * theta;
        double stheta = sin(theta);
        double ctheta = cos(theta);
        double oneMinusCosTheta2 = (1 - ctheta) / (theta2);
        Eigen::Matrix3d A2 = A * A;
        R = I + (stheta / theta) * A + oneMinusCosTheta2 * A2;
        Jl = I + oneMinusCosTheta2 * A + ((theta - stheta) / (theta2 * theta)) * A2;
    }
    X.block<3, 3>(0, 0) = R;
    for (int i = 0; i < K; ++i)
    {
        X.block<3, 1>(0, 3 + i) = Jl * v.segment<3>(3 + 3 * i);
    }
    return X;
}

Eigen::Matrix3d Exp_SO3(const Eigen::Vector3d &w) { return Gamma_SO3(w, 0); }

template <typename T>
Eigen::Matrix<T, 3, 1> Log_SO3(const Eigen::Matrix<T, 3, 3>& R) {
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

inline Eigen::Matrix4d Exp_SE3(const Eigen::Vector3d &w,
                               const Eigen::Vector3d &u)
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
    double theta = w.norm();

    if (!(theta < 1e-10))
    {
        Eigen::Matrix3d A = skewSymmetric(w);
        Eigen::Matrix3d A2 = A * A;
        double theta2 = theta * theta;
        double stheta = sin(theta);
        double ctheta = cos(theta);

        R += stheta / theta * A + (1 - ctheta) / theta2 * A2;
        V += (1 - ctheta) / theta2 * A + (theta - stheta) / (theta2 * theta) * A2;
    }

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = V * u;

    return T;
}

inline Eigen::Matrix<double, 5, 5> Exp_SE3(const Eigen::Vector3d &w,
                                           const Eigen::Vector3d &u,
                                           const Eigen::Vector3d &y)
{
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
    double theta = w.norm();

    if (!(theta < 1e-10))
    {
        Eigen::Matrix3d A = skewSymmetric(w);
        Eigen::Matrix3d A2 = A * A;
        double theta2 = theta * theta;
        double stheta = sin(theta);
        double ctheta = cos(theta);

        R += stheta / theta * A + (1 - ctheta) / theta2 * A2;
        V += (1 - ctheta) / theta2 * A + (theta - stheta) / (theta2 * theta) * A2;
    }

    Eigen::Matrix<double, 5, 5> T = Eigen::Matrix<double, 5, 5>::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = V * u;
    T.block<3, 1>(0, 4) = V * y;

    return T;
}

} // end namespace msckf_vio

#endif // MSCKF_VIO_MATH_UTILS_HPP
