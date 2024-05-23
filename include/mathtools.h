# pragma once

#include <Eigen/Dense>
#include <iostream>

Eigen::MatrixXd pinv_eigen_based(Eigen::MatrixXd &origin, const float er = 0)
{
    // 进行svd分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(origin, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // 构建SVD分解结果
    Eigen::MatrixXd U = svd_holder.matrixU();
    Eigen::MatrixXd V = svd_holder.matrixV();
    Eigen::MatrixXd D = svd_holder.singularValues();
    // 构建S矩阵
    Eigen::MatrixXd S(V.cols(), U.cols());
    S.setZero();
    for (unsigned int i = 0; i < D.size(); ++i)
    {
        if (D(i, 0) > er)
        {
            S(i, i) = 1 / D(i, 0);
        }
        else
        {
            S(i, i) = 0;
        }
    }
    // pinv_matrix = V * S * U^T
    return V * S * U.transpose();
}
