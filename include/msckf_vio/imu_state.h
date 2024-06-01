/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_IMU_STATE_H
#define MSCKF_VIO_IMU_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define GRAVITY_ACCELERATION 9.81

namespace msckf_vio
{

/**
 * @brief S-MSCKF中IMU状态相关
 */
struct RobotState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef long long int StateIDType;

    /// 唯一的IMU状态ID
    StateIDType id;

    /// 下一个IMU状态的ID
    static StateIDType next_id;

    /// IMU状态记录的时间
    double time;

    Eigen::MatrixXd X_i;    /// 主要的状态矩阵
    Eigen::VectorXd X_frak; /// 储存Bias
    double X_td;            /// 时间延迟

    const Eigen::Matrix3d getR_GI() const { return X_i.block<3, 3>(0, 0); }
    const Eigen::Vector3d getv_GI() const { return X_i.block<3, 1>(0, 3); }
    const Eigen::Vector3d getp_GI() const { return X_i.block<3, 1>(0, 4); }
    const Eigen::Vector3d getd_GI(int legid) const
    {
        // 分配的腿id, 从0开始, 因为会动态增删, 所以不会把id绑定到某个腿上
        assert(dimX_i() - 5 > legid);
        return X_i.block<3, 1>(0, 5 + legid);
    }
    const Eigen::Vector3d getbg() const { return X_frak.head(3); }
    const Eigen::Vector3d getba() const { return X_frak.tail(3); }

    void setR_GI(const Eigen::Matrix3d &R_GI) { X_i.block<3, 3>(0, 0) = R_GI; }
    void setv_GI(const Eigen::Vector3d &v_GI) { X_i.block<3, 1>(0, 3) = v_GI; }
    void setp_GI(const Eigen::Vector3d &p_GI) { X_i.block<3, 1>(0, 4) = p_GI; }
    void setd_GI(const Eigen::Vector3d &d_GI, int legid)
    {
        assert(dimX_i() - 5 > legid);
        X_i.block<3, 1>(0, 5 + legid) = d_GI;
    }
    void setbg(const Eigen::Vector3d &bg) { X_frak.head(3) = bg; }
    void setba(const Eigen::Vector3d &ba) { X_frak.tail(3) = ba; }

    const int dimX_i() const { return X_i.cols(); }
    const int dimX_frak() const { return X_frak.rows(); }
    const int dimP_i() const { return (X_i.cols() - 2) * 3 + 6 + 1; }

    /// 左相机坐标系到IMU坐标系的旋转矩阵，外参
    Eigen::Matrix3d R_imu_cam0;
    /// 左相机坐标系到IMU坐标系的平移向量，外参
    Eigen::Vector3d t_cam0_imu;

    /// 用于可观性约束，可观性矩阵的零空间，实际为存储上次预测时的姿态
    Eigen::Vector4d orientation_null;
    /// 用于可观性约束，可观性矩阵的零空间，实际为储存上次预测时的位置
    Eigen::Vector3d position_null;
    /// 用于可观性约束，可观性矩阵的零空间，实际为储存上次预测时的速度
    Eigen::Vector3d velocity_null;

    static double gyro_noise;
    static double acc_noise;
    static double gyro_bias_noise;
    static double acc_bias_noise;
    static double td_noise;
    static double contact_noise;

    static Eigen::Vector3d gravity;
    
    /// IMU到机身坐标系的变换矩阵，安装误差，一般为单位矩阵
    static Eigen::Isometry3d T_imu_body;

    RobotState() 
        : id(0), time(0),
        X_i(Eigen::MatrixXd::Identity(5, 5)),
        X_frak(Eigen::VectorXd::Zero(6)),
        X_td(0.0),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) 
        {
            setR_GI(Eigen::Matrix3d::Identity());
            setv_GI(Eigen::Vector3d::Zero());
            setp_GI(Eigen::Vector3d::Zero());
            setbg(Eigen::Vector3d::Zero());
            setba(Eigen::Vector3d::Zero());    
        }

    RobotState(const StateIDType &new_id)
        : id(new_id), time(0),
        X_i(Eigen::MatrixXd::Identity(5, 5)),
        X_frak(Eigen::VectorXd::Zero(6)),
        X_td(0.0),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) 
        {
            setR_GI(Eigen::Matrix3d::Identity());
            setv_GI(Eigen::Vector3d::Zero());
            setp_GI(Eigen::Vector3d::Zero());
            setbg(Eigen::Vector3d::Zero());
            setba(Eigen::Vector3d::Zero());
        }
};

typedef RobotState::StateIDType StateIDType;

} // namespace msckf_vio

#endif // MSCKF_VIO_IMU_STATE_H
