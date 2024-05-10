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
struct IMUState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef long long int StateIDType;

    /// 唯一的IMU状态ID
    StateIDType id;

    /// 下一个IMU状态的ID
    static StateIDType next_id;

    /// IMU状态记录的时间
    double time;

    /// Rbw, Take a vector from the world frame to the IMU (body) frame.
    Eigen::Vector4d orientation;

    /// twb, Position of the IMU (body) frame in the world frame.
    Eigen::Vector3d position;

    /// Vwb, Velocity of the IMU (body) frame in the world frame.
    Eigen::Vector3d velocity;

    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

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

    static Eigen::Vector3d gravity;
    
    /// IMU到机身坐标系的变换矩阵，安装误差，一般为单位矩阵
    static Eigen::Isometry3d T_imu_body;

    IMUState() 
        : id(0), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        gyro_bias(Eigen::Vector3d::Zero()),
        acc_bias(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) {}

    IMUState(const StateIDType &new_id)
        : id(new_id), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        gyro_bias(Eigen::Vector3d::Zero()),
        acc_bias(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) {}
};

typedef IMUState::StateIDType StateIDType;

} // namespace msckf_vio

#endif // MSCKF_VIO_IMU_STATE_H
