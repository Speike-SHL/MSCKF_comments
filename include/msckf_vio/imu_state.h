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
#include "leg_state.hpp"

#define GRAVITY_ACCELERATION 9.81

namespace msckf_vio
{

    /**
     * @brief S-MSCKF中IMU状态相关
     */
    struct RobotState
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef long long int StateIDType;
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        typedef Eigen::Matrix<double, 9, 9> Matrix9d;

        /// 唯一的IMU状态ID
        StateIDType id;

        /// 下一个IMU状态的ID
        static StateIDType next_id;

        /// IMU状态记录的时间
        double time;

        Matrix9d X_i;                    /// 主要的状态矩阵
        int X_i_valid_size;              /// X_i中有效维度的大小
        Vector6d X_frak;                 /// 储存Bias
        double X_td;                     /// 时间延迟

        const Eigen::MatrixXd getX_i() const
        {
            assert(X_i_valid_size >= 5 && X_i_valid_size <= 9);
            return X_i.topLeftCorner(X_i_valid_size, X_i_valid_size);
        }
        const Vector6d getX_frak() const { return X_frak; }
        const Eigen::Matrix3d getR_GI() const { return X_i.block<3, 3>(0, 0); }
        const Eigen::Vector3d getv_GI() const { return X_i.block<3, 1>(0, 3); }
        const Eigen::Vector3d getp_GI() const { return X_i.block<3, 1>(0, 4); }
        const Eigen::Vector3d getd_GI(int index) const
        {
            // 分配的腿id, 从0开始, 因为会动态增删, 所以不会把id绑定到某个腿上
            assert(index >= 5 && index <= X_i_valid_size - 1);
            return X_i.block<3, 1>(0, index);
        }
        const Eigen::Vector3d getbg() const { return X_frak.head(3); }
        const Eigen::Vector3d getba() const { return X_frak.tail(3); }

        void setX_i(const Eigen::MatrixXd &X_i_)
        {
            assert(X_i_.rows() == X_i_valid_size);
            assert(X_i_.rows() >= 5 && X_i_.rows() <= 9);
            X_i.topLeftCorner(X_i_valid_size, X_i_valid_size) = X_i_;
        }
        void setX_frak(const Vector6d &X_frak_) { X_frak = X_frak_; }
        void setR_GI(const Eigen::Matrix3d &R_GI) { X_i.block<3, 3>(0, 0) = R_GI; }
        void setv_GI(const Eigen::Vector3d &v_GI) { X_i.block<3, 1>(0, 3) = v_GI; }
        void setp_GI(const Eigen::Vector3d &p_GI) { X_i.block<3, 1>(0, 4) = p_GI; }
        void setd_GI(const Eigen::Vector3d &d_GI, int index)
        {
            assert(index >= 5 && index <= X_i_valid_size - 1);
            X_i.block<3, 1>(0, index) = d_GI;
        }
        void setbg(const Eigen::Vector3d &bg) { X_frak.head(3) = bg; }
        void setba(const Eigen::Vector3d &ba) { X_frak.tail(3) = ba; }

        int dimX_i() const { return X_i_valid_size; }
        int dimX_frak() const { return X_frak.rows(); }
        int dimP_i() const { return (X_i_valid_size - 2) * 3 + 6; } // NOTE: 加td的话要+1

        /// 左相机坐标系到IMU坐标系的外参
        Eigen::Matrix3d R_cam0_imu;
        Eigen::Vector3d t_cam0_imu;

        // cam1到cam0的外参
        Eigen::Matrix3d R_cam1_cam0;
        Eigen::Vector3d t_cam1_cam0;

        static double gyro_noise;
        static double acc_noise;
        static double gyro_bias_noise;
        static double acc_bias_noise;
        static double td_noise;
        static double contact_noise;
        static double encoder_noise;
        static double kinematics_additive_noise;

        static Eigen::Vector3d gravity;

        /// IMU到机身坐标系的变换矩阵，安装误差，一般为单位矩阵
        static Eigen::Isometry3d T_imu_body;

        /// 记录已经加入状态估计的腿的ID和在X_i中的索引
        std::map<LegID, int> estimated_contact_position;

        RobotState()
            : id(0), time(0),
              X_i(Matrix9d::Identity()),
              X_i_valid_size(5),
              X_frak(Vector6d::Zero()),
              X_td(0.0)
        {
            setR_GI(Eigen::Matrix3d::Identity());
            setv_GI(Eigen::Vector3d::Zero());
            setp_GI(Eigen::Vector3d::Zero());
            setbg(Eigen::Vector3d::Zero());
            setba(Eigen::Vector3d::Zero());
        }

        RobotState(const StateIDType &new_id)
            : id(new_id), time(0),
              X_i(Matrix9d::Identity()),
              X_i_valid_size(5),
              X_frak(Vector6d::Zero()),
              X_td(0.0)
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
