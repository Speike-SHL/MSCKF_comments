#ifndef WEBOTS_HPP
#define WEBOTS_HPP

#include <Eigen/Dense>

namespace msckf_vio
{

    class WebotsRealState
    {
    public:
        WebotsRealState() {
            time = 0;
            body_acc_in_IMU = Eigen::Vector3d::Zero();
            body_gyro_in_IMU = Eigen::Vector3d::Zero();
            body_rotation_in_world = Eigen::Matrix3d::Identity();
            body_position_in_world = Eigen::Vector3d::Zero();
            body_velocity_in_world = Eigen::Vector3d::Zero();
            FLlun_position_in_world = Eigen::Vector3d::Zero();
            RLlun_position_in_world = Eigen::Vector3d::Zero();
            RRlun_position_in_world = Eigen::Vector3d::Zero();
            FRlun_position_in_world = Eigen::Vector3d::Zero();
            FLlun_velocity_in_world = Eigen::Vector3d::Zero();
            RLlun_velocity_in_world = Eigen::Vector3d::Zero();
            RRlun_velocity_in_world = Eigen::Vector3d::Zero();
            FRlun_velocity_in_world = Eigen::Vector3d::Zero();
            FLlun_velocity_in_body = Eigen::Vector3d::Zero();
            RLlun_velocity_in_body = Eigen::Vector3d::Zero();
            RRlun_velocity_in_body = Eigen::Vector3d::Zero();
            FRlun_velocity_in_body = Eigen::Vector3d::Zero();
            FLlun_Transform_to_body = Eigen::Matrix4d::Identity();
            RLlun_Transform_to_body = Eigen::Matrix4d::Identity();
            RRlun_Transform_to_body = Eigen::Matrix4d::Identity();
            FRlun_Transform_to_body = Eigen::Matrix4d::Identity();
            FLlun_DHJacobian_to_body = Eigen::Matrix<double, 6, 3>::Zero();
            RLlun_DHJacobian_to_body = Eigen::Matrix<double, 6, 3>::Zero();
            RRlun_DHJacobian_to_body = Eigen::Matrix<double, 6, 3>::Zero();
            FRlun_DHJacobian_to_body = Eigen::Matrix<double, 6, 3>::Zero();
            Fourlun_contact = Eigen::Matrix<bool, 4, 1>(false, false, false, false);

            R_NWU2NUE << 1, 0, 0, 0, 0, -1, 0, 1, 0;
        }
        ~WebotsRealState() {}

        double time;
        Eigen::Vector3d body_acc_in_IMU;
        Eigen::Vector3d body_gyro_in_IMU;
        Eigen::Matrix3d body_rotation_in_world;
        Eigen::Vector3d body_position_in_world;
        Eigen::Vector3d body_velocity_in_world;
        Eigen::Vector3d FLlun_position_in_world;
        Eigen::Vector3d RLlun_position_in_world;
        Eigen::Vector3d RRlun_position_in_world;
        Eigen::Vector3d FRlun_position_in_world;
        Eigen::Vector3d FLlun_velocity_in_world;
        Eigen::Vector3d RLlun_velocity_in_world;
        Eigen::Vector3d RRlun_velocity_in_world;
        Eigen::Vector3d FRlun_velocity_in_world;
        Eigen::Vector3d FLlun_velocity_in_body;
        Eigen::Vector3d RLlun_velocity_in_body;
        Eigen::Vector3d RRlun_velocity_in_body;
        Eigen::Vector3d FRlun_velocity_in_body;
        Eigen::Matrix4d FLlun_Transform_to_body;
        Eigen::Matrix4d RLlun_Transform_to_body;
        Eigen::Matrix4d RRlun_Transform_to_body;
        Eigen::Matrix4d FRlun_Transform_to_body;
        Eigen::Matrix<double, 6, 3> FLlun_DHJacobian_to_body;
        Eigen::Matrix<double, 6, 3> RLlun_DHJacobian_to_body;
        Eigen::Matrix<double, 6, 3> RRlun_DHJacobian_to_body;
        Eigen::Matrix<double, 6, 3> FRlun_DHJacobian_to_body;
        Eigen::Matrix<bool, 4, 1> Fourlun_contact;

        Eigen::Matrix3d R_NWU2NUE;
    };
}

#endif // WEBOTS_HPP
