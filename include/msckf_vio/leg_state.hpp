#ifndef MSCKF_VIO_LEG_STATE_H
#define MSCKF_VIO_LEG_STATE_H

#include "modern_robotics.h"
#include "math_utils.hpp"

#define DEG2RAD 3.14159265358979323846 / 180.0

namespace msckf_vio
{
    enum LegID
    {
        unnamed = -1,
        FrontLeft = 0,
        RearLeft = 1,
        RearRight = 2,
        FrontRight = 3
    };

    /**
     * @brief S-MSCKF中腿状态相关
     */
    class LegState
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LegState()
        {
            legs.push_back(SingleLeg(LegID::FrontLeft));
            legs.push_back(SingleLeg(LegID::RearLeft));
            legs.push_back(SingleLeg(LegID::RearRight));
            legs.push_back(SingleLeg(LegID::FrontRight));
        }

        struct SingleLeg
        {
            LegID id;
            double time;
            double thigh_angle;            // 大腿角度 rad
            double knee_angle;             // 膝盖角度 rad
            double wheel_angle;            // 轮子角度 rad
            double wheel_angleVelocity;    // 轮子角速度 rad/s
            double wheel_radius;           // 轮子半径 m
            bool contact;                  // 是否接触地面
            Eigen::Matrix4d T;             // 足端到机身的变换矩阵
            Eigen::Matrix<double, 6, 3> J; // 足端到机身的雅可比矩阵
            Eigen::Matrix3d Cov;           // 腿的协方差
            SingleLeg(LegID id = LegID::unnamed) : id(id), time(0.0), thigh_angle(0.0), knee_angle(0.0),
                                                   wheel_angle(0.0), wheel_angleVelocity(0.0),
                                                   wheel_radius(0.05), contact(false),
                                                   T(Eigen::Matrix4d::Identity()),
                                                   J(Eigen::Matrix<double, 6, 3>::Zero()),
                                                   Cov(Eigen::Matrix3d::Zero()){}
        };
        std::vector<SingleLeg> legs;

        void calc_FKAndJacobian();
        Eigen::Vector3d getVelocityInBodyFrame(const LegID &id) const;
    };

    void LegState::calc_FKAndJacobian()
    {
        Eigen::Matrix4d M_FL, M_RL, M_RR, M_FR;
        Eigen::Matrix<double, 3, 6> Slist_FL, Slist_RL, Slist_RR, Slist_FR;
        M_FL << std::cos(DEG2RAD * -76.426), 0.0, std::sin(DEG2RAD * -76.426), 250.01 / 1000, 0.0, 1.0, 0.0, 152.0 / 1000,
            -std::sin(DEG2RAD * -76.426), 0.0, std::cos(DEG2RAD * -76.426), -98.1 / 1000, 0.0, 0.0, 0.0, 1.0;
        M_RL << std::cos(DEG2RAD * -76.426), 0.0, std::sin(DEG2RAD * -76.426), -129.99 / 1000, 0.0, 1.0, 0.0, 152.0 / 1000,
            -std::sin(DEG2RAD * -76.426), 0.0, std::cos(DEG2RAD * -76.426), -98.1 / 1000, 0.0, 0.0, 0.0, 1.0;
        M_RR << std::cos(DEG2RAD * -76.426), 0.0, std::sin(DEG2RAD * -76.426), -129.99 / 1000, 0.0, 1.0, 0.0, -152.0 / 1000,
            -std::sin(DEG2RAD * -76.426), 0.0, std::cos(DEG2RAD * -76.426), -98.1 / 1000, 0.0, 0.0, 0.0, 1.0;
        M_FR << std::cos(DEG2RAD * -76.426), 0.0, std::sin(DEG2RAD * -76.426), 250.01 / 1000, 0.0, 1.0, 0.0, -152.0 / 1000,
            -std::sin(DEG2RAD * -76.426), 0.0, std::cos(DEG2RAD * -76.426), -98.1 / 1000, 0.0, 0.0, 0.0, 1.0;
        Slist_FL << 0.0, 1.0, 0.0, 0.0, 0.0, 190.0 / 1000, 0.0, 1.0, 0.0, 129.1 / 1000, 0.0, 52.42 / 1000, -0.972067605,
            0.0, 0.234701027, 35.6745561 / 1000, 41.27399049 / 1000, 147.7542759 / 1000;
        Slist_RL << 0.0, 1.0, 0.0, 0.0, 0.0, -190.0 / 1000, 0.0, 1.0, 0.0, 129.1 / 1000, 0.0, -327.58 / 1000, -0.972067605,
            0.0, 0.234701027, 35.6745561 / 1000, 123.4914685 / 1000, 147.7542759 / 1000;
        Slist_RR << 0.0, -1.0, 0.0, 0.0, 0.0, 190.0 / 1000, 0.0, -1.0, 0.0, -129.1 / 1000, 0.0, 327.58 / 1000, -0.972067605,
            0.0, 0.234701027, -35.6745561 / 1000, 123.4914685 / 1000, -147.7542759 / 1000;
        Slist_FR << 0.0, -1.0, 0.0, 0.0, 0.0, -190.0 / 1000, 0.0, -1.0, 0.0, -129.1 / 1000, 0.0, -52.42 / 1000,
            -0.972067605, 0.0, 0.234701027, -35.6745561 / 1000, 41.27399049 / 1000, -147.7542759 / 1000;
        Eigen::Matrix<double, 3, 1> thetalist_FL, thetalist_RL, thetalist_RR, thetalist_FR;
        thetalist_FL << this->legs[LegID::FrontLeft].thigh_angle, this->legs[LegID::FrontLeft].knee_angle, this->legs[LegID::FrontLeft].wheel_angle;
        thetalist_RL << this->legs[LegID::RearLeft].thigh_angle, this->legs[LegID::RearLeft].knee_angle, this->legs[LegID::RearLeft].wheel_angle;
        thetalist_RR << this->legs[LegID::RearRight].thigh_angle, this->legs[LegID::RearRight].knee_angle, this->legs[LegID::RearRight].wheel_angle;
        thetalist_FR << this->legs[LegID::FrontRight].thigh_angle, this->legs[LegID::FrontRight].knee_angle, this->legs[LegID::FrontRight].wheel_angle;
        // std::cout << "thetalist_FL: " << thetalist_FL.transpose() << std::endl;
        // std::cout << "thetalist_RL: " << thetalist_RL.transpose() << std::endl;
        // std::cout << "thetalist_RR: " << thetalist_RR.transpose() << std::endl;
        // std::cout << "thetalist_FR: " << thetalist_FR.transpose() << std::endl;
        // std::cout << "T_FL: \n" << result.T_FL << std::endl;
        // std::cout << "T_RL: \n" << result.T_RL << std::endl;
        // std::cout << "T_RR: \n" << result.T_RR << std::endl;
        // std::cout << "T_FR: \n" << result.T_FR << std::endl;
        // std::cout << "====================================================\n";
        this->legs[LegID::FrontLeft].T = mr::FKinSpace(M_FL, Slist_FL.transpose(), thetalist_FL);
        this->legs[LegID::RearLeft].T = mr::FKinSpace(M_RL, Slist_RL.transpose(), thetalist_RL);
        this->legs[LegID::RearRight].T = mr::FKinSpace(M_RR, Slist_RR.transpose(), thetalist_RR);
        this->legs[LegID::FrontRight].T = mr::FKinSpace(M_FR, Slist_FR.transpose(), thetalist_FR);
        Eigen::Matrix<double, 6, 3> J_FL_PoE = mr::JacobianSpace(Slist_FL.transpose(), thetalist_FL);
        Eigen::Matrix<double, 6, 3> J_RL_PoE = mr::JacobianSpace(Slist_RL.transpose(), thetalist_RL);
        Eigen::Matrix<double, 6, 3> J_RR_PoE = mr::JacobianSpace(Slist_RR.transpose(), thetalist_RR);
        Eigen::Matrix<double, 6, 3> J_FR_PoE = mr::JacobianSpace(Slist_FR.transpose(), thetalist_FR);
        Eigen::Matrix3d p_FL_hat = skewSymmetric(Eigen::Matrix<double, 3, 1>(this->legs[LegID::FrontLeft].T.block<3, 1>(0, 3)));
        Eigen::Matrix3d p_RL_hat = skewSymmetric(Eigen::Matrix<double, 3, 1>(this->legs[LegID::RearLeft].T.block<3, 1>(0, 3)));
        Eigen::Matrix3d p_RR_hat = skewSymmetric(Eigen::Matrix<double, 3, 1>(this->legs[LegID::RearRight].T.block<3, 1>(0, 3)));
        Eigen::Matrix3d p_FR_hat = skewSymmetric(Eigen::Matrix<double, 3, 1>(this->legs[LegID::FrontRight].T.block<3, 1>(0, 3)));
        Eigen::Matrix<double, 6, 6> FL_PoE2DH, RL_PoE2DH, RR_PoE2DH, FR_PoE2DH;
        FL_PoE2DH << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), -p_FL_hat, Eigen::Matrix3d::Identity();
        RL_PoE2DH << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), -p_RL_hat, Eigen::Matrix3d::Identity();
        RR_PoE2DH << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), -p_RR_hat, Eigen::Matrix3d::Identity();
        FR_PoE2DH << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(), -p_FR_hat, Eigen::Matrix3d::Identity();
        this->legs[LegID::FrontLeft].J = FL_PoE2DH * J_FL_PoE;
        this->legs[LegID::RearLeft].J = RL_PoE2DH * J_RL_PoE;
        this->legs[LegID::RearRight].J = RR_PoE2DH * J_RR_PoE;
        this->legs[LegID::FrontRight].J = FR_PoE2DH * J_FR_PoE;
    }

    Eigen::Vector3d LegState::getVelocityInBodyFrame(const LegID &id) const
    {
        double wheel_velocity;
        if(id == LegID::FrontLeft || id == LegID::RearLeft)
            wheel_velocity = this->legs[id].wheel_angleVelocity * this->legs[id].wheel_radius;
        else if(id == LegID::RearRight || id == LegID::FrontRight)
            wheel_velocity = -this->legs[id].wheel_angleVelocity * this->legs[id].wheel_radius;

        Eigen::Vector3d w = this->legs[id].T.block<3, 1>(0, 2);
        double theta3 = atan(tan(this->legs[id].wheel_angle) * w[2]);
        Eigen::Vector3d velocity_in_body_frame;
        velocity_in_body_frame[0] = wheel_velocity * cos(theta3);
        velocity_in_body_frame[1] = -wheel_velocity * sin(theta3);
        velocity_in_body_frame[2] = 0.0;
        return velocity_in_body_frame;
    }

} // namespace msckf_vio

#endif // MSCKF_VIO_LEG_STATE_H
