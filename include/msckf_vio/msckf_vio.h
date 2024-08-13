/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include "sophus/se3.hpp"
#include <iomanip>
#include <iterator>
#include "yaml-cpp/yaml.h"

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <thread>
#include <mutex>
#include "mathtools.h"
#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>

#include "imu_state.h"
#include "cam_state.h"
#include "leg_state.hpp"
#include "feature.hpp"
#include "../tic_toc.h"
#include <msckf_vio/CameraMeasurement.h>
#include "dog_msg/isTouchdown.h"
#include "dog_msg/qNow_dqNow_TNow.h"
#include "dog_msg/wheel_motor_fb.h"

namespace msckf_vio
{
/*
 * @brief MsckfVio Implements the algorithm in
 *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
 *    "A Multi-State Constraint Kalman Filter for Vision-aided
 *    Inertial Navigation",
 *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */
class MsckfVio
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MsckfVio(ros::NodeHandle &pnh);
    // Disable copy and assign constructor
    MsckfVio(const MsckfVio &) = delete;
    MsckfVio operator=(const MsckfVio &) = delete;

    // Destructor
    ~MsckfVio() {
        if (align_thread.joinable()) {
            align_thread.join();
        }
    }

    bool initialize();

    void reset();

    typedef boost::shared_ptr<MsckfVio> Ptr;
    typedef boost::shared_ptr<const MsckfVio> ConstPtr;

private:

    /**
     * @brief 管理S-MSCKF中的所有状态，包括IMU相关的状态和多个相机的状态
     */
    struct StateServer
    {
        /// 状态量中IMU相关的状态
        RobotState robot_state;
        /// 状态量中相机相关的状态，是一个map容器，key为相机帧id, 值为CAMState
        CamStateServer cam_states;
        /// 状态量中腿相关的状态
        LegState leg_state;
        /// 所有状态的误差协方差矩阵P
        Eigen::MatrixXd state_cov;
        /// 噪声协方差矩阵
        Eigen::Matrix<double, 12, 12> continuous_noise_cov; /// DISCARD
        /// IMU gyro noise cov
        Eigen::Matrix3d Qg;
        /// IMU acc noise cov
        Eigen::Matrix3d Qa;
        /// IMU gyro bias noise cov
        Eigen::Matrix3d Qbg;
        /// IMU acc bias noise cov
        Eigen::Matrix3d Qba;
        /// IMU contact noise cov
        Eigen::Matrix3d Qc;
        /// Encoder noise cov
        Eigen::Matrix3d Qe;
        /// kinematic additive noise cov
        Eigen::Matrix3d Qkinematic_additive;
    };

    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters();

    /*
     * @brief createRosIO
     *    Create ros publisher and subscirbers.
     */
    bool createRosIO();

    /*
     * @brief imuCallback
     *    Callback function for the imu message.
     * @param msg IMU msg.
     */
    void imuCallback(const sensor_msgs::ImuConstPtr &msg);

    /*
     * @brief featureCallback
     *    Callback function for feature measurements.
     * @param msg Stereo feature measurements.
     */
    void featureCallback(const CameraMeasurementConstPtr &msg);

    /*
     * @brief publish Publish the results of VIO.
     * @param time The time stamp of output msgs.
     */
    void publish(const ros::Time &time);

    /*
     * @brief initializegravityAndBias
     *    Initialize the IMU bias and initial orientation
     *    based on the first few IMU readings.
     */
    void initializeGravityAndBias();

    /*
     * @biref resetCallback
     *    Callback function for the reset service.
     *    Note that this is NOT anytime-reset. This function should
     *    only be called before the sensor suite starts moving.
     *    e.g. while the robot is still on the ground.
     */
    bool resetCallback(std_srvs::Trigger::Request &req,
                        std_srvs::Trigger::Response &res);

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(
        const double &time_bound);
    void processModel(const double &time, const Eigen::Vector3d &m_gyro, const Eigen::Vector3d &m_acc);
    void predictNewState(const double &dt, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc);

    // Measurement update
    void stateAugmentation(const double &time);
    void addFeatureObservations(const CameraMeasurementConstPtr &msg);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const FeatureIDType &feature_id,
                             const StateIDType &cam_state_id,
                             Eigen::Matrix<double, 4, 6> &H_x,
                             Eigen::Matrix<double, 4, 6> &H_c,
                             Eigen::Matrix<double, 4, 3> &H_f,
                             Eigen::Vector4d &r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType &feature_id, const std::vector<StateIDType> &cam_state_ids, Eigen::MatrixXd &H_x, Eigen::VectorXd &r);
    void measurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);
    bool gatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r, const int &dof);
    void removeLostFeatures();
    void findRedundantCamStates(std::vector<StateIDType> &rm_cam_state_ids);
    void pruneCamStateBuffer();
    // Reset the system online if the uncertainty is too large.
    void onlineReset();

    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;

    /// 包含S-MSKCF中的IMU状态和所有相机状态
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;

    /// 包含所有的特征点的map容器，key为特征点id，值为Feature类,
    MapServer map_server;

    /// 储存进入的IMU的消息，有时间同步的作用
    std::vector<sensor_msgs::Imu> imu_msg_buffer;

    /// Indicate if the gravity vector is set， 即是否做了IMU的初始化
    bool is_gravity_set;

    /// 判断是否是第一帧图像，后端在接收到第一帧图像后才开始工作
    bool is_first_img;
    /// 判断是否是第一帧腿的数据，后端腿部分在接收到第一帧数据后开始工作
    bool is_first_leg;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // Ros node handle
    ros::NodeHandle nh;

    // Subscribers and publishers
    ros::Subscriber imu_sub;
    ros::Subscriber feature_sub;
    ros::Publisher odom_pub;
    ros::Publisher feature_pub;
    tf::TransformBroadcaster tf_pub;
    ros::ServiceServer reset_srv;
    ros::Subscriber leica_sub;  // Euroc MH-* 数据集中的3D真实位置
    ros::Subscriber vicon_sub;  // Euroc VH-* 数据集中的6D真实位姿
    ros::Publisher ground_truth_pub;  // 真实轨迹
    ros::Publisher ground_truth_odom_pub; // 用于显示真实姿态
    ros::Publisher vio_path_pub;  // VIO估计的轨迹

    // leg Subscriber
    ros::Subscriber isTouchdown_sub;
    ros::Subscriber qNow_dqNow_TNow_sub;
    ros::Subscriber wheelMotor_fb_sub;
    std::vector<dog_msg::isTouchdown> isTouchdown_buffer;
    std::vector<dog_msg::qNow_dqNow_TNow> qNow_dqNow_TNow_buffer;
    std::vector<dog_msg::wheel_motor_fb> wheelMotor_fb_buffer;
    void isTouchdownCallback(const dog_msg::isTouchdown::ConstPtr &isTouchdown);
    void qNow_dqNow_TNowCallback(const dog_msg::qNow_dqNow_TNow::ConstPtr &qNow_dqNow_TNow);
    void wheelMotor_fbCallback(const dog_msg::wheel_motor_fb::ConstPtr &wheelMotor_fb);

    void InEKF_Propagate(const double &time, const Eigen::Vector3d &m_gyro, const Eigen::Vector3d &m_acc);
    void InEKF_Correct(const Eigen::MatrixXd &Z, const Eigen::MatrixXd &H, const Eigen::MatrixXd &N);
    Eigen::MatrixXd StateTransitionMatrix(const Eigen::Vector3d &w, const Eigen::Vector3d &a, double dt);
    Eigen::MatrixXd DiscreteNoiseMatrix(const Eigen::MatrixXd &Phi, const double dt);
    void RemoveRowAndColumn(Eigen::MatrixXd &M, int index, int remove_dim);

    nav_msgs::Path ground_truth_path;
    nav_msgs::Path vio_path;
    void leicaCallback(const geometry_msgs::PointStampedConstPtr &msg);
    void viconCallback(const geometry_msgs::TransformStampedConstPtr &msg);

    // 计算轨迹对齐转换矩阵
    std::vector<Eigen::Vector3d> path_vio;
    std::vector<Eigen::Vector3d> path_ground_truth;
    bool vio_flag;
    bool ground_truth_flag;
    Eigen::Vector3d vio_point;
    Eigen::Vector3d ground_truth_point;
    Eigen::Matrix4d T_WR = Eigen::Matrix4d::Identity();
    std::thread align_thread;
    std::mutex mtx;
    void alignThreadTask();

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Whether to publish tf or not.
    bool publish_tf;

    // Framte rate of the stereo images. This variable is
    // only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;
    };

typedef MsckfVio::Ptr MsckfVioPtr;
typedef MsckfVio::ConstPtr MsckfVioConstPtr;

} // namespace msckf_vio

#endif
