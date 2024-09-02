/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>
#include <msckf_vio/utils.h>
#include "msckf_vio/webots.hpp"

using namespace std;
using namespace Eigen;

#define REDCOUT(STRING) cout << "\033[31m" << STRING << "\033[m"      // 红色输出
#define GREENCOUT(STRING) cout << "\033[32m" << STRING << "\033[m\n"    // 绿色输出
#define YELLOWCOUT(STRING) cout << "\033[33m" << STRING << "\033[m\n"   // 黄色输出
#define BLUECOUT(STRING) cout << "\033[34m" << STRING << "\033[m\n"     // 蓝色输出
#define PURPLECOUT(STRING) cout << "\033[35m" << STRING << "\033[m\n"   // 紫色输出
#define CYANCOUT(STRING) cout << "\033[36m" << STRING << "\033[m\n"     // 青色输出

#define ROS_DEBUG_STREAM_BLUE(STRING) ROS_DEBUG_STREAM("\033[34m" << STRING << "\033[m")
#define ROS_INFO_STREAM_CYAN(STRING) ROS_INFO_STREAM("\033[36m" << STRING << "\033[m")

bool diagHasNegativeOrNaN(const Eigen::MatrixXd &M, string name)
{
    assert(M.rows() == M.cols());
    // if(M.array().isNaN().any())
    // {
    //     ROS_ERROR_STREAM(name + ": 矩阵含有NaN");
    // }
    // else
    // {
    //     ROS_INFO_STREAM(name + ": 矩阵不含NaN");
    // }
    name = name + " 总维度为 " + to_string(M.rows()) + "x" + to_string(M.cols());
    vector<int> neg_idx;
    for (int i = 0; i < M.rows(); ++i)
    {
        if (M(i, i) < 0)
        {
            neg_idx.push_back(i);
        }
    }
    if(neg_idx.size() > 0)
    {
        REDCOUT(name + ": 矩阵对角线含有负数，索引为 ");
        for (auto idx : neg_idx)
            REDCOUT(idx << " ");
        cout << endl;
        cout << setprecision(2) << name + ": 矩阵对角 R|v|p       \t" << M.topLeftCorner(9, 9).diagonal().transpose() << endl;
        cout << setprecision(2) << name + ": 矩阵对角 d1|d2|d3|d4 \t" << M.block<12, 12>(9, 9).diagonal().transpose() << endl;
        cout << setprecision(2) << name + ": 矩阵对角 bg|ba       \t" << M.block<6, 6>(21, 21).diagonal().transpose() << endl;
        return true;
    }
    ROS_INFO_STREAM(name + ": 矩阵对角线不含负数");
    cout << setprecision(2) << name + ": 矩阵对角 R|v|p       \t" << M.topLeftCorner(9, 9).diagonal().transpose() << endl;
    cout << setprecision(2) << name + ": 矩阵对角 d1|d2|d3|d4 \t" << M.block<12, 12>(9, 9).diagonal().transpose() << endl;
    cout << setprecision(2) << name + ": 矩阵对角 bg|ba       \t" << M.block<6, 6>(21, 21).diagonal().transpose() << endl;
    return false;
}

namespace msckf_vio
{
    // Static member variables in RobotState class.
    StateIDType RobotState::next_id = 0;
    double RobotState::gyro_noise = 0.005;
    double RobotState::acc_noise = 0.05;
    double RobotState::gyro_bias_noise = 0.001;
    double RobotState::acc_bias_noise = 0.01;
    double RobotState::encoder_noise = 0.0174533;
    double RobotState::contact_noise = 0.1;
    double RobotState::kinematics_additive_noise = 0.05;
    Vector3d RobotState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
    Isometry3d RobotState::T_imu_body = Isometry3d::Identity();

    // Static member variables in CAMState class.
    Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

    // Static member variables in Feature class.
    FeatureIDType Feature::next_id = 0;
    double Feature::observation_noise = 0.01;
    Feature::OptimizationConfig Feature::optimization_config;

    map<int, double> MsckfVio::chi_squared_test_table;

    std::string msckf_vio_config_file = "none";
    std::string output_file_path = "none";
    bool use_gatingTest = true;
    bool path_alignment = false;
    bool merge_visual = true;
    bool merge_leg = false;

    WebotsRealState webotsRealState;

    /**
     * @brief MsckfVio构造函数
     * @param pnh Ros节点句柄
     * @param is_gravity_set False 设置未初始化重力
     * @param is_first_img True 设置是第一帧图像
     */
    MsckfVio::MsckfVio(ros::NodeHandle &pnh) : is_gravity_set(false), is_first_img(true), is_first_leg(true), nh(pnh)
    {
        align_thread = std::thread(&MsckfVio::alignThreadTask, this);
        return;
    }

    /**
     * @brief 导入各种参数，包括阈值、传感器误差标准差等
     */
    bool MsckfVio::loadParameters()
    {
        // Frame id
        // 坐标系名字
        nh.param<std::string>("fixed_frame_id", fixed_frame_id, "world");
        nh.param<std::string>("child_frame_id", child_frame_id, "robot");

        nh.param<bool>("publish_tf", publish_tf, true);
        nh.param<double>("frame_rate", frame_rate, 40.0);
        // 用于判断状态是否发散
        nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

        // 判断是否删除状态
        nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
        nh.param<double>("translation_threshold", translation_threshold, 0.4);
        nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

        // Feature optimization parameters
        // 判断点是否能够做三角化，这个参数用的非常精彩
        nh.param<double>(
            "feature/config/translation_threshold",
            Feature::optimization_config.translation_threshold, 0.2);

        // 读取配置文件
        nh.param<std::string>("msckf_vio_config_file", msckf_vio_config_file, "config/default.yaml");
        ROS_INFO("Start loading parameters from %s", msckf_vio_config_file.c_str());
        YAML::Node config;
        try
        {
            config = YAML::LoadFile(msckf_vio_config_file);
        }
        catch (YAML::BadFile &e)
        {
            ROS_ERROR("Failed to open the config file: %s", msckf_vio_config_file.c_str());
        }
        if (config.IsNull())
            ROS_ERROR("config file is empty");

        // 噪声参数
        RobotState::gyro_noise = config["noises"]["IMU"]["gyroscope_std"]
                                     ? config["noises"]["IMU"]["gyroscope_std"].as<double>()
                                     : 0.005;
        RobotState::acc_noise = config["noises"]["IMU"]["accelerometer_std"]
                                    ? config["noises"]["IMU"]["accelerometer_std"].as<double>()
                                    : 0.05;
        RobotState::gyro_bias_noise = config["noises"]["IMU"]["gyroscope_bias_std"]
                                          ? config["noises"]["IMU"]["gyroscope_bias_std"].as<double>()
                                          : 0.001;
        RobotState::acc_bias_noise = config["noises"]["IMU"]["accelerometer_bias_std"]
                                         ? config["noises"]["IMU"]["accelerometer_bias_std"].as<double>()
                                         : 0.01;
        Feature::observation_noise = config["noises"]["Image"]["feature_std"]
                                         ? config["noises"]["Image"]["feature_std"].as<double>()
                                         : 0.035;
        RobotState::encoder_noise = config["noises"]["Leg"]["encoder_std"]
                                        ? config["noises"]["Leg"]["encoder_std"].as<double>()
                                        : 0.0174533;
        RobotState::contact_noise = config["noises"]["Leg"]["contact_std"]
                                        ? config["noises"]["Leg"]["contact_std"].as<double>()
                                        : 0.1;
        RobotState::kinematics_additive_noise = config["noises"]["Leg"]["kinematics_additive_std"]
                                                    ? config["noises"]["Leg"]["kinematics_additive_std"].as<double>()
                                                    : 0.05;
        ROS_INFO_STREAM("==================== noise param ====================");
        ROS_INFO_STREAM("gyro_noise std: " << RobotState::gyro_noise);
        ROS_INFO_STREAM("acc_noise std: " << RobotState::acc_noise);
        ROS_INFO_STREAM("gyro_bias_noise std: " << RobotState::gyro_bias_noise);
        ROS_INFO_STREAM("acc_bias_noise std: " << RobotState::acc_bias_noise);
        ROS_INFO_STREAM("observation_noise std: " << Feature::observation_noise);
        ROS_INFO_STREAM("encoder_noise std: " << RobotState::encoder_noise);
        ROS_INFO_STREAM("contact_noise std: " << RobotState::contact_noise);
        ROS_INFO_STREAM("kinematics_additive_noise std: " << RobotState::kinematics_additive_noise);
        // 方差
        RobotState::gyro_noise *= RobotState::gyro_noise;
        RobotState::acc_noise *= RobotState::acc_noise;
        RobotState::gyro_bias_noise *= RobotState::gyro_bias_noise;
        RobotState::acc_bias_noise *= RobotState::acc_bias_noise;
        Feature::observation_noise *= Feature::observation_noise;
        RobotState::encoder_noise *= RobotState::encoder_noise;
        RobotState::contact_noise *= RobotState::contact_noise;
        RobotState::kinematics_additive_noise *= RobotState::kinematics_additive_noise;

        // 读取初始状态
        const std::vector<double> init_velocity = config["init"]["velocity"]
                                                      ? config["init"]["velocity"].as<std::vector<double>>()
                                                      : std::vector<double>{0.0, 0.0, 0.0};
        Eigen::Vector3d initial_velocity(init_velocity.data());
        state_server.robot_state.setv_GI(initial_velocity);
        const std::vector<double> init_position = config["init"]["position"]
                                                      ? config["init"]["position"].as<std::vector<double>>()
                                                      : std::vector<double>{0.0, 0.0, 0.0};
        Eigen::Vector3d initial_position(init_position.data());
        state_server.robot_state.setp_GI(initial_position);
        ROS_INFO_STREAM("==================== init param ====================");
        ROS_INFO_STREAM("initial velocity: " << initial_velocity.transpose());
        ROS_INFO_STREAM("initial position: " << initial_position.transpose());

        // The initial covariance of orientation and position can be
        // set to 0. But for velocity, bias and extrinsic parameters,
        // there should be nontrivial uncertainty.
        // 初始协方差的赋值（误差状态的协方差）
        // 为什么旋转平移就可以是0？因为在正式开始之前我们通过初始化找好了重力方向，确定了第一帧的位姿
        double gyro_bias_cov, acc_bias_cov, velocity_cov;
        if (!nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25))
            ROS_WARN("Failed to load initial_covariance/velocity param, use default 0.25");
        if (!nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4))
            ROS_WARN("Failed to load initial_covariance/gyro_bias param, use default 1e-4");
        if (!nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2))
            ROS_WARN("Failed to load initial_covariance/acc_bias param, use default 1e-2");

        double extrinsic_rotation_cov, extrinsic_translation_cov;
        if (!nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4))
            ROS_WARN("Failed to load initial_covariance/extrinsic_rotation_cov param, use default 3.0462e-4");
        if (!nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4))
            ROS_WARN("Failed to load initial_covariance/extrinsic_translation_cov param, use default 1e-4");

        double stereo_extrinsic_rotation_cov, stereo_extrinsic_translation_cov;
        if (!nh.param<double>("initial_covariance/stereo_extrinsic_rotation_cov", stereo_extrinsic_rotation_cov, 3.0462e-4))
            ROS_WARN("Failed to load initial_covariance/stereo_extrinsic_rotation_cov param, use default 3.0462e-4");
        if (!nh.param<double>("initial_covariance/stereo_extrinsic_translation_cov", stereo_extrinsic_translation_cov, 1e-4))
            ROS_WARN("Failed to load initial_covariance/stereo_extrinsic_translation_cov param, use default 1e-4");

        // 0~3 旋转 3~6 速度 6~9 位移 9~12 陀螺仪偏置 12~15 加速度计偏置
        // 15~18 左目到IMU的旋转 18~21 左目到IMU的平移
        // 21~24 右目到左目的旋转 24~27 右目到左目的平移
        state_server.state_cov = MatrixXd::Zero(27, 27);
        for (int i = 3; i < 6; ++i)
            state_server.state_cov(i, i) = velocity_cov;
        for (int i = 9; i < 12; ++i)
            state_server.state_cov(i, i) = gyro_bias_cov;
        for (int i = 12; i < 15; ++i)
            state_server.state_cov(i, i) = acc_bias_cov;
        for (int i = 15; i < 18; ++i)
            state_server.state_cov(i, i) = extrinsic_rotation_cov;
        for (int i = 18; i < 21; ++i)
            state_server.state_cov(i, i) = extrinsic_translation_cov;
        for (int i = 21; i < 24; ++i)
            state_server.state_cov(i, i) = stereo_extrinsic_rotation_cov;
        for (int i = 24; i < 27; ++i)
            state_server.state_cov(i, i) = stereo_extrinsic_translation_cov;

        // Transformation offsets between the frames involved.
        // 外参注意还是从左往右那么看
        Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
        Isometry3d T_cam0_imu = T_imu_cam0.inverse();

        // 关于外参状态的初始值设置
        state_server.robot_state.R_cam0_imu = T_cam0_imu.linear();
        state_server.robot_state.t_cam0_imu = T_cam0_imu.translation();

        // 一些其他外参
        CAMState::T_cam0_cam1 =
            utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
        RobotState::T_imu_body =
            utils::getTransformEigen(nh, "T_imu_body").inverse();

        state_server.robot_state.R_cam1_cam0 = CAMState::T_cam0_cam1.inverse().linear();
        state_server.robot_state.t_cam1_cam0 = CAMState::T_cam0_cam1.inverse().translation();

        // Maximum number of camera states to be stored
        nh.param<int>("max_cam_state_size", max_cam_state_size, 30);

        // 自己的一些设置
        merge_visual = config["settings"]["merge_visual"]
                           ? config["settings"]["merge_visual"].as<bool>()
                           : true;
        merge_leg = config["settings"]["merge_leg"]
                        ? config["settings"]["merge_leg"].as<bool>()
                        : false;
        use_gatingTest = config["settings"]["use_gatingTest"]
                             ? config["settings"]["use_gatingTest"].as<bool>()
                             : true;
        path_alignment = config["settings"]["path_alignment"]
                             ? config["settings"]["path_alignment"].as<bool>()
                             : false;
        output_file_path = config["settings"]["output_file_path"]
                               ? config["settings"]["output_file_path"].as<std::string>()
                               : "none";
        ROS_INFO_STREAM("==================== settings param ====================");
        ROS_INFO_STREAM("merge_visual: " << merge_visual);
        ROS_INFO_STREAM("merge_leg: " << merge_leg);
        ROS_INFO_STREAM("use_gatingTest: " << use_gatingTest);
        ROS_INFO_STREAM("path_alignment: " << path_alignment);
        ROS_INFO_STREAM("output_file_path: " << output_file_path);

        double wheel_radius = config["dog_param"]["wheel_radius"]
                                  ? config["dog_param"]["wheel_radius"].as<double>()
                                  : 0.05;
        for (auto &leg : state_server.leg_state.legs)
        {
            leg.wheel_radius = wheel_radius;
        }
        ROS_INFO_STREAM("======================= dog param ======================");
        ROS_INFO_STREAM("wheel_radius: " << wheel_radius);

        // 剩下的都是打印的东西了
        ROS_INFO("===========================================");
        ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
        ROS_INFO("child frame id: %s", child_frame_id.c_str());
        ROS_INFO("publish tf: %d", publish_tf);
        ROS_INFO("frame rate: %f", frame_rate);
        ROS_INFO("position std threshold: %f", position_std_threshold);
        ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
        ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
        ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
        ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
        ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
        ROS_INFO("initial velocity cov: %f", velocity_cov);
        ROS_INFO("initial extrinsic rotation cov: %f", extrinsic_rotation_cov);
        ROS_INFO("initial extrinsic translation cov: %f", extrinsic_translation_cov);
        ROS_INFO("initial stereo extrinsic rotation cov: %f", stereo_extrinsic_rotation_cov);
        ROS_INFO("initial stereo extrinsic translation cov: %f", stereo_extrinsic_translation_cov);

        ROS_INFO_STREAM("T_imu_cam0:\n"
                        << T_imu_cam0.matrix());

        ROS_INFO("max camera state #: %d", max_cam_state_size);
        ROS_INFO("===========================================");
        return true;
    }

    bool MsckfVio::createRosIO()
    {
        /// 1. 发布 "odom", 后端计算出的位姿
        odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
        /// 2. 发布 "feature_point_cloud", 后端计算出的世界系下的点云
        feature_pub = nh.advertise<sensor_msgs::PointCloud2>("feature_point_cloud", 10);

        /// 3. 服务，重置后端
        reset_srv = nh.advertiseService("reset", &MsckfVio::resetCallback, this);

        /// 4. 接收 "imu", 接收IMU数据。@see MsckfVio::imuCallback(const sensor_msgs::ImuConstPtr &msg)
        imu_sub = nh.subscribe("imu", 100, &MsckfVio::imuCallback, this);
        /// 5. 接收 "features", 前端特征点数据。 @see MsckfVio::featureCallback(const CameraMeasurementConstPtr &msg)
        feature_sub = nh.subscribe("features", 40, &MsckfVio::featureCallback, this);

        /// 8. 接收 "/leica/position"， 即Euroc MH-* 数据集中的3D真实位置
        leica_sub = nh.subscribe("/leica/position", 10, &MsckfVio::leicaCallback, this);
        /// 9. 接收 "/vicon/firefly_sbx/firefly_sbx"， 即Euroc VH-* 数据集中的6D真实位姿
        vicon_sub = nh.subscribe("/vicon/firefly_sbx/firefly_sbx", 10, &MsckfVio::viconCallback, this);
        /// 10. 发布 "ground_truth_path"，真实轨迹
        ground_truth_pub = nh.advertise<nav_msgs::Path>("ground_truth_path", 1);
        /// 11. 发布 "ground_truth_odom"，真实姿态
        ground_truth_odom_pub = nh.advertise<nav_msgs::Odometry>("ground_truth_odom", 10);
        /// 12. 发布 "vio_path"，VIO估计的轨迹
        vio_path_pub = nh.advertise<nav_msgs::Path>("vio_path", 1);

        // 腿相关
        isTouchdown_sub = nh.subscribe("/isTouchdown", 100, &MsckfVio::isTouchdownCallback, this);
        qNow_dqNow_TNow_sub = nh.subscribe("/qNow_dqNow_TNow", 100, &MsckfVio::qNow_dqNow_TNowCallback, this);
        wheelMotor_fb_sub = nh.subscribe("/wheelMotor_fb", 100, &MsckfVio::wheelMotor_fbCallback, this);
        WheellegState_sub = nh.subscribe("/wheelleg_state_pub", 100, &MsckfVio::WheellegStateCallback, this);
        return true;
    }

    bool MsckfVio::initialize()
    {
        /// 1. 加载参数 @see MsckfVio::loadParameters()
        if (!loadParameters())
            return false;
        ROS_INFO("Finish loading ROS parameters...");

        /// 2. 设置imu观测的协方差  // DISCARD
        state_server.continuous_noise_cov =
            Matrix<double, 12, 12>::Zero();
        state_server.continuous_noise_cov.block<3, 3>(0, 0) =
            Matrix3d::Identity() * RobotState::gyro_noise;
        state_server.continuous_noise_cov.block<3, 3>(3, 3) =
            Matrix3d::Identity() * RobotState::acc_noise;
        state_server.continuous_noise_cov.block<3, 3>(6, 6) =
            Matrix3d::Identity() * RobotState::gyro_bias_noise;
        state_server.continuous_noise_cov.block<3, 3>(9, 9) =
            Matrix3d::Identity() * RobotState::acc_bias_noise;

        /// 2. 设置各个噪声协方差
        state_server.Qg = Matrix3d::Identity() * RobotState::gyro_noise;
        state_server.Qa = Matrix3d::Identity() * RobotState::acc_noise;
        state_server.Qbg = Matrix3d::Identity() * RobotState::gyro_bias_noise;
        state_server.Qba = Matrix3d::Identity() * RobotState::acc_bias_noise;
        state_server.Qc = Matrix3d::Identity() * RobotState::contact_noise;
        state_server.Qe = Matrix3d::Identity() * RobotState::encoder_noise;
        state_server.Qkinematic_additive = Matrix3d::Identity() * RobotState::kinematics_additive_noise;

        // 卡方检验表，计算自由度从1到99的卡方分布的95％置信水平的分位数
        // Initialize the chi squared test table with confidence level 0.95.
        for (int i = 1; i < 100; ++i)
        {
            boost::math::chi_squared chi_squared_dist(i);
            chi_squared_test_table[i] =
                boost::math::quantile(chi_squared_dist, 0.05);
        }

        /// 3. 创建后端话题的接受与发布 @see MsckfVio::createRosIO()
        if (!createRosIO())
            return false;
        ROS_INFO("Finish creating ROS IO...");

        return true;
    }

    /**
     * @brief 接受IMU数据存入imu_msg_buffer中，并不立刻进行状态递推
     * @note 前200个imu数据需要静止不动进行初始化，如果移动会导致轨迹飘
     */
    void MsckfVio::imuCallback(const sensor_msgs::ImuConstPtr &msg)
    {
        // IMU msgs are pushed backed into a buffer instead of
        // being processed immediately. The IMU msgs are processed
        // when the next image is available, in which way, we can
        // easily handle the transfer delay.
        // 1. 存放imu数据
        imu_msg_buffer.push_back(*msg);

        // 2. 用200个imu数据做静止初始化，不够则不做，初始化后数据并没删，
        // 接收到第一帧图片后会进行了删除
        if (!is_gravity_set)
        {
            if (imu_msg_buffer.size() < 200)
                return;
            // if (imu_msg_buffer.size() < 10) return;
            // imu初始化，200个数据必须都是静止时采集的
            // 这里面没有判断是否成功，也就是一开始如果运动会导致轨迹飘
            initializeGravityAndBias();
            is_gravity_set = true;
        }

        return;
    }

    /**
     * @brief imu初始化，计算陀螺仪偏置，重力方向以及初始姿态，必须都是静止，且不做加速度计的偏置估计
     * @note 为什么要做IMU初始化？因为初始时刻IMU的摆放和机器人的位置是未知的，可能在斜坡上，可能不水平，或者IMU竖直安装，
     *       但是使用IMU积分时是需要减去重力的，因此需要知道重力方向，另外陀螺仪的偏置也是未知的，因此需要初始化
     */
    void MsckfVio::initializeGravityAndBias()
    {
        // TODO: 加上ba的静止初始化，drift中ImuPropagation::InitImuBias
        // Initialize gravity and gyro bias.
        // 1. 求角速度与加速度的和
        Vector3d sum_angular_vel = Vector3d::Zero();
        Vector3d sum_linear_acc = Vector3d::Zero();
        for (const auto &imu_msg : imu_msg_buffer)
        {
            Vector3d angular_vel = Vector3d::Zero();
            Vector3d linear_acc = Vector3d::Zero();

            tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
            tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

            sum_angular_vel += angular_vel;
            sum_linear_acc += linear_acc;
        }

        // 2. 因为假设静止的，因此陀螺仪理论应该都是0，额外读数包括偏置+噪声，但是噪声属于高斯分布
        // 因此这一段相加噪声被认为互相抵消了，所以剩下的均值被认为是陀螺仪的初始偏置
        Vector3d bg = sum_angular_vel / imu_msg_buffer.size();
        state_server.robot_state.setbg(bg);
        // RobotState::gravity =
        //   -sum_linear_acc / imu_msg_buffer.size();
        //  This is the gravity in the IMU frame.
        // 3. 计算重力，忽略加速度计的偏置，剩下的就只有重力了，a_measure = R(a_truth - g) + ba + na
        // 因为假设静止，a_truth = 0, 认为噪声抵消，同时ba为小量，因此a_measure = R(-g)
        Vector3d gravity_imu =
            sum_linear_acc / imu_msg_buffer.size();
        ROS_INFO_STREAM("gravity_imu: " << gravity_imu.transpose());

        // Initialize the initial orientation, so that the estimation
        // is consistent with the inertial frame.
        // 重力的模长就是重力的大小
        double gravity_norm = gravity_imu.norm();
        // 重力本来的方向
        RobotState::gravity = Vector3d(0.0, 0.0, -gravity_norm);
        ROS_INFO_STREAM("gravity: " << RobotState::gravity.transpose());

        // 求出当前imu状态的重力方向与实际重力方向的旋转 R‘， 为什么加负号？因为测量值是R(-g)，而真实值是g
        Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
            gravity_imu, -RobotState::gravity);
        // 得出姿态
        Eigen::Matrix3d R0_i_w = q0_i_w.toRotationMatrix();
        state_server.robot_state.setR_GI(R0_i_w);
        ROS_INFO_STREAM("Init Rotation R_GI: \n"
                        << q0_i_w.toRotationMatrix());

        return;
    }

    /**
     * @brief 重置
     */
    bool MsckfVio::resetCallback(
        std_srvs::Trigger::Request &req,
        std_srvs::Trigger::Response &res)
    {
        ROS_WARN("Start resetting msckf vio...");
        // Temporarily shutdown the subscribers to prevent the
        // state from updating.
        feature_sub.shutdown();
        imu_sub.shutdown();

        // Reset the IMU state.
        RobotState &robot_state = state_server.robot_state;
        Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
        Eigen::Vector3d Zero3 = Eigen::Vector3d::Zero();
        robot_state.time = 0.0;
        robot_state.setR_GI(I3);
        robot_state.setp_GI(Zero3);
        robot_state.setv_GI(Zero3);
        robot_state.setbg(Zero3);
        robot_state.setba(Zero3);

        // Remove all existing camera states.
        state_server.cam_states.clear();

        // Reset the state covariance.
        double gyro_bias_cov, acc_bias_cov, velocity_cov;
        nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25);
        nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4);
        nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2);

        double extrinsic_rotation_cov, extrinsic_translation_cov;
        nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4);
        nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4);

        double stereo_extrinsic_rotation_cov, stereo_extrinsic_translation_cov;
        nh.param<double>("initial_covariance/stereo_extrinsic_rotation_cov", stereo_extrinsic_rotation_cov, 3.0462e-4);
        nh.param<double>("initial_covariance/stereo_extrinsic_translation_cov", stereo_extrinsic_translation_cov, 1e-4);

        // 0~3 旋转 3~6 速度 6~9 位移 9~12 陀螺仪偏置 12~15 加速度计偏置
        // 15~18 左目到IMU的旋转 18~21 左目到IMU的平移
        // 21~24 右目到左目的旋转 24~27 右目到左目的平移
        state_server.state_cov = MatrixXd::Zero(27, 27);
        for (int i = 3; i < 6; ++i)
            state_server.state_cov(i, i) = velocity_cov;
        for (int i = 9; i < 12; ++i)
            state_server.state_cov(i, i) = gyro_bias_cov;
        for (int i = 12; i < 15; ++i)
            state_server.state_cov(i, i) = acc_bias_cov;
        for (int i = 15; i < 18; ++i)
            state_server.state_cov(i, i) = extrinsic_rotation_cov;
        for (int i = 18; i < 21; ++i)
            state_server.state_cov(i, i) = extrinsic_translation_cov;
        for (int i = 21; i < 24; ++i)
            state_server.state_cov(i, i) = stereo_extrinsic_rotation_cov;
        for (int i = 24; i < 27; ++i)
            state_server.state_cov(i, i) = stereo_extrinsic_translation_cov;

        // Clear all exsiting features in the map.
        map_server.clear();

        // Clear the IMU msg buffer.
        imu_msg_buffer.clear();

        // Reset the starting flags.
        is_gravity_set = false;
        is_first_img = true;
        is_first_leg = true;

        // Restart the subscribers.
        imu_sub = nh.subscribe("imu", 100, &MsckfVio::imuCallback, this);
        feature_sub = nh.subscribe("features", 40, &MsckfVio::featureCallback, this);

        res.success = true;
        ROS_WARN("Resetting msckf vio completed...");
        return true;
    }

    /**
     * @brief 后端主要函数，处理新来的数据
     * @param msg 新来的数据, 包括时间戳、点id、和左右目去畸变后归一化坐标
     */
    void MsckfVio::featureCallback(const CameraMeasurementConstPtr &msg)
    {
        int seq = msg->header.seq;
        double cur_time = msg->header.stamp.toSec();
        ROS_INFO_STREAM_CYAN("featureCallback header: seq: " << seq << " time: " << cur_time);
        /// 1. 必须经过imu(重力)初始化才能继续进行
        if (!is_gravity_set)
        {
            ROS_WARN_THROTTLE(1, "Feature Callback: Waiting for the IMU initialization!");
            return;
        }

        /// 2. 接收到第一帧图像后，记录时间，后端开始工作
        if (is_first_img)
        {
            is_first_img = false;
            state_server.robot_state.time = msg->header.stamp.toSec();
            ROS_DEBUG("featureCallback: 初始化完成，第一帧图像...");
        }

        // 调试使用
        diagHasNegativeOrNaN(state_server.state_cov, "featureCallback开头");
        static double max_processing_time = 0.0;
        static int critical_time_cntr = 0;
        double processing_start_time = ros::Time::now().toSec();

        /// 3. 批量IMU积分，取出上次IMU积分时间到当前特征接收时间中的IMU数据进行积分
        /// @see MsckfVio::batchImuProcessing(const double &time_bound)
        ros::Time start_time = ros::Time::now();
        batchImuProcessing(msg->header.stamp.toSec());
        double imu_processing_time = (ros::Time::now() - start_time).toSec();
        diagHasNegativeOrNaN(state_server.state_cov, "batchImuProcessing后");


        /// 4. 状态增广，包括名义状态增广和误差协方差矩阵P的增广，主要是增广新的相机状态
        /// @see MsckfVio::stateAugmentation(const double &time)
        start_time = ros::Time::now();
        stateAugmentation(msg->header.stamp.toSec());
        double state_augmentation_time = (ros::Time::now() - start_time).toSec();
        diagHasNegativeOrNaN(state_server.state_cov, "stateAugmentation后");

        /// 5. 向map_server中添加新的特征，和旧特征在新相机帧上的观测
        start_time = ros::Time::now();
        addFeatureObservations(msg);
        double add_observations_time = (ros::Time::now() - start_time).toSec();
        diagHasNegativeOrNaN(state_server.state_cov, "addFeatureObservations后");

        // Perform measurement update if necessary.
        // 5. 使用不再跟踪上的点来更新
        start_time = ros::Time::now();
        removeLostFeatures();
        double remove_lost_features_time = (ros::Time::now() - start_time).toSec();
        diagHasNegativeOrNaN(state_server.state_cov, "removeLostFeatures后");

        // 6. 当cam状态数达到最大值时，挑出若干cam状态待删除
        // 并基于能被2帧以上这些cam观测到的feature进行MSCKF测量更新
        start_time = ros::Time::now();
        pruneCamStateBuffer();
        double prune_cam_states_time = (ros::Time::now() - start_time).toSec();
        diagHasNegativeOrNaN(state_server.state_cov, "pruneCamStateBuffer后");

        // Publish the odometry.
        // 7. 发布位姿
        start_time = ros::Time::now();
        publish(msg->header.stamp);
        double publish_time = (ros::Time::now() - start_time).toSec();
        diagHasNegativeOrNaN(state_server.state_cov, "publish后");

        // Reset the system if necessary.
        // 8. 根据IMU状态位置协方差判断是否重置整个系统
        onlineReset();

        // 一些调试数据
        double processing_end_time = ros::Time::now().toSec();
        double processing_time =
            processing_end_time - processing_start_time;
        if (processing_time > 1.0 / frame_rate)
        {
            ++critical_time_cntr;
            ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
                    processing_time, critical_time_cntr);
            ROS_INFO("IMU processing time: %f/%f",
                    imu_processing_time, imu_processing_time / processing_time);
            ROS_INFO("State augmentation time: %f/%f",
                    state_augmentation_time, state_augmentation_time / processing_time);
            ROS_INFO("Add observations time: %f/%f",
                    add_observations_time, add_observations_time / processing_time);
            ROS_INFO("Remove lost features time: %f/%f",
                    remove_lost_features_time, remove_lost_features_time / processing_time);
            ROS_INFO("Remove camera states time: %f/%f",
                    prune_cam_states_time, prune_cam_states_time / processing_time);
            ROS_INFO("Publish time: %f/%f\n",
                    publish_time, publish_time / processing_time);
        }
        // ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
        //          processing_time, critical_time_cntr);
        // ROS_INFO("IMU processing time: %f/%f",
        //          imu_processing_time, imu_processing_time / processing_time);
        // ROS_INFO("State augmentation time: %f/%f",
        //          state_augmentation_time, state_augmentation_time / processing_time);
        // ROS_INFO("Add observations time: %f/%f",
        //          add_observations_time, add_observations_time / processing_time);
        // ROS_INFO("Remove lost features time: %f/%f",
        //          remove_lost_features_time, remove_lost_features_time / processing_time);
        // ROS_INFO("Remove camera states time: %f/%f",
        //          prune_cam_states_time, prune_cam_states_time / processing_time);
        // ROS_INFO("Publish time: %f/%f\n",
        //          publish_time, publish_time / processing_time);
        cout << "旋转协方差: \n" << state_server.state_cov.block<3, 3>(0, 0) << endl;
        cout << "速度协方差: \n"
             << state_server.state_cov.block<3, 3>(3, 3) << endl;
        cout << "位置协方差: \n"
             << state_server.state_cov.block<3, 3>(6, 6) << endl;
        cout << "腿1协方差: \n" << state_server.state_cov.block<3, 3>(9, 9) << endl;
        cout << "腿2协方差: \n" << state_server.state_cov.block<3, 3>(12, 12) << endl;
        cout << "腿3协方差: \n" << state_server.state_cov.block<3, 3>(15, 15) << endl;
        cout << "腿4协方差: \n" << state_server.state_cov.block<3, 3>(18, 18) << endl;

        diagHasNegativeOrNaN(state_server.state_cov, "featureCallback末尾");
        ROS_INFO_STREAM_CYAN("featureCallback end...");
        return;
    }

    /**
     * @brief imu积分，批量处理imu数据
     * @param  time_bound 从state_server.robot_state.time上次处理到这个时间
     */
    void MsckfVio::batchImuProcessing(const double &time_bound)
    {
        int used_imu_msg_cntr = 0;

        /// 1. 遍历imu_msg_buffer中的所有imu数据，找到state_server.robot_state.time到
        /// 当前前端特征时间之间的imu数据，然后进行状态递推
        /// @see MsckfVio::processModel(const double &time, const Eigen::Vector3d &m_gyro, const Eigen::Vector3d &m_acc)
        for (const auto &imu_msg : imu_msg_buffer)
        {
            double imu_time = imu_msg.header.stamp.toSec();
            // 小于，说明这个数据比较旧，因为state_server.robot_state.time代表已经处理过的imu数据的时间
            if (imu_time < state_server.robot_state.time)
            {
                ++used_imu_msg_cntr;
                continue;
            }
            // 超过的供下次使用
            if (imu_time > time_bound)
                break;

            // Convert the msgs.
            Vector3d m_gyro, m_acc;
            tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
            tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

            // 递推位姿，核心函数
            processModel(imu_time, m_gyro, m_acc);
            ++used_imu_msg_cntr;
        }

        /// 2. 更新IMU状态的id  state_server.robot_state.id, 相机状态id也根据这个赋值
        state_server.robot_state.id = RobotState::next_id++;

        /// 3. 从imu_msg_buffer中删除已经使用过的数据
        imu_msg_buffer.erase(
            imu_msg_buffer.begin(),
            imu_msg_buffer.begin() + used_imu_msg_cntr);
        return;
    }

    /**
     * @brief 对一帧IMU数据进行状态预估，包括名义状态递推，误差协方差矩阵的预估
     * @param  time IMU数据的时间戳
     * @param  m_gyro 角速度
     * @param  m_acc 加速度
     */
    void MsckfVio::processModel(
        const double &time, const Vector3d &m_gyro, const Vector3d &m_acc)
    {
        /// 1. 引用的方式取出imu状态
        RobotState &robot_state = state_server.robot_state;

        /// 2. 角速度和加速度减去偏置，计算dt
        Vector3d gyro = m_gyro - robot_state.getbg();
        Vector3d acc = m_acc - robot_state.getba(); // acc_bias 初始值是0
        double dtime = time - robot_state.time;

        Matrix3d R = robot_state.getR_GI();
        Vector3d v = robot_state.getv_GI();
        Vector3d p = robot_state.getp_GI();
        MatrixXd X_i = robot_state.getX_i();
        int dimP_i = robot_state.dimP_i();
        int dimX_i = robot_state.dimX_i();
        int dimX_frak = robot_state.dimX_frak();

        MatrixXd F = MatrixXd::Zero(dimP_i + 12, dimP_i + 12); // IMU+腿+bias  + 4个外参
        MatrixXd G = MatrixXd::Zero(dimP_i + 12, dimP_i - 3);  // 减去 p 所在列 和 4个外参

        F.block<3, 3>(3, 0) = skewSymmetric(RobotState::gravity);
        F.block<3, 3>(6, 3) = Matrix3d::Identity();
        F.block<3, 3>(0, dimP_i - dimX_frak) = -R;
        F.block<3, 3>(3, dimP_i - dimX_frak + 3) = -R;
        for (int i = 3; i < dimX_i; ++i)
        {
            F.block<3, 3>(3 * i - 6, dimP_i - dimX_frak) = -skewSymmetric(X_i.block<3, 1>(0, i)) * R;
        }

        MatrixXd Adj = Adjoint_SEK3(X_i);
        G.block(0, 0, dimP_i - dimX_frak, 6) = Adj.block(0, 0, dimP_i - dimX_frak, 6);
        G.block(0, 6, dimP_i - dimX_frak, dimP_i - dimX_frak - 9) = Adj.block(0, 9, dimP_i - dimX_frak, dimP_i - dimX_frak - 9);
        G.block<3, 3>(dimP_i - dimX_frak, dimP_i - dimX_frak - 3) = Matrix3d::Identity();
        G.block<3, 3>(dimP_i - dimX_frak + 3, dimP_i - dimX_frak) = Matrix3d::Identity();

        MatrixXd Fdt = F * dtime;
        MatrixXd Fdt2 = Fdt * Fdt;
        MatrixXd Fdt3 = Fdt2 * Fdt;
        MatrixXd Phi = MatrixXd::Identity(dimP_i + 12, dimP_i + 12) +
                       Fdt + (1.0 / 2.0) * Fdt2 + (1.0 / 6.0) * Fdt3;

        /// 4. 四阶龙格库塔积分预测名义状态，旋转，速度，位置
        /// @see MsckfVio::predictNewState(const double &dt, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc)
        predictNewState(dtime, gyro, acc);

        /// 6. 使用OC后的Phi阵计算过程噪声协方差矩阵Q, 见笔记pdf中《误差状态转移矩阵和过程噪声协方差矩阵》
        MatrixXd Cov = MatrixXd::Zero(dimP_i - 3, dimP_i - 3);
        Cov.block<3, 3>(0, 0) = state_server.Qg;
        Cov.block<3, 3>(3, 3) = state_server.Qa;
        Cov.block<3, 3>(dimP_i - dimX_frak - 3, dimP_i - dimX_frak - 3) = state_server.Qbg;
        Cov.block<3, 3>(dimP_i - dimX_frak, dimP_i - dimX_frak) = state_server.Qba;
        // TODO: 乘FkR?
        for (int i = 5; i < dimX_i; ++i)
            Cov.block<3, 3>(3 * (i - 5) + 6, 3 * (i - 5) + 6) = state_server.Qc;
        MatrixXd PhiG = Phi * G;
        MatrixXd Q = PhiG * Cov * PhiG.transpose() * dtime;

        /// 7. 预测系统误差状态协方差矩阵P，如果有相机状态量，那么也更新imu状态量与相机状态量交叉的部分
        /// 见笔记pdf中《预测系统状态协方差矩阵P》
        state_server.state_cov.block(0, 0, dimP_i + 12, dimP_i + 12) =
            Phi * state_server.state_cov.block(0, 0, dimP_i + 12, dimP_i + 12) * Phi.transpose() + Q;
        if (state_server.cam_states.size() > 0)
        {
            state_server.state_cov.block(0, dimP_i + 12, dimP_i + 12, state_server.state_cov.cols() - dimP_i - 12) =
                Phi * state_server.state_cov.block(0, dimP_i + 12, dimP_i + 12, state_server.state_cov.cols() - dimP_i - 12);
            state_server.state_cov.block(dimP_i + 12, 0, state_server.state_cov.rows() - dimP_i - 12, dimP_i + 12) =
                state_server.state_cov.block(dimP_i + 12, 0, state_server.state_cov.rows() - dimP_i - 12, dimP_i + 12) * Phi.transpose();
        }

        /// 8. 强制对称，因为协方差矩阵就是对称的
        MatrixXd state_cov_fixed =
            0.5 * (state_server.state_cov + state_server.state_cov.transpose());
        state_server.state_cov = state_cov_fixed;

        /// 10. 更新imu状态的时间state_server.robot_state.time
        state_server.robot_state.time = time;
        return;
    }

    /**
     * @brief 四阶龙格库塔积对IMU状态递推，见笔记pdf中《名义状态递推》
     * @param  dt 相对上一个数据的间隔时间
     * @param  gyro 角速度减去偏置后的
     * @param  acc 加速度减去偏置后的
     */
    void MsckfVio::predictNewState(
        const double &dt, const Vector3d &gyro, const Vector3d &acc)
    {
        Matrix3d R = state_server.robot_state.getR_GI();
        Vector3d v = state_server.robot_state.getv_GI();
        Vector3d p = state_server.robot_state.getp_GI();

        Matrix3d dR_dt = R * Sophus::SO3d::exp(gyro * dt).matrix();
        Matrix3d dR_dt2 = R * Sophus::SO3d::exp(gyro * dt / 2.0).matrix();

        // k1 = f(tn, yn)
        Vector3d k1_v_dot = R * acc + RobotState::gravity;
        Vector3d k1_p_dot = v;

        // k2 = f(tn+dt/2, yn+k1*dt/2)
        // 这里的4阶LK法用了匀加速度假设，即认为前一时刻的加速度和当前时刻相等
        Vector3d k1_v = v + k1_v_dot * dt / 2.0;
        Vector3d k2_v_dot = dR_dt2 * acc + RobotState::gravity;
        Vector3d k2_p_dot = k1_v;

        // k3 = f(tn+dt/2, yn+k2*dt/2)
        Vector3d k2_v = v + k2_v_dot * dt / 2.0;
        Vector3d k3_v_dot = dR_dt2 * acc + RobotState::gravity;
        Vector3d k3_p_dot = k2_v;

        // k4 = f(tn+dt, yn+k3*dt)
        Vector3d k3_v = v + k3_v_dot * dt;
        Vector3d k4_v_dot = dR_dt * acc + RobotState::gravity;
        Vector3d k4_p_dot = k3_v;

        // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        v = v + dt / 6.0 * (k1_v_dot + 2.0 * k2_v_dot + 2.0 * k3_v_dot + k4_v_dot);
        p = p + dt / 6.0 * (k1_p_dot + 2.0 * k2_p_dot + 2.0 * k3_p_dot + k4_p_dot);
        state_server.robot_state.setR_GI(dR_dt);
        state_server.robot_state.setv_GI(v);
        state_server.robot_state.setp_GI(p);
        return;
    }

    /**
     * @brief 状态增广，向状态中增加新的相机状态，同时增广误差协方差矩阵P
     * @param  time 图片的时间戳
     */
    void MsckfVio::stateAugmentation(const double &time)
    {
        const Matrix3d &R_c_i = state_server.robot_state.R_cam0_imu; // Ric
        const Vector3d &p_c_i = state_server.robot_state.t_cam0_imu; // pic

        // 1.2 取出imu旋转平移，按照外参，将这个时刻cam0的位姿算出来
        Matrix3d R_i_w = state_server.robot_state.getR_GI(); // Rwi
        Vector3d p_i_w = state_server.robot_state.getp_GI(); // pwi
        Matrix3d R_c_w = R_i_w * R_c_i;                      // Rwc = Rwi * Ric
        Vector3d p_c_w = p_i_w + R_i_w * p_c_i;              // pwc = pwi + Rwi * pic

        /// 2. 注册新的相机状态到状态库state_server中,
        /// 包括id(使用此时的imu状态id作为该帧相机的id), 时间戳，位姿
        /// QUERY 以及用于OC的零空间(第一次相机帧估计的数据)
        state_server.cam_states[state_server.robot_state.id] =
            CAMState(state_server.robot_state.id);
        CAMState &cam_state = state_server.cam_states[state_server.robot_state.id];

        cam_state.time = time;
        cam_state.R_G_Cam0 = R_c_w;
        cam_state.p_G_Cam0 = p_c_w;

        // TODO 这里的求偏导总感觉不对，有可能就是这里导致了InEKF的不稳定
        // Update the covariance matrix of the state.
        int dimP_i = state_server.robot_state.dimP_i();
        int dimX_i = state_server.robot_state.dimX_i();
        int dimX_frak = state_server.robot_state.dimX_frak();
        MatrixXd J = MatrixXd::Zero(6, dimP_i + 12);
        J.block<3, 3>(0, 0) = Matrix3d::Identity();
        J.block<3, 3>(0, dimP_i) = R_i_w;
        J.block<3, 3>(3, 6) = Matrix3d::Identity();
        J.block<3, 3>(3, dimP_i) = skewSymmetric(p_i_w) * R_i_w;
        J.block<3, 3>(3, dimP_i + 3) = R_i_w;

        /// 4. 增广误差协方差矩阵P，见笔记pdf中《误差协方差矩阵增广》
        // 简单地说就是原来的协方差是 21 + 6n 维的，现在新来了一个伙计，维度要扩了
        // 并且对应位置的值要根据雅可比跟这个时刻（也就是最新时刻）的imu协方差计算
        // 4.1 扩展矩阵大小 conservativeResize函数不改变原矩阵对应位置的数值
        // Resize the state covariance matrix.
        size_t old_rows = state_server.state_cov.rows();
        size_t old_cols = state_server.state_cov.cols();
        state_server.state_cov.conservativeResize(old_rows + 6, old_cols + 6);

        // imu的协方差矩阵
        const MatrixXd &P11 = state_server.state_cov.block(0, 0, dimP_i + 12, dimP_i + 12);

        // imu相对于各个相机状态量的协方差矩阵（不包括最新的）
        const MatrixXd &P12 = state_server.state_cov.block(0, dimP_i + 12, dimP_i + 12, old_cols - dimP_i - 12);

        // 4.2 计算协方差矩阵
        // 左下角
        state_server.state_cov.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;

        // 右上角
        state_server.state_cov.block(0, old_cols, old_rows, 6) =
            state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();

        // 右下角，关于相机部分的J都是0所以省略了
        state_server.state_cov.block<6, 6>(old_rows, old_cols) = J * P11 * J.transpose();

        /// 5. 进行强制对称
        MatrixXd state_cov_fixed =
            0.5 * (state_server.state_cov + state_server.state_cov.transpose());
        state_server.state_cov = state_cov_fixed;

        return;
    }

    /**
     * @brief 向map_server中添加新的特征，和旧特征在新相机帧上的观测
     * @param  msg 前端发来的当前帧的所有特征以及对应的归一化坐标
     */
    void MsckfVio::addFeatureObservations(
        const CameraMeasurementConstPtr &msg)
    {
        /// 1. 取出当前imu状态的id作为当前相机帧的id
        StateIDType state_id = state_server.robot_state.id;

        /// 2. 获取当前地图内特征点的数量
        int curr_feature_num = map_server.size();
        int tracked_feature_num = 0;

        /// 3. 若此次前端传来的特征点在地图中没有，那么就添加新的特征点
        /// 否则就更新旧特征点的观测
        for (const auto &feature : msg->features)
        {
            if (map_server.find(feature.id) == map_server.end())
            {
                // This is a new feature.
                map_server[feature.id] = Feature(feature.id);
                map_server[feature.id].observations[state_id] =
                    Vector4d(feature.u0, feature.v0,
                             feature.u1, feature.v1);
            }
            else
            {
                // This is an old feature.
                map_server[feature.id].observations[state_id] =
                    Vector4d(feature.u0, feature.v0,
                             feature.u1, feature.v1);
                ++tracked_feature_num;
            }
        }

        /// 4. 计算跟踪率。当前帧前端进来的所有点中，有多少是已经在地图中的
        tracking_rate =
            static_cast<double>(tracked_feature_num) /
            static_cast<double>(curr_feature_num);

        return;
    }

    /**
     * @brief 使用不再跟踪上的点来更新
     */
    void MsckfVio::removeLostFeatures()
    {
        // Remove the features that lost track.
        // BTW, find the size the final Jacobian matrix and residual vector.
        int jacobian_row_size = 0;
        // FeatureIDType 这是个long long int 嗯。。。。直接当作int理解吧
        vector<FeatureIDType> invalid_feature_ids(0);   // 无效点，最后要删的
        vector<FeatureIDType> processed_feature_ids(0); // 待参与更新的点，用完也被无情的删掉

        // 遍历所有特征管理里面的点，包括新进来的
        for (auto iter = map_server.begin();
             iter != map_server.end(); ++iter)
        {
            // Rename the feature to be checked.
            // 引用，改变feature相当于改变iter->second，类似于指针的效果
            auto &feature = iter->second;

            // Pass the features that are still being tracked.
            // 1. 这个点被当前状态观测到，说明这个点后面还有可能被跟踪
            // 跳过这些点
            if (feature.observations.find(state_server.robot_state.id) !=
                feature.observations.end())
                continue;

            // 2. 跟踪小于3帧的点，认为是质量不高的点
            // 也好理解，三角化起码要两个观测，但是只有两个没有其他观测来验证
            if (feature.observations.size() < 3)
            {
                invalid_feature_ids.push_back(feature.id);
                continue;
            }

            // Check if the feature can be initialized if it
            // has not been.
            // 3. 如果这个特征没有被初始化，尝试去初始化
            // 初始化就是三角化
            if (!feature.is_initialized)
            {
                // 3.1 看看运动是否足够，没有足够视差或者平移小旋转多这种不符合三角化
                // 所以就不要这些点了
                if (!feature.checkMotion(state_server.cam_states))
                {
                    invalid_feature_ids.push_back(feature.id);
                    continue;
                }
                else
                {
                    // 3.3 尝试三角化，失败也不要了
                    if (!feature.initializePosition(state_server.cam_states, state_server.robot_state))
                    {
                        invalid_feature_ids.push_back(feature.id);
                        continue;
                    }
                }
            }

            // 4. 到这里表示这个点能用于更新，所以准备下一步计算
            // 一个观测代表一帧，一帧有左右两个观测
            // 也就是算重投影误差时维度将会是4 * feature.observations.size()
            // 这里为什么减3下面会提到
            jacobian_row_size += 4 * feature.observations.size() - 3;
            // 接下来要参与优化的点加入到这个变量中
            processed_feature_ids.push_back(feature.id);
        }

        // Remove the features that do not have enough measurements.
        // 5. 删掉非法点
        for (const auto &feature_id : invalid_feature_ids)
            map_server.erase(feature_id);

        // Return if there is no lost feature to be processed.
        if (processed_feature_ids.size() == 0)
            return;

        // 准备好误差相对于状态量的雅可比
        int dimP_i = state_server.robot_state.dimP_i();
        MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                      dimP_i + 12 + 6 * state_server.cam_states.size());
        VectorXd r = VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;
        BLUECOUT("dimP_i: " << dimP_i << " state_server.cam_states.size(): " << state_server.cam_states.size());

        // Process the features which lose track.
        // 6. 处理特征点
        for (const auto &feature_id : processed_feature_ids)
        {
            auto &feature = map_server[feature_id];

            vector<StateIDType> cam_state_ids(0);
            for (const auto &measurement : feature.observations)
                cam_state_ids.push_back(measurement.first);

            MatrixXd H_xj;
            VectorXd r_j;
            // 6.1 计算雅可比，计算重投影误差
            featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

            // 6.2 卡方检验，剔除错误点，并不是所有点都用
            if (gatingTest(H_xj, r_j, cam_state_ids.size() - 1))
            {
                H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
                r.segment(stack_cntr, r_j.rows()) = r_j;
                stack_cntr += H_xj.rows();
            }

            // Put an upper bound on the row size of measurement Jacobian,
            // which helps guarantee the executation time.
            // 限制最大更新量
            if (stack_cntr > 1500)
                break;
        }

        // resize成实际大小
        H_x.conservativeResize(stack_cntr, H_x.cols());
        r.conservativeResize(stack_cntr);

        // Perform the measurement update step.
        // 7. 使用误差及雅可比更新状态processing_time

        measurementUpdate(H_x, r);

        // Remove all processed features from the map.
        // 8. 删除用完的点
        for (const auto &feature_id : processed_feature_ids)
            map_server.erase(feature_id);

        return;
    }

    /**
     * @brief 更新
     * @param  H 雅可比
     * @param  r 误差
     */
    void MsckfVio::measurementUpdate(
        const MatrixXd &H, const VectorXd &r)
    {
        if (!merge_visual)
        {
            ROS_DEBUG_THROTTLE(10.0, "unmerged visual");
            return;
        }
        ROS_DEBUG_STREAM_BLUE("measurementUpdate In");

        int dimP_i = state_server.robot_state.dimP_i();
        int dimX_i = state_server.robot_state.dimX_i();
        int dimX_frak = state_server.robot_state.dimX_frak();

        if (H.rows() == 0 || r.rows() == 0)
            return;

        // Decompose the final Jacobian matrix to reduce computational
        // complexity as in Equation (28), (29).
        MatrixXd H_thin;
        VectorXd r_thin;

        if (H.rows() > H.cols())
        {
            // Convert H to a sparse matrix.
            SparseMatrix<double> H_sparse = H.sparseView();

            // Perform QR decompostion on H_sparse.
            // 利用H矩阵稀疏性，QR分解
            // 这段结合零空间投影一起理解，主要作用就是降低计算量
            SPQR<SparseMatrix<double>> spqr_helper;
            spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
            spqr_helper.compute(H_sparse);

            MatrixXd H_temp;
            VectorXd r_temp;
            (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
            (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

            H_thin = H_temp.topRows(dimP_i + 12 + state_server.cam_states.size() * 6);
            r_thin = r_temp.head(dimP_i + 12 + state_server.cam_states.size() * 6);
        }
        else
        {
            H_thin = H;
            r_thin = r;
        }

        // 2. 标准的卡尔曼计算过程
        // Compute the Kalman gain.
        const MatrixXd &P = state_server.state_cov;
        MatrixXd S = H_thin * P * H_thin.transpose() +
                     Feature::observation_noise * MatrixXd::Identity(
                                                      H_thin.rows(), H_thin.rows());
        MatrixXd K_transpose = S.ldlt().solve(H_thin * P);
        MatrixXd K = K_transpose.transpose();

        // Compute the error of the state.
        VectorXd delta_x = K * r_thin;

        // Update the IMU state.
        const VectorXd &delta_x_imu = delta_x.head(dimP_i + 12);

        const MatrixXd dT_imu = Exp_SEK3(delta_x_imu.head(dimP_i - dimX_frak));
        const Matrix3d dR_imu = dT_imu.block<3, 3>(0, 0);
        Eigen::Matrix3d R_GI_pred = dR_imu * state_server.robot_state.getR_GI();
        Eigen::Vector3d v_GI_pred = dR_imu * state_server.robot_state.getv_GI() + dT_imu.block<3, 1>(0, 3);
        Eigen::Vector3d p_GI_pred = dR_imu * state_server.robot_state.getp_GI() + dT_imu.block<3, 1>(0, 4);
        state_server.robot_state.setR_GI(R_GI_pred);
        state_server.robot_state.setv_GI(v_GI_pred);
        state_server.robot_state.setp_GI(p_GI_pred);
        // TODO: 这里是否要更新腿？dT_imu中还有腿，可以打印一下看看增量是多少

        Eigen::Vector3d bg_pred = state_server.robot_state.getbg() + delta_x_imu.segment<3>(dimP_i - dimX_frak);
        Eigen::Vector3d ba_pred = state_server.robot_state.getba() + delta_x_imu.segment<3>(dimP_i - dimX_frak + 3);
        state_server.robot_state.setbg(bg_pred);
        state_server.robot_state.setba(ba_pred);

        const Matrix4d dT_ext = Exp_SEK3(delta_x_imu.segment<6>(dimP_i));
        const Matrix3d dR_ext = dT_ext.block<3, 3>(0, 0);
        state_server.robot_state.R_cam0_imu = dR_ext * state_server.robot_state.R_cam0_imu;
        state_server.robot_state.t_cam0_imu = dR_ext * state_server.robot_state.t_cam0_imu + dT_ext.block<3, 1>(0, 3);

        const Matrix4d dT_stereo = Exp_SEK3(delta_x_imu.segment<6>(dimP_i + 6));
        const Matrix3d dR_stereo = dT_stereo.block<3, 3>(0, 0);
        state_server.robot_state.R_cam1_cam0 = dR_stereo * state_server.robot_state.R_cam1_cam0;
        state_server.robot_state.t_cam1_cam0 = dR_stereo * state_server.robot_state.t_cam1_cam0 + dT_stereo.block<3, 1>(0, 3);

        // 更新相机姿态
        auto cam_state_iter = state_server.cam_states.begin();
        for (int i = 0; i < state_server.cam_states.size(); ++i, ++cam_state_iter)
        {
            const VectorXd &delta_x_cam = delta_x.segment<6>(dimP_i + 12 + i * 6);
            const Matrix4d dT_cam = Exp_SE3(delta_x_cam.head<3>(),
                                            delta_x_cam.tail<3>());
            const Matrix3d dR_cam = dT_cam.block<3, 3>(0, 0);
            cam_state_iter->second.R_G_Cam0 = dR_cam * cam_state_iter->second.R_G_Cam0;
            cam_state_iter->second.p_G_Cam0 = dR_cam * cam_state_iter->second.p_G_Cam0 + dT_cam.block<3, 1>(0, 3);
        }

        // 4. 更新协方差
        MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
        // state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
        //   K*K.transpose()*Feature::observation_noise;
        state_server.state_cov = I_KH * state_server.state_cov;

        // Fix the covariance to be symmetric
        MatrixXd state_cov_fixed =
            0.5 * (state_server.state_cov + state_server.state_cov.transpose());
        state_server.state_cov = state_cov_fixed;

        // ofstream out("/home/speike/VSLAM/MSCKF_vio_bak_ws/state_cov.txt", ios::app);
        // if(out.is_open())
        // {
        //     out << setprecision(1) << state_server.state_cov << endl << endl;
        //     out.close();
        // }

        // cout << setprecision(15) << "T_cam0_cam1: \n" << CAMState::T_cam0_cam1.matrix() << endl;
        // cout << setprecision(15) << "T_cam0_cam1.inverse: \n" << CAMState::T_cam0_cam1.inverse().matrix() << endl;
        // cout << setprecision(15) << "R_cam1_cam0: \n" << state_server.robot_state.R_cam1_cam0 << endl;
        // cout << setprecision(15) << "t_cam1_cam0: " << state_server.robot_state.t_cam1_cam0.transpose() << endl;

        // Eigen::Vector4d q(0.670, 0.122, 0.472, 0.560);     // JPL x y z w
        // Eigen::Quaterniond q1(0.560, 0.670, 0.122, 0.472); // Harmilton w x y z
        // cout << setprecision(10) << q1.coeffs().transpose() << endl; // 输出的是 x y z w
        // cout << setprecision(10) << q1.normalized().toRotationMatrix() << endl;
        // cout << setprecision(10) << quaternionToRotation(q) << endl;
        // cout << "----------1. JPL转为旋转矩阵为R_LG, Harmilton转为旋转矩阵为R_GL, 相差一个转置-----------" << endl;
        // Eigen::Matrix3d R = q1.normalized().toRotationMatrix();
        // cout << setprecision(10) << Eigen::Quaterniond(R).normalized().coeffs().transpose() << endl;
        // cout << setprecision(10) << rotationToQuaternion(R).transpose() << endl;
        // cout << "----------2. 相同旋转矩阵转为JPL和Harmilton相差一个共轭-------------------------------" << endl;
        // Eigen::Vector3d small_delta(0.1, 0.2, 0.3);
        // Eigen::Vector4d dq = smallAngleQuaternion(small_delta);
        // Eigen::Matrix3d dR = Sophus::SO3d::exp(small_delta).matrix();
        // cout << setprecision(10) << dq.transpose() << endl;
        // cout << setprecision(10) << Eigen::Quaterniond(dR).normalized().coeffs().transpose() << endl;
        // cout << "----------3. 两种小旋转结果一样-----------------------------------------------------" << endl;
        // cout << setprecision(10) << quaternionMultiplication(dq, q).transpose() << endl;
        // cout << setprecision(10) << Eigen::Quaterniond(dR * R).normalized().coeffs().transpose() << endl;
        // cout << setprecision(10) << Eigen::Quaterniond(R * dR).normalized().coeffs().transpose() << endl;
        // cout << setprecision(10)
        //      << (Eigen::Quaterniond(dq(3), dq(0), dq(1), dq(2)) * q1).normalized().coeffs().transpose() << endl;
        // cout << setprecision(10)
        //      << (q1 * Eigen::Quaterniond(dq(3), dq(0), dq(1), dq(2))).normalized().coeffs().transpose() << endl;
        // cout << "----------4. JPL中左乘等于Harmiton中的右乘------------------------------------------" << endl;
        // Eigen::Matrix3d R_GC = quaternionToRotation(q).transpose();
        // Eigen::Vector4d q_CG = q;
        // Eigen::Vector4d q_CG_new = quaternionMultiplication(dq, q);
        // Eigen::Matrix3d R_GC_new = R_GC * dR;
        // Eigen::Matrix3d R_GC_new2 = dR * R_GC;
        // cout << setprecision(10) << q_CG_new.transpose() << endl;
        // cout << setprecision(10) << rotationToQuaternion(R_GC_new.transpose()).transpose() << endl; // JPL形式的q_CG
        // cout << setprecision(10) << Eigen::Quaterniond(R_GC_new).normalized().coeffs().transpose() << endl; // Har形式的q_GC
        // cout << setprecision(10) << rotationToQuaternion(R_GC_new2.transpose()).transpose() << endl; // JPL形式的q_CG
        // cout << setprecision(10) << Eigen::Quaterniond(R_GC_new2).normalized().coeffs().transpose() << endl; // Har形式的q_GC
        // cout << "----------------------------------------------------" << endl;
        ROS_DEBUG_STREAM_BLUE("measurementUpdate Out");
        return;
    }

    /**
     * @brief 计算一个路标点的雅可比
     * @param  feature_id 路标点id
     * @param  cam_state_ids 这个点对应的所有的相机状态id
     * @param  H_x 雅可比
     * @param  r 误差
     */
    void MsckfVio::featureJacobian(
        const FeatureIDType &feature_id,
        const std::vector<StateIDType> &cam_state_ids,
        MatrixXd &H_x, VectorXd &r)
    {
        // 取出特征
        const auto &feature = map_server[feature_id];

        // Check how many camera states in the provided camera
        // id camera has actually seen this feature.
        // 1. 统计有效观测的相机状态，因为对应的个别状态有可能被滑走了
        vector<StateIDType> valid_cam_state_ids(0);
        for (const auto &cam_id : cam_state_ids)
        {
            if (feature.observations.find(cam_id) ==
                feature.observations.end())
                continue;

            valid_cam_state_ids.push_back(cam_id);
        }

        int jacobian_row_size = 4 * valid_cam_state_ids.size();

        // 误差相对于状态量的雅可比，没有约束列数，因为列数一直是最新的
        int dimP_i = state_server.robot_state.dimP_i();
        MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
                                       dimP_i + 12 + state_server.cam_states.size() * 6);
        // 误差相对于三维点的雅可比
        MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
        // 误差
        VectorXd r_j = VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;

        // 2. 计算每一个观测（同一帧左右目这里被叫成一个观测）的雅可比与误差
        for (const auto &cam_id : valid_cam_state_ids)
        {
            Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
            Matrix<double, 4, 6> H_ci = Matrix<double, 4, 6>::Zero();
            Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
            Vector4d r_i = Vector4d::Zero();
            // 2.1 计算一个左右目观测的雅可比
            measurementJacobian(feature.id, cam_id, H_xi, H_ci, H_fi, r_i);

            // 计算这个cam_id在整个矩阵的列数，因为要在大矩阵里面放
            auto cam_state_iter = state_server.cam_states.find(cam_id);
            int cam_state_cntr = std::distance(
                state_server.cam_states.begin(), cam_state_iter);

            // Stack the Jacobians.
            H_xj.block<4, 6>(stack_cntr, dimP_i + 6) = H_xi;
            H_xj.block<4, 6>(stack_cntr, dimP_i + 12 + 6 * cam_state_cntr) = H_ci;
            H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
            r_j.segment<4>(stack_cntr) = r_i;
            stack_cntr += 4;
        }

        // Project the residual and Jacobians onto the nullspace
        // of H_fj.
        // 零空间投影
        JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
        MatrixXd A = svd_helper.matrixU().rightCols(
            jacobian_row_size - 3);

        // 上面的效果跟QR分解一样，下面的代码可以测试打印对比
        // Eigen::ColPivHouseholderQR<MatrixXd> qr(H_fj);
        // MatrixXd Q = qr.matrixQ();
        // std::cout << "spqr_helper.matrixQ(): " << std::endl << Q << std::endl << std::endl;
        // std::cout << "A: " << std::endl << A << std::endl;

        // 0空间投影
        H_x = A.transpose() * H_xj;
        r = A.transpose() * r_j;

        return;
    }

    /**
     * @brief 计算一个路标点的雅可比
     * @param  cam_state_id 有效的相机状态id
     * @param  feature_id 路标点id
     * @param  H_x 误差相对于位姿的雅可比
     * @param  H_f 误差相对于三维点的雅可比
     * @param  r 误差
     */
    void MsckfVio::measurementJacobian(const FeatureIDType &feature_id,
                                       const StateIDType &cam_state_id,
                                       Matrix<double, 4, 6> &H_x,
                                       Matrix<double, 4, 6> &H_c,
                                       Matrix<double, 4, 3> &H_f,
                                       Vector4d &r)
    {
        // 1. 取出相机状态与特征
        const CAMState &cam_state = state_server.cam_states[cam_state_id];
        const Feature &feature = map_server[feature_id];

        // 2. 取出左目位姿，根据外参计算右目位姿
        Matrix3d R_w_c0 = cam_state.R_G_Cam0.transpose(); // Rc0w
        const Vector3d &p_c0_w = cam_state.p_G_Cam0;      // pwc0

        // Cam1 pose.
        // Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
        // Vector3d p_c1_c0 = CAMState::T_cam0_cam1.inverse().translation();
        // TAG 2  这里可以设置是否使用双目外参更新
        Matrix3d R_c0_c1 = state_server.robot_state.R_cam1_cam0.transpose(); // Rc1c0
        Vector3d p_c1_c0 = state_server.robot_state.t_cam1_cam0;             // pc0c1
        Matrix3d R_w_c1 = R_c0_c1 * R_w_c0;                                  // Rc1w = Rc1c0 * Rc0w
        Vector3d p_c1_w = p_c0_w + R_w_c0.transpose() * p_c1_c0;             // pwc1 = pwc0 + Rwc0 * pc0c1

        // 3. 取出三维点坐标与归一化的坐标点，因为前端发来的是归一化坐标的
        const Vector3d &p_w = feature.position;
        const Vector4d &z = feature.observations.find(cam_state_id)->second;

        // 4. 转到左右目相机坐标系下
        Vector3d p_c0 = R_w_c0 * (p_w - p_c0_w);
        Vector3d p_c1 = R_w_c1 * (p_w - p_c1_w);

        // Compute the Jacobians.
        Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
        dz_dpc0(0, 0) = 1 / p_c0(2);
        dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
        dz_dpc0(1, 1) = 1 / p_c0(2);
        dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

        Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
        dz_dpc1(2, 0) = 1 / p_c1(2);
        dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2) * p_c1(2));
        dz_dpc1(3, 1) = 1 / p_c1(2);
        dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2) * p_c1(2));

        Matrix<double, 3, 6> dpc1_dx = Matrix<double, 3, 6>::Zero();
        dpc1_dx.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0 - p_c1_c0);
        dpc1_dx.rightCols(3) = -R_c0_c1;

        Matrix<double, 3, 6> dpc0_dxc0 = Matrix<double, 3, 6>::Zero();
        dpc0_dxc0.leftCols(3) = R_w_c0 * skewSymmetric(p_w - p_c0_w);
        dpc0_dxc0.rightCols(3) = -R_w_c0;

        Matrix<double, 3, 6> dpc1_dxc0 = Matrix<double, 3, 6>::Zero();
        dpc1_dxc0 = R_c0_c1 * dpc0_dxc0;

        Matrix3d dpc0_dpw = R_w_c0;
        Matrix3d dpc1_dpw = R_w_c1;

        // Follow the chain rule.
        H_x = dz_dpc1 * dpc1_dx;
        H_c = dz_dpc0 * dpc0_dxc0 + dz_dpc1 * dpc1_dxc0;
        H_f = dz_dpc0 * dpc0_dpw + dz_dpc1 * dpc1_dpw;

        // Compute the residual.
        r = z - Vector4d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2),
                         p_c1(0) / p_c1(2), p_c1(1) / p_c1(2));

        return;
    }

    /**
     * @brief 当cam状态数达到最大值时，挑出若干cam状态待删除
     */
    void MsckfVio::pruneCamStateBuffer()
    {
        // 数量还不到该删的程度，配置文件里面是20个
        if (state_server.cam_states.size() < max_cam_state_size)
            return;

        // 1. 找出两个该删的相机状态的id
        vector<StateIDType> rm_cam_state_ids(0);
        findRedundantCamStates(rm_cam_state_ids);

        // 2. 找到待删除帧涉及的观测数量，从而计算雅可比与误差的行数
        int jacobian_row_size = 0;
        // 遍历所有特征点
        for (auto &item : map_server)
        {
            auto &feature = item.second;
            // 2.1 查找该特征点中储存的观测中(特征点中储存了哪些帧观测到了该特征点)是否有待删除的帧
            vector<StateIDType> involved_cam_state_ids(0);
            for (const auto &cam_id : rm_cam_state_ids)
            {
                if (feature.observations.find(cam_id) !=
                    feature.observations.end())
                    involved_cam_state_ids.push_back(cam_id);
            }

            // 如果involved_cam_state_ids为空，说明该特征点没有被待删除的帧观测到，直接跳过
            if (involved_cam_state_ids.size() == 0)
                continue;
            // 2.2 如果只有一个待删除的帧观测到了该特征点，那么将该特征点中关于该待删除帧的观测删除(即该特征点不再被该帧观测到，因为该帧删除了)
            if (involved_cam_state_ids.size() == 1)
            {
                feature.observations.erase(involved_cam_state_ids[0]);
                continue;
            }
            // 程序到这里说明involved_cam_state_ids至少大于等于2
            // 说明该特征点记录的相机帧id中至少有两个帧是待删除的帧
            // 2.3 如果该特征点没有做过三角化，做一下三角化，如果失败直接从 该特征点记录的 观测到该特征点的相机帧id 中删除该待删除的帧
            if (!feature.is_initialized)
            {
                // Check if the feature can be initialize.
                if (!feature.checkMotion(state_server.cam_states))
                {
                    for (const auto &cam_id : involved_cam_state_ids)
                        feature.observations.erase(cam_id);
                    continue;
                }
                else
                {
                    if (!feature.initializePosition(state_server.cam_states, state_server.robot_state))
                    {
                        for (const auto &cam_id : involved_cam_state_ids)
                            feature.observations.erase(cam_id);
                        continue;
                    }
                }
            }

            // 2.4 计算出雅可比与误差的行数
            // 因为删除这些帧不能只删除，其中还包含有用的观测信息，最终还可以做一次更新
            jacobian_row_size += 4 * involved_cam_state_ids.size() - 3;
        }

        // cout << "jacobian row #: " << jacobian_row_size << endl;

        // 3. 计算待删掉的这部分观测的雅可比与误差
        // 预设大小
        int dimP_i = state_server.robot_state.dimP_i();
        MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                      dimP_i + 12 + 6 * state_server.cam_states.size());
        VectorXd r = VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;

        // 又做了一遍类似上面的遍历，只不过该三角化的已经三角化，该删的已经删了
        for (auto &item : map_server)
        {
            auto &feature = item.second;
            // 查找该特征点中储存的观测中是否有待删除的帧，把待删除的帧id存到involved_cam_state_ids中
            vector<StateIDType> involved_cam_state_ids(0);
            for (const auto &cam_id : rm_cam_state_ids)
            {
                if (feature.observations.find(cam_id) !=
                    feature.observations.end())
                    involved_cam_state_ids.push_back(cam_id);
            }

            // 一个的情况已经被删掉了
            if (involved_cam_state_ids.size() == 0)
                continue;

            // 累积该特征点相对于involved_cam_state_ids中的相机帧的雅可比与误差
            MatrixXd H_xj;
            VectorXd r_j;
            featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

            if (gatingTest(H_xj, r_j, involved_cam_state_ids.size()))
            {
                H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
                r.segment(stack_cntr, r_j.rows()) = r_j;
                stack_cntr += H_xj.rows();
            }

            // 计算完雅可比和误差后，从该特征点记录的 观测到该特征点的相机帧id 中删除involved_cam_state_ids中的帧
            // 即该特征点以后不再被involved_cam_state_ids中的帧观测到
            for (const auto &cam_id : involved_cam_state_ids)
                feature.observations.erase(cam_id);
        }

        H_x.conservativeResize(stack_cntr, H_x.cols());
        r.conservativeResize(stack_cntr);

        // 4. 用上述计算的雅可比与误差更新状态
        measurementUpdate(H_x, r);

        // 5. 删除相机状态和协方差矩阵中对应的行列
        for (const auto &cam_id : rm_cam_state_ids)
        {
            // 找到相机状态在状态向量中的位置
            int cam_sequence = std::distance(
                state_server.cam_states.begin(), state_server.cam_states.find(cam_id));
            int cam_state_start = dimP_i + 12 + 6 * cam_sequence;
            int cam_state_end = cam_state_start + 6;

            // 直接删除状态误差协方差矩阵中对应的行列
            if (cam_state_end < state_server.state_cov.rows())
            {
                state_server.state_cov.block(cam_state_start, 0,
                                             state_server.state_cov.rows() - cam_state_end,
                                             state_server.state_cov.cols()) =
                    state_server.state_cov.block(cam_state_end, 0,
                                                 state_server.state_cov.rows() - cam_state_end,
                                                 state_server.state_cov.cols());

                state_server.state_cov.block(0, cam_state_start,
                                             state_server.state_cov.rows(),
                                             state_server.state_cov.cols() - cam_state_end) =
                    state_server.state_cov.block(0, cam_state_end,
                                                 state_server.state_cov.rows(),
                                                 state_server.state_cov.cols() - cam_state_end);

                state_server.state_cov.conservativeResize(
                    state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
            }
            else
            {
                state_server.state_cov.conservativeResize(
                    state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
            }

            // 在相机状态中删除帧
            state_server.cam_states.erase(cam_id);
        }

        return;
    }

    /**
     * @brief 找出该删的相机状态的id
     * @param  rm_cam_state_ids 要删除的相机状态id
     */
    void MsckfVio::findRedundantCamStates(vector<StateIDType> &rm_cam_state_ids)
    {
        // 1. 找到倒数第四个相机状态(次新的)，作为关键状态
        auto key_cam_state_iter = state_server.cam_states.end();
        for (int i = 0; i < 4; ++i)
            --key_cam_state_iter;

        // 倒数第三个相机状态
        auto cam_state_iter = key_cam_state_iter;
        ++cam_state_iter;

        // 序列中，第一个相机状态(滑窗中最老的)
        auto first_cam_state_iter = state_server.cam_states.begin();

        // 2. 取出关键状态的位姿
        const Vector3d key_position = key_cam_state_iter->second.p_G_Cam0;
        const Matrix3d key_rotation = key_cam_state_iter->second.R_G_Cam0.transpose();

        // 3. 遍历两次，必然删掉两个状态，有可能是相对新的，有可能是最旧的
        // 但是永远删不到最新的
        for (int i = 0; i < 2; ++i)
        {
            // 从倒数第三个开始，取出位姿
            const Vector3d position = cam_state_iter->second.p_G_Cam0;
            const Matrix3d rotation = cam_state_iter->second.R_G_Cam0.transpose();

            // 计算相对于关键相机状态的平移与旋转
            double distance = (position - key_position).norm();
            double angle = AngleAxisd(rotation * key_rotation.transpose()).angle();

            // 判断该帧与关键相机状态是否有足够的旋转和平移，如果旋转平移过小或者跟踪率过高，说明该帧与关键帧高度相似，可以删除。如果删除了，下次比较第二帧和关键帧
            // 否则删除最老的帧。如果删除了，下次比较倒数第二老的帧和关键帧
            if (angle < rotation_threshold &&
                distance < translation_threshold &&
                tracking_rate > tracking_rate_threshold)
            {
                rm_cam_state_ids.push_back(cam_state_iter->first);
                ++cam_state_iter;
            }
            else
            {
                rm_cam_state_ids.push_back(first_cam_state_iter->first);
                ++first_cam_state_iter;
            }
        }

        // Sort the elements in the output vector.
        // 4. 排序
        sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

        return;
    }

    // 卡方检验，这部分有点乱
    bool MsckfVio::gatingTest(
        const MatrixXd &H, const VectorXd &r, const int &dof)
    {
        if (use_gatingTest == false)
            return true;
        // 输入的dof的值是所有相机观测，且没有去掉滑窗的
        // 而且按照维度这个卡方的维度也不对
        //
        MatrixXd P1 = H * state_server.state_cov * H.transpose();
        MatrixXd P2 = Feature::observation_noise *
                      MatrixXd::Identity(H.rows(), H.rows());
        double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

        if (gamma < chi_squared_test_table[dof])
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void MsckfVio::onlineReset()
    {

        // Never perform online reset if position std threshold
        // is non-positive.
        if (position_std_threshold <= 0)
            return;
        static long long int online_reset_counter = 0;

        // Check the uncertainty of positions to determine if
        // the system can be reset.
        // double position_x_std = std::sqrt(std::abs(state_server.state_cov(6, 6)));
        // double position_y_std = std::sqrt(std::abs(state_server.state_cov(7, 7)));
        // double position_z_std = std::sqrt(std::abs(state_server.state_cov(8, 8)));
        double position_x_std = std::sqrt(state_server.state_cov(6, 6));
        double position_y_std = std::sqrt(state_server.state_cov(7, 7));
        double position_z_std = std::sqrt(state_server.state_cov(8, 8));

        if (position_x_std < position_std_threshold &&
            position_y_std < position_std_threshold &&
            position_z_std < position_std_threshold)
            return;

        ROS_WARN("Start %lld online reset procedure...",
                 ++online_reset_counter);
        ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
                 position_x_std, position_y_std, position_z_std);
        cout << "旋转协方差: \n" << state_server.state_cov.block<3, 3>(0, 0) << endl;
        cout << "速度协方差: \n" << state_server.state_cov.block<3, 3>(3, 3) << endl;
        cout << "位置协方差: \n" << state_server.state_cov.block<3, 3>(6, 6) << endl;
        cout << "腿1协方差: \n" << state_server.state_cov.block<3, 3>(9, 9) << endl;
        cout << "腿2协方差: \n" << state_server.state_cov.block<3, 3>(12, 12) << endl;
        cout << "腿3协方差: \n" << state_server.state_cov.block<3, 3>(15, 15) << endl;
        cout << "腿4协方差: \n" << state_server.state_cov.block<3, 3>(18, 18) << endl;
        if (online_reset_counter >= 5)
            ROS_ERROR("结果严重发散，请终止程序");

        // Remove all existing camera states.
        state_server.cam_states.clear();

        // Clear all exsiting features in the map.
        map_server.clear();

        // Reset the state covariance.
        double gyro_bias_cov, acc_bias_cov, velocity_cov;
        nh.param<double>("initial_covariance/velocity",
                         velocity_cov, 0.25);
        nh.param<double>("initial_covariance/gyro_bias",
                         gyro_bias_cov, 1e-4);
        nh.param<double>("initial_covariance/acc_bias",
                         acc_bias_cov, 1e-2);

        double extrinsic_rotation_cov, extrinsic_translation_cov;
        nh.param<double>("initial_covariance/extrinsic_rotation_cov",
                         extrinsic_rotation_cov, 3.0462e-4);
        nh.param<double>("initial_covariance/extrinsic_translation_cov",
                         extrinsic_translation_cov, 1e-4);

        double stereo_extrinsic_rotation_cov, stereo_extrinsic_translation_cov;
        nh.param<double>("initial_covariance/stereo_extrinsic_rotation_cov",
                         stereo_extrinsic_rotation_cov, 3.0462e-4);
        nh.param<double>("initial_covariance/stereo_extrinsic_translation_cov",
                         stereo_extrinsic_translation_cov, 1e-4);

        // 0~3 旋转 3~6 速度 6~9 位移 9~12 陀螺仪偏置 12~15 加速度计偏置
        // 15~18 左目到IMU的旋转 18~21 左目到IMU的平移
        // 21~24 右目到左目的旋转 24~27 右目到左目的平移
        state_server.state_cov = MatrixXd::Zero(27, 27);
        for (int i = 3; i < 6; ++i)
            state_server.state_cov(i, i) = velocity_cov;
        for (int i = 9; i < 12; ++i)
            state_server.state_cov(i, i) = gyro_bias_cov;
        for (int i = 12; i < 15; ++i)
            state_server.state_cov(i, i) = acc_bias_cov;
        for (int i = 15; i < 18; ++i)
            state_server.state_cov(i, i) = extrinsic_rotation_cov;
        for (int i = 18; i < 21; ++i)
            state_server.state_cov(i, i) = extrinsic_translation_cov;
        for (int i = 21; i < 24; ++i)
            state_server.state_cov(i, i) = stereo_extrinsic_rotation_cov;
        for (int i = 24; i < 27; ++i)
            state_server.state_cov(i, i) = stereo_extrinsic_translation_cov;

        ROS_WARN("%lld online reset complete...", online_reset_counter);
        return;
    }

    void MsckfVio::publish(const ros::Time &time)
    {
        // Convert the IMU frame to the body frame.
        // 1. 计算body坐标，因为imu与body相对位姿是单位矩阵，所以就是imu的坐标
        const RobotState &robot_state = state_server.robot_state;
        Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
        T_i_w.linear() = robot_state.getR_GI();
        T_i_w.translation() = robot_state.getp_GI();

        Eigen::Isometry3d T_b_w = RobotState::T_imu_body * T_i_w *
                                  RobotState::T_imu_body.inverse();
        Eigen::Vector3d body_velocity = RobotState::T_imu_body.linear() * robot_state.getv_GI();

        // Publish tf
        // 2. 发布tf，实时的位姿，没有轨迹，没有协方差
        if (publish_tf)
        {
            tf::Transform T_b_w_tf;
            tf::transformEigenToTF(T_b_w, T_b_w_tf);
            tf_pub.sendTransform(tf::StampedTransform(
                T_b_w_tf, time, fixed_frame_id, child_frame_id));
        }

        // Publish the odometry
        // 3. 发布位姿，能在rviz留下轨迹的
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = time;
        odom_msg.header.frame_id = fixed_frame_id;
        odom_msg.child_frame_id = child_frame_id;

        tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
        tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

        // Convert the covariance.
        // 协方差，取出旋转平移部分，以及它们之间的公共部分组成6自由度的协方差
        Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
        Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 6);
        Matrix3d P_po = state_server.state_cov.block<3, 3>(6, 0);
        Matrix3d P_pp = state_server.state_cov.block<3, 3>(6, 6);
        Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
        P_imu_pose << P_pp, P_po, P_op, P_oo;

        // 转下坐标，但是这里都是单位矩阵
        Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
        H_pose.block<3, 3>(0, 0) = RobotState::T_imu_body.linear();
        H_pose.block<3, 3>(3, 3) = RobotState::T_imu_body.linear();
        Matrix<double, 6, 6> P_body_pose = H_pose *
                                           P_imu_pose * H_pose.transpose();

        // 填充协方差
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                odom_msg.pose.covariance[6 * i + j] = P_body_pose(i, j);

        // Construct the covariance for the velocity.
        // 速度协方差
        Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(3, 3);
        Matrix3d H_vel = RobotState::T_imu_body.linear();
        Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                odom_msg.twist.covariance[i * 6 + j] = P_body_vel(i, j);

        // 发布位姿
        odom_pub.publish(odom_msg);

        // 发布轨迹
        vio_path.header = odom_msg.header;
        geometry_msgs::PoseStamped pose;
        pose.header = odom_msg.header;
        pose.pose = odom_msg.pose.pose;
        vio_path.poses.push_back(pose);
        vio_path_pub.publish(vio_path);
        vio_flag = true;
        vio_point = Eigen::Vector3d(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);

        if (output_file_path != "none")
        {
            ofstream out(output_file_path, ios::app);
            if (out.is_open())
            {
                out << setprecision(18) << odom_msg.header.stamp.toSec() << " " << setprecision(9) << pose.pose.position.x << " " << pose.pose.position.y << " " << pose.pose.position.z
                    << " " << pose.pose.orientation.x << " " << pose.pose.orientation.y << " " << pose.pose.orientation.z << " " << pose.pose.orientation.w << endl;
            }
        }

        // 4. 发布点云
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> feature_msg_ptr(
            new pcl::PointCloud<pcl::PointXYZ>());
        feature_msg_ptr->header.frame_id = fixed_frame_id;
        feature_msg_ptr->height = 1;
        for (const auto &item : map_server)
        {
            const auto &feature = item.second;
            if (feature.is_initialized)
            {
                Vector3d feature_position =
                    RobotState::T_imu_body.linear() * feature.position;
                feature_msg_ptr->points.push_back(pcl::PointXYZ(
                    feature_position(0), feature_position(1), feature_position(2)));
            }
        }
        feature_msg_ptr->width = feature_msg_ptr->points.size();

        feature_pub.publish(feature_msg_ptr);

        return;
    }

    void MsckfVio::leicaCallback(const geometry_msgs::PointStampedConstPtr &msg)
    {
        // 这里leica和Vicon的轨迹与估计间还差一个转换，但是数据集中没有给这个转换，用EVO评估的话是
        // 自动对齐的，但这里要想显示的话需要人工求解。即先时间软同步得到估计和真实的轨迹点，再用最小二乘求
        // p1 = T12p2
        ground_truth_path.header = msg->header;
        ground_truth_path.header.frame_id = fixed_frame_id;

        geometry_msgs::PoseStamped pose;
        pose.header = ground_truth_path.header;
        // 自己根据轨迹用最小二乘求解的转换矩阵，因为每个数据包都不一样，所以实际比对时需要用evo工具做轨迹对齐。
        Eigen::Vector4d point_leica(msg->point.x, msg->point.y, msg->point.z, 1);
        Eigen::Vector4d point_leica_w = T_WR * point_leica;
        pose.pose.position.x = point_leica_w(0);
        pose.pose.position.y = point_leica_w(1);
        pose.pose.position.z = point_leica_w(2);
        ground_truth_path.poses.push_back(pose);
        ground_truth_pub.publish(ground_truth_path);
        ground_truth_flag = true;
        ground_truth_point = Eigen::Vector3d(msg->point.x, msg->point.y, msg->point.z);
    }

    void MsckfVio::viconCallback(const geometry_msgs::TransformStampedConstPtr &msg)
    {
        ground_truth_path.header = msg->header;
        ground_truth_path.header.frame_id = fixed_frame_id;

        geometry_msgs::PoseStamped pose;
        pose.header = ground_truth_path.header;
        Eigen::Isometry3d T_RS;
        tf::transformMsgToEigen(msg->transform, T_RS);
        Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();
        // 官方提供的外参
        T_BS << 0.33638, -0.01749, 0.94156, 0.06901, -0.02078, -0.99972, -0.01114, -0.02781, 0.94150, -0.01582, -0.33665, -0.12395, 0.0, 0.0, 0.0, 1.0;
        Eigen::Isometry3d T_RB = T_RS * Eigen::Isometry3d(T_BS).inverse();
        // 官方没给，自己根据轨迹用最小二乘求解的转换矩阵，因为数据集中没给，需要做轨迹对齐
        // T_WR = Eigen::Matrix4d::Identity();
        Eigen::Isometry3d T_WB = Eigen::Isometry3d(T_WR) * T_RB;
        tf::poseEigenToMsg(T_WB, pose.pose);
        ground_truth_path.poses.push_back(pose);
        ground_truth_pub.publish(ground_truth_path);

        nav_msgs::Odometry odom_msg;
        odom_msg.header = ground_truth_path.header;
        odom_msg.child_frame_id = child_frame_id;
        odom_msg.pose.pose = pose.pose;
        ground_truth_odom_pub.publish(odom_msg);
        ground_truth_flag = true;
        ground_truth_point = T_RB.translation();
    }

    void MsckfVio::alignThreadTask()
    {
        while (true)
        {
            if (path_alignment)
            {
                if (vio_flag && ground_truth_flag)
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    vio_flag = false;
                    ground_truth_flag = false;
                    path_vio.push_back(vio_point);
                    path_ground_truth.push_back(ground_truth_point);
                    vio_point = Eigen::Vector3d::Zero();
                    ground_truth_point = Eigen::Vector3d::Zero();
                    lock.unlock();
                }
                // 数据量足够后，计算转换矩阵
                if (path_vio.size() > 1000)
                {
                    Eigen::MatrixXd P1, P2;
                    P1 = Eigen::MatrixXd::Ones(4, (int)path_vio.size());
                    P2 = Eigen::MatrixXd::Ones(4, (int)path_ground_truth.size());
                    for (int i = 0; i < (int)path_vio.size(); ++i)
                    {
                        P1.block<3, 1>(0, i) = path_vio[i];
                        P2.block<3, 1>(0, i) = path_ground_truth[i];
                    }
                    T_WR = P1 * pinv_eigen_based(P2);
                    // 然后把所有的历史轨迹点都转换一下
                    for (int i = 0; i < (int)ground_truth_path.poses.size(); ++i)
                    {
                        Eigen::Vector4d point_ground_truth(ground_truth_path.poses[i].pose.position.x, ground_truth_path.poses[i].pose.position.y, ground_truth_path.poses[i].pose.position.z, 1);
                        Eigen::Vector4d point_ground_truth_w = T_WR * point_ground_truth;
                        ground_truth_path.poses[i].pose.position.x = point_ground_truth_w(0);
                        ground_truth_path.poses[i].pose.position.y = point_ground_truth_w(1);
                        ground_truth_path.poses[i].pose.position.z = point_ground_truth_w(2);
                    }
                    ROS_INFO("================ align done =================");
                    // 最后清空储存的轨迹点并结束线程
                    path_vio.clear();
                    path_ground_truth.clear();
                    break;
                }
                // 延时一段时间
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
    }

    void MsckfVio::isTouchdownCallback(const dog_msg::isTouchdown::ConstPtr &msg)
    {
        ROS_DEBUG("isTouchdownCallback In");
        isTouchdown_buffer.push_back(*msg);
        ROS_DEBUG("isTouchdownCallback Out");
    }

    void MsckfVio::wheelMotor_fbCallback(const dog_msg::wheel_motor_fb::ConstPtr &msg)
    {
        ROS_DEBUG("wheelMotor_fbCallback In");
        wheelMotor_fb_buffer.push_back(*msg);
        ROS_DEBUG("wheelMotor_fbCallback Out");
    }

    void MsckfVio::qNow_dqNow_TNowCallback(const dog_msg::qNow_dqNow_TNow::ConstPtr &msg)
    {
        ROS_DEBUG_STREAM_BLUE("qNow_dqNow_TNowCallback In");
        int seq = msg->header.seq;
        double cur_time = msg->header.stamp.toSec();
        ROS_DEBUG_STREAM("qNow_dqNow_TNowCallback header: seq: " << seq << " time: " << cur_time);
        // BUG？ 清理, 不然内存一直增大
        // qNow_dqNow_TNow_buffer.push_back(*msg);

        /// 1. 等待IMU初始化
        if (!is_gravity_set)
        {
            ROS_WARN_THROTTLE(1, "Leg Callback: Waiting for IMU initialization!");
            return;
        }

        /// 2. 接收到第一帧腿时，记录时间，开始工作
        if (is_first_leg)
        {
            is_first_leg = false;
            state_server.robot_state.time = msg->header.stamp.toSec();
        }

        /// 3. IMU递推
        int used_imu_msg_cntr = 0;
        Vector3d m_gyro, m_acc;
        for (const auto &imu_msg : imu_msg_buffer)
        {
            double imu_time = imu_msg.header.stamp.toSec();
            if (imu_time < state_server.robot_state.time)
            {
                ++used_imu_msg_cntr;
                continue;
            }
            if (imu_time > msg->header.stamp.toSec())
                break;

            tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
            tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);
            InEKF_Propagate(imu_time, m_gyro, m_acc);
            ++used_imu_msg_cntr;
        }
        // 再继续外推到腿更新的时间
        // QUERY 需不需要
        // InEKF_Propagate(msg->header.stamp.toSec(), m_gyro, m_acc);
        imu_msg_buffer.erase(imu_msg_buffer.begin(), imu_msg_buffer.begin() + used_imu_msg_cntr);
        diagHasNegativeOrNaN(state_server.state_cov, "InEKF_Propagate后");


        /// 4. 开始进行腿运动学的更新
        // return;
        if (merge_leg == true)
        {
            ROS_DEBUG_THROTTLE(1, "merge_leg");
            Eigen::VectorXd Z;
            Eigen::MatrixXd H, N;

            vector<pair<LegID, int>> remove_contacts; // 要删除的腿列表 <腿id, 腿在X中的索引>
            vector<LegID> new_contacts;               // 要添加的腿列表

            // 分别在isTouchdown_buffer和wheelMotor_fb_buffer中取出上次更新后到此刻的数据
            // 第一次进入会有很多数据，是正常的
            vector<dog_msg::isTouchdown> isTouchdown_buffer_tmp;
            isTouchdown_buffer_tmp.clear();
            isTouchdown_buffer_tmp.reserve(isTouchdown_buffer.size() + 10);
            int used_isTouchdown_cntr = 0;
            for (auto it = isTouchdown_buffer.begin(); it != isTouchdown_buffer.end(); ++it)
            {
                if (it->header.stamp.toSec() > msg->header.stamp.toSec())
                    break;
                used_isTouchdown_cntr++;
            }
            if (used_isTouchdown_cntr > 0)
            {
                isTouchdown_buffer_tmp.assign(isTouchdown_buffer.begin(), isTouchdown_buffer.begin() + used_isTouchdown_cntr);
                isTouchdown_buffer.erase(isTouchdown_buffer.begin(), isTouchdown_buffer.begin() + used_isTouchdown_cntr);
            }
            vector<dog_msg::wheel_motor_fb> wheelMotor_fb_buffer_tmp;
            wheelMotor_fb_buffer_tmp.clear();
            wheelMotor_fb_buffer_tmp.reserve(wheelMotor_fb_buffer.size() + 10);
            int used_wheelMotor_fb_cntr = 0;
            for (auto it = wheelMotor_fb_buffer.begin(); it != wheelMotor_fb_buffer.end(); ++it)
            {
                if (it->header.stamp.toSec() > msg->header.stamp.toSec())
                    break;
                used_wheelMotor_fb_cntr++;
            }
            if (used_wheelMotor_fb_cntr > 0)
            {
                wheelMotor_fb_buffer_tmp.assign(wheelMotor_fb_buffer.begin(), wheelMotor_fb_buffer.begin() + used_wheelMotor_fb_cntr);
                wheelMotor_fb_buffer.erase(wheelMotor_fb_buffer.begin(), wheelMotor_fb_buffer.begin() + used_wheelMotor_fb_cntr);
            }
            // 将最后一个轮子的转向角度更新到leg_states中
            if (!wheelMotor_fb_buffer_tmp.empty())
            {
                state_server.leg_state.legs[FrontLeft].wheel_angle = -wheelMotor_fb_buffer_tmp.back().wheel_motor_fb[0] * DEG2RAD;
                state_server.leg_state.legs[RearLeft].wheel_angle = -wheelMotor_fb_buffer_tmp.back().wheel_motor_fb[2] * DEG2RAD;
                state_server.leg_state.legs[RearRight].wheel_angle = -wheelMotor_fb_buffer_tmp.back().wheel_motor_fb[4] * DEG2RAD;
                state_server.leg_state.legs[FrontRight].wheel_angle = -wheelMotor_fb_buffer_tmp.back().wheel_motor_fb[6] * DEG2RAD;
            }
            // 将四条腿大小腿角度更新到leg_states中
            state_server.leg_state.legs[FrontLeft].thigh_angle = msg->qNow[0];
            state_server.leg_state.legs[FrontLeft].knee_angle = msg->qNow[1];
            state_server.leg_state.legs[RearLeft].thigh_angle = msg->qNow[2];
            state_server.leg_state.legs[RearLeft].knee_angle = msg->qNow[3];
            state_server.leg_state.legs[RearRight].thigh_angle = msg->qNow[4];
            state_server.leg_state.legs[RearRight].knee_angle = msg->qNow[5];
            state_server.leg_state.legs[FrontRight].thigh_angle = msg->qNow[6];
            state_server.leg_state.legs[FrontRight].knee_angle = msg->qNow[7];
            // 计算运动学与雅可比
            state_server.leg_state.calc_FKAndJacobian();
            // 开始针对每个腿进行处理
            for (int id = 0; id < 4; ++id)
            {
                LegID leg_id = static_cast<LegID>(id);
                // 腿能否从状态量中找到
                auto it_estimated = state_server.robot_state.estimated_contact_position.find(leg_id);
                bool found = it_estimated != state_server.robot_state.estimated_contact_position.end();
                // 当前触地状态的处理
                bool true_contact = true; // 真: 全1, 真实的触地; false: 有0, 假的触地
                for (auto &isTouchdown : isTouchdown_buffer_tmp)
                {
                    state_server.leg_state.legs[leg_id].contact = isTouchdown.isTouchdown[leg_id];
                    true_contact = true_contact && isTouchdown.isTouchdown[leg_id];
                }
                bool has_contact = state_server.leg_state.legs[leg_id].contact; // 最后的触地状态
                // 计算该腿的协方差，用于构造N矩阵
                Eigen::Vector3d pose = state_server.leg_state.legs[leg_id].T.block<3, 1>(0, 3);
                Eigen::Matrix3d J = state_server.leg_state.legs[leg_id].J.block<3, 3>(0, 0);
                Eigen::Matrix3d cov = J * state_server.Qe * J.transpose() + state_server.Qkinematic_additive;
                state_server.leg_state.legs[leg_id].Cov = cov;

                // 开始进行四种情况的判断
                if (found && !true_contact) // 1. 在状态量中, 假的触地
                {
                    remove_contacts.push_back(*it_estimated);
                }
                else if (!found && has_contact) // 2. 不在状态量中, 最后状态为触地
                {
                    new_contacts.push_back(leg_id);
                }
                else if (found && true_contact) // 3. 在状态量中, 真的触地
                {
                    int dimX_i = state_server.robot_state.dimX_i();
                    int dimX_frak = state_server.robot_state.dimX_frak();
                    int dimP_i = state_server.robot_state.dimP_i();
                    int startIndex;

                    // 使用轮速数据对状态量中的该腿进行位置预估, 这里肯定是触地的，不用考虑isTouchdown_buffer
                    ROS_DEBUG_STREAM("wheelMotor_fb_buffer_tmp.size()" << wheelMotor_fb_buffer_tmp.size());
                    // BUG？ 为什么一定要加这个判断
                    if (wheelMotor_fb_buffer_tmp.size() > 1)
                    {
                        for (int i = 0; i < wheelMotor_fb_buffer_tmp.size() - 1; ++i)
                        {
                            double dt = wheelMotor_fb_buffer_tmp[i].header.stamp.toSec() - state_server.leg_state.legs[leg_id].time;
                            if (dt < 0)
                            {
                                ROS_DEBUG_STREAM("wheelmotor time: " << wheelMotor_fb_buffer_tmp[i].header.stamp.toSec());
                                ROS_DEBUG_STREAM("state_server.leg_state.legs[leg_id].time: " << state_server.leg_state.legs[leg_id].time);
                                ROS_ERROR("轮速预估时间戳错误");
                            }
                            state_server.leg_state.legs[leg_id].wheel_angleVelocity = wheelMotor_fb_buffer_tmp[i].wheel_motor_fb[2 * leg_id + 1] * DEG2RAD;
                            Eigen::Vector3d v_b = state_server.leg_state.getVelocityInBodyFrame(leg_id);
                            Eigen::Vector3d v_w = state_server.robot_state.getR_GI() * v_b;
                            v_w[2] = 0;
                            // NOTE 找到bug了，原因是这里v_w的norm值导致矩阵中出现了nan
                            double v_w_norm = v_w.norm();
                            if (v_w_norm > 1e-6)
                            {
                                v_w = v_w / v_w_norm * v_b.norm();
                            }
                            int index = it_estimated->second;
                            Eigen::Vector3d d_GI = state_server.robot_state.getd_GI(index);
                            d_GI += v_w * dt;
                            state_server.robot_state.setd_GI(d_GI, index);
                            state_server.leg_state.legs[leg_id].time = wheelMotor_fb_buffer_tmp[i].header.stamp.toSec();
                        }
                    }
                    if (!wheelMotor_fb_buffer_tmp.empty())
                        state_server.leg_state.legs[leg_id].wheel_angleVelocity = wheelMotor_fb_buffer_tmp.back().wheel_motor_fb[2 * leg_id + 1] * DEG2RAD;
                    double dt = msg->header.stamp.toSec() - state_server.leg_state.legs[leg_id].time;
                    if (dt < 0)
                    {
                        ROS_DEBUG_STREAM("msg time: " << msg->header.stamp.toSec());
                        ROS_DEBUG_STREAM("state_server.leg_state.legs[leg_id].time: " << state_server.leg_state.legs[leg_id].time);
                        ROS_ERROR("轮速预估时间戳错误");
                    }
                    Eigen::Vector3d v_b = state_server.leg_state.getVelocityInBodyFrame(leg_id);
                    Eigen::Vector3d v_w = state_server.robot_state.getR_GI() * v_b;
                    v_w[2] = 0;
                    double v_w_norm = v_w.norm();
                    if (v_w_norm > 1e-6)
                    {
                        v_w = v_w / v_w_norm * v_b.norm();
                    }
                    int index = it_estimated->second;
                    Eigen::Vector3d d_GI = state_server.robot_state.getd_GI(index);
                    d_GI += v_w * dt;
                    state_server.robot_state.setd_GI(d_GI, index);
                    state_server.leg_state.legs[leg_id].time = msg->header.stamp.toSec();

                    // if (id == 0)
                    // {
                    //     BLUECOUT("FL leg");
                    //     BLUECOUT("  v_b: " << v_b.transpose());
                    //     BLUECOUT("r v_b: " << webotsRealState.FLlun_velocity_in_body.transpose());
                    //     BLUECOUT("  v_w: " << v_w.transpose());
                    //     BLUECOUT("r v_w: " << (webotsRealState.R_NWU2NUE * webotsRealState.FLlun_velocity_in_world).transpose());
                    //     BLUECOUT("  R_GI:\n" << state_server.robot_state.getR_GI());
                    //     BLUECOUT("r R_GI:\n" << webotsRealState.R_NWU2NUE * webotsRealState.body_rotation_in_world);
                    //     cout << endl;
                    // }
                    // else if(id == 1)
                    // {
                    //     BLUECOUT("RL leg");
                    //     BLUECOUT("  v_b: " << v_b.transpose());
                    //     BLUECOUT("r v_b: " << webotsRealState.RLlun_velocity_in_body.transpose());
                    //     BLUECOUT("  v_w: " << v_w.transpose());
                    //     BLUECOUT("r v_w: " << (webotsRealState.R_NWU2NUE * webotsRealState.RLlun_velocity_in_world).transpose());
                    //     cout << endl;
                    // }
                    // else if(id == 2)
                    // {
                    //     BLUECOUT("RR leg");
                    //     BLUECOUT("  v_b: " << v_b.transpose());
                    //     BLUECOUT("r v_b: " << webotsRealState.RRlun_velocity_in_body.transpose());
                    //     BLUECOUT("  v_w: " << v_w.transpose());
                    //     BLUECOUT("r v_w: " << (webotsRealState.R_NWU2NUE * webotsRealState.RRlun_velocity_in_world).transpose());
                    //     cout << endl;
                    // }
                    // else if(id == 3)
                    // {
                    //     BLUECOUT("FR leg");
                    //     BLUECOUT("  v_b: " << v_b.transpose());
                    //     BLUECOUT("r v_b: " << webotsRealState.FRlun_velocity_in_body.transpose());
                    //     BLUECOUT("  v_w: " << v_w.transpose());
                    //     BLUECOUT("r v_w: " << (webotsRealState.R_NWU2NUE * webotsRealState.FRlun_velocity_in_world).transpose());
                    //     cout << endl;
                    //     cout << endl;
                    // }

                    // 开始构造几大矩阵
                    // QUERY 腿更新时带不带上相机外参，现在先带上
                    // H阵
                    startIndex = H.rows();
                    H.conservativeResize(startIndex + 3, dimP_i + 12);
                    H.block(startIndex, 0, 3, dimP_i + 12) = MatrixXd::Zero(3, dimP_i + 12);
                    H.block<3, 3>(startIndex, 6) = -Matrix3d::Identity();                                   // p项
                    H.block<3, 3>(startIndex, 3 * it_estimated->second - dimX_frak) = Matrix3d::Identity(); // d项
                    // ROS_DEBUG_STREAM("H: \n" << H << endl);

                    // N阵
                    startIndex = N.rows();
                    N.conservativeResize(startIndex + 3, startIndex + 3);
                    N.block(startIndex, 0, 3, startIndex) = MatrixXd::Zero(3, startIndex); // 左下角置0
                    N.block(0, startIndex, startIndex, 3) = MatrixXd::Zero(startIndex, 3); // 右上角置0
                    Eigen::Matrix3d R = state_server.robot_state.getR_GI();
                    N.block(startIndex, startIndex, 3, 3) = R * cov * R.transpose();
                    // ROS_DEBUG_STREAM("N: \n" << N << endl);

                    // Z阵
                    startIndex = Z.rows();
                    Z.conservativeResize(startIndex + 3, Eigen::NoChange);
                    Eigen::Vector3d p = state_server.robot_state.getp_GI();
                    Eigen::Vector3d d = state_server.robot_state.getd_GI(it_estimated->second);
                    Z.segment(startIndex, 3) = R * pose - (d - p);
                    // ROS_DEBUG_STREAM("Z: \n" << Z << endl);
                }
                else // 不在状态量中，且为假触地，跳过
                {
                    continue;
                }
            }
            // 使用构造的观测数据进行更新
            if (Z.rows() > 0)
            {
                diagHasNegativeOrNaN(state_server.state_cov, "InEKF_Correct前");
                InEKF_Correct(Z, H, N);
                diagHasNegativeOrNaN(state_server.state_cov, "InEKF_Correct后");
            }
            // 从状态量中移除不再触地的腿
            if (remove_contacts.size() > 0)
            {
                Eigen::MatrixXd X_rem = state_server.robot_state.getX_i();
                Eigen::MatrixXd P_rem = state_server.state_cov; // TODO: 优化为引用
                for (vector<pair<LegID, int>>::iterator it = remove_contacts.begin(); it != remove_contacts.end(); ++it)
                {
                    int index = it->second;
                    state_server.robot_state.estimated_contact_position.erase(it->first);
                    RemoveRowAndColumn(X_rem, index, 1);
                    int startIndex = 3 + 3 * (index - 3);
                    RemoveRowAndColumn(P_rem, startIndex, 3);
                    for (map<LegID, int>::iterator it2 = state_server.robot_state.estimated_contact_position.begin();
                         it2 != state_server.robot_state.estimated_contact_position.end(); ++it2)
                    {
                        if (it2->second > index)
                            it2->second -= 1;
                    }
                    for (vector<pair<LegID, int>>::iterator it2 = it; it2 != remove_contacts.end(); ++it2)
                    {
                        if (it2->second > index)
                            it2->second -= 1;
                    }
                    // TODO: 是否需要放在循环中
                    state_server.robot_state.X_i_valid_size -= 1;
                    state_server.robot_state.setX_i(X_rem);
                    state_server.state_cov = P_rem;
                    // ROS_DEBUG_STREAM("X_i :\n" << state_server.robot_state.getX_i() << endl);
                    ROS_DEBUG_STREAM("P size: " << P_rem.rows() << " " << P_rem.cols() << endl);
                    for (auto &tmp : state_server.robot_state.estimated_contact_position)
                    {
                        ROS_DEBUG_STREAM(tmp.first << "->" << tmp.second << " ");
                    }
                }
            }
            // 向状态中增加新触地的腿
            if (new_contacts.size() > 0)
            {
                Eigen::MatrixXd X_i_aug = state_server.robot_state.getX_i();
                // TODO: 更改为引用
                Eigen::MatrixXd P_aug = state_server.state_cov;
                Eigen::Vector3d p_GI = state_server.robot_state.getp_GI();
                Eigen::Matrix3d R_GI = state_server.robot_state.getR_GI();
                int dimX_frak = state_server.robot_state.dimX_frak();
                for (LegID new_contact : new_contacts)
                {
                    Eigen::Vector3d pose = state_server.leg_state.legs[new_contact].T.block<3, 1>(0, 3);
                    int startIndex = X_i_aug.rows();
                    X_i_aug.conservativeResizeLike(Eigen::MatrixXd::Identity(startIndex + 1, startIndex + 1)); // 将X大小扩充1                                              // 新增对角线部分置1
                    X_i_aug.block(0, startIndex, 3, 1) = p_GI + R_GI * pose;

                    // TODO 如果不想让腿的更新影响到相机，是不是可以把其协方差置0？
                    int dimP_aug = P_aug.rows();
                    int dimP_i = state_server.robot_state.dimP_i();
                    Eigen::SparseMatrix<double> F_sparse(dimP_aug + 3, dimP_aug);
                    std::vector<Eigen::Triplet<double>> tripletList;
                    tripletList.reserve(dimP_aug + 3);
                    for (int i = 0; i < dimP_i - dimX_frak; ++i) // for old X
                    {
                        tripletList.push_back(Eigen::Triplet<double>(i, i, 1.0));
                    }
                    for (int i = dimP_i - dimX_frak + 3; i < dimP_aug + 3; ++i) // for theta~Camera
                    {
                        tripletList.push_back(Eigen::Triplet<double>(i, i - 3, 1.0));
                    }
                    for (int i = dimP_i - dimX_frak; i < dimP_i - dimX_frak + 3; ++i) // for new leg
                    {
                        tripletList.push_back(Eigen::Triplet<double>(i, 6 + (i - (dimP_i - dimX_frak)), 1.0));
                    }
                    F_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

                    Eigen::SparseMatrix<double> G_sparse(F_sparse.rows(), 3);
                    tripletList.clear();
                    tripletList.reserve(9);
                    for (int i = 0; i < 3; ++i)
                    {
                        for (int j = 0; j < 3; ++j)
                        {
                            // 因为计算cov时候加过雅可比了，所以这里不用再加雅可比
                            tripletList.push_back(Eigen::Triplet<double>(dimP_i - dimX_frak + i, j, R_GI(i, j)));
                        }
                    }
                    G_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
                    P_aug = (F_sparse * P_aug * F_sparse.transpose() + G_sparse * state_server.leg_state.legs[new_contact].Cov * G_sparse.transpose()).eval();

                    state_server.robot_state.X_i_valid_size += 1;
                    state_server.robot_state.setX_i(X_i_aug);
                    ROS_DEBUG_STREAM("X_i \n " << state_server.robot_state.getX_i() << endl);
                    ROS_DEBUG_STREAM("P_aug size " << P_aug.rows() << " " << P_aug.cols() << endl);
                    state_server.state_cov = P_aug;
                    state_server.robot_state.estimated_contact_position.insert(pair<LegID, int>(new_contact, startIndex));
                    state_server.leg_state.legs[new_contact].time = msg->header.stamp.toSec();
                }
            }
        }
        diagHasNegativeOrNaN(state_server.state_cov, "qNow_dqNow_TNowCallback后");
        ROS_DEBUG_STREAM_BLUE("qNow_dqNow_TNowCallback Out");
        return;
    }

    void MsckfVio::InEKF_Propagate(const double &time, const Eigen::Vector3d &m_gyro, const Eigen::Vector3d &m_acc)
    {
        ROS_DEBUG("InEKF_Propagate In");
        RobotState &robot_state = state_server.robot_state;
        Vector3d gyro = m_gyro - robot_state.getbg();
        Vector3d acc = m_acc - robot_state.getba();
        double dt = time - robot_state.time;

        Matrix3d R = robot_state.getR_GI();
        Vector3d v = robot_state.getv_GI();
        Vector3d p = robot_state.getp_GI();
        int dimP_i = state_server.robot_state.dimP_i();

        //  ------------ Propagate Covariance --------------- //
        Eigen::MatrixXd Phi = this->StateTransitionMatrix(gyro, acc, dt);
        Eigen::MatrixXd Qd = this->DiscreteNoiseMatrix(Phi, dt);
        // BUG ?
        state_server.state_cov.block(0, 0, dimP_i + 12, dimP_i + 12) =
            (Phi * state_server.state_cov.block(0, 0, dimP_i + 12, dimP_i + 12) * Phi.transpose() + Qd).eval();
        if (state_server.cam_states.size() > 0)
        {
            state_server.state_cov.block(0, dimP_i + 12, dimP_i + 12, state_server.state_cov.cols() - dimP_i - 12) =
                (Phi * state_server.state_cov.block(0, dimP_i + 12, dimP_i + 12, state_server.state_cov.cols() - dimP_i - 12)).eval();
            state_server.state_cov.block(dimP_i + 12, 0, state_server.state_cov.rows() - dimP_i - 12, dimP_i + 12) =
                (state_server.state_cov.block(dimP_i + 12, 0, state_server.state_cov.rows() - dimP_i - 12, dimP_i + 12) * Phi.transpose()).eval();
        }
        MatrixXd state_cov_fixed =
            0.5 * (state_server.state_cov + state_server.state_cov.transpose());
        state_server.state_cov = state_cov_fixed;

        //  ------------ Propagate Mean --------------- //
        Eigen::Matrix3d G0 = Gamma_SO3(gyro * dt, 0);
        Eigen::Matrix3d G1 = Gamma_SO3(gyro * dt, 1);
        Eigen::Matrix3d G2 = Gamma_SO3(gyro * dt, 2);
        Eigen::Matrix3d R_pred = R * G0;
        Eigen::Vector3d v_pred = v + (R * G1 * acc + RobotState::gravity) * dt;
        Eigen::Vector3d p_pred = p + v * dt + (R * G2 * acc + 0.5 * RobotState::gravity) * dt * dt;
        robot_state.setR_GI(R_pred);
        robot_state.setv_GI(v_pred);
        robot_state.setp_GI(p_pred);

        // 更新IMU状态的时间
        state_server.robot_state.time = time;
        ROS_DEBUG("InEKF_Propagate Out");
    }

    Eigen::MatrixXd MsckfVio::StateTransitionMatrix(const Eigen::Vector3d &w, const Eigen::Vector3d &a, double dt)
    {
        ROS_DEBUG("StateTransitionMatrix In");
        Eigen::Vector3d phi = w * dt;
        Eigen::Matrix3d G0 = Gamma_SO3(phi, 0);
        Eigen::Matrix3d G1 = Gamma_SO3(phi, 1);
        Eigen::Matrix3d G2 = Gamma_SO3(phi, 2);
        Eigen::Matrix3d G0t = G0.transpose();
        Eigen::Matrix3d G1t = G1.transpose();
        Eigen::Matrix3d G2t = G2.transpose();
        Eigen::Matrix3d G3t = Gamma_SO3(-phi, 3);

        int dimX_i = state_server.robot_state.dimX_i();
        int dimX_frak = state_server.robot_state.dimX_frak();
        int dimP_i = state_server.robot_state.dimP_i();
        // 同样考虑了4个相机参数相关变量
        Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(dimP_i + 12, dimP_i + 12);

        Eigen::Matrix3d ax = skewSymmetric(a);
        Eigen::Matrix3d wx = skewSymmetric(w);
        Eigen::Matrix3d wx2 = wx * wx;
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double theta = w.norm();
        double theta2 = theta * theta;
        double theta3 = theta2 * theta;
        double theta4 = theta3 * theta;
        double theta5 = theta4 * theta;
        double theta6 = theta5 * theta;
        double theta7 = theta6 * theta;
        double thetadt = theta * dt;
        double thetadt2 = thetadt * thetadt;
        double thetadt3 = thetadt2 * thetadt;
        double sinthetadt = sin(thetadt);
        double costhetadt = cos(thetadt);
        double sin2thetadt = sin(2 * thetadt);
        double cos2thetadt = cos(2 * thetadt);
        double thetadtcosthetadt = thetadt * costhetadt;
        double thetadtsinthetadt = thetadt * sinthetadt;

        // Contact-aided 论文附录公式55 & 56
        Eigen::Matrix3d Phi25L =
            G0t * (ax * G2t * dt2 + ((sinthetadt - thetadtcosthetadt) / (theta3)) * (wx * ax) -
                   ((cos2thetadt - 4 * costhetadt + 3) / (4 * theta4)) * (wx * ax * wx) +
                   ((4 * sinthetadt + sin2thetadt - 4 * thetadtcosthetadt - 2 * thetadt) / (4 * theta5)) * (wx * ax * wx2) +
                   ((thetadt2 - 2 * thetadtsinthetadt - 2 * costhetadt + 2) / (2 * theta4)) * (wx2 * ax) -
                   ((6 * thetadt - 8 * sinthetadt + sin2thetadt) / (4 * theta5)) * (wx2 * ax * wx) +
                   ((2 * thetadt2 - 4 * thetadtsinthetadt - cos2thetadt + 1) / (4 * theta6)) * (wx2 * ax * wx2));

        // Contact-aided 论文附录公式55 & 57
        Eigen::Matrix3d Phi35L =
            G0t *
            (ax * G3t * dt3 - ((thetadtsinthetadt + 2 * costhetadt - 2) / (theta4)) * (wx * ax) -
             ((6 * thetadt - 8 * sinthetadt + sin2thetadt) / (8 * theta5)) * (wx * ax * wx) -
             ((2 * thetadt2 + 8 * thetadtsinthetadt + 16 * costhetadt + cos2thetadt - 17) / (8 * theta6)) *
                 (wx * ax * wx2) +
             ((thetadt3 + 6 * thetadt - 12 * sinthetadt + 6 * thetadtcosthetadt) / (6 * theta5)) * (wx2 * ax) -
             ((6 * thetadt2 + 16 * costhetadt - cos2thetadt - 15) / (8 * theta6)) * (wx2 * ax * wx) +
             ((4 * thetadt3 + 6 * thetadt - 24 * sinthetadt - 3 * sin2thetadt + 24 * thetadtcosthetadt) / (24 * theta7)) *
                 (wx2 * ax * wx2));

        const double tol = 1e-6;
        if (theta < tol)
        {
            Phi25L = (1 / 2) * ax * dt2;
            Phi35L = (1 / 6) * ax * dt3;
        }

        // 求解World Centric右不变，即Contact-aided论文附录公式58
        Eigen::Matrix3d gx = skewSymmetric(RobotState::gravity);
        Eigen::Matrix3d R = state_server.robot_state.getR_GI();
        Eigen::Vector3d v = state_server.robot_state.getv_GI();
        Eigen::Vector3d p = state_server.robot_state.getp_GI();
        Eigen::Matrix3d RG0 = R * G0;
        Eigen::Matrix3d RG1dt = R * G1 * dt;
        Eigen::Matrix3d RG2dt2 = R * G2 * dt2;
        Phi.block<3, 3>(3, 0) = gx * dt;                          // Phi_21
        Phi.block<3, 3>(6, 0) = 0.5 * gx * dt2;                   // Phi_31
        Phi.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt; // Phi_32
        Phi.block<3, 3>(0, dimP_i - dimX_frak) = -RG1dt;          // Phi_15
        Phi.block<3, 3>(3, dimP_i - dimX_frak) =
            -skewSymmetric(v + RG1dt * a + RobotState::gravity * dt) * RG1dt + RG0 * Phi25L; // Phi_25
        Phi.block<3, 3>(6, dimP_i - dimX_frak) =
            -skewSymmetric(p + v * dt + RG2dt2 * a + 0.5 * RobotState::gravity * dt2) * RG1dt + RG0 * Phi35L; // Phi_35
        for (int i = 5; i < dimX_i; ++i)
        {
            Phi.block<3, 3>((i - 2) * 3, dimP_i - dimX_frak) =
                -skewSymmetric(state_server.robot_state.getd_GI(i)) * RG1dt; // Phi_(3+i)5
        }
        Phi.block<3, 3>(3, dimP_i - dimX_frak + 3) = -RG1dt;  // Phi_26
        Phi.block<3, 3>(6, dimP_i - dimX_frak + 3) = -RG2dt2; // Phi_36
        ROS_DEBUG("StateTransitionMatrix Out");
        return Phi;
    }

    Eigen::MatrixXd MsckfVio::DiscreteNoiseMatrix(const Eigen::MatrixXd &Phi, const double dt)
    {
        ROS_DEBUG("DiscreteNoiseMatrix In");
        int dimX_i = state_server.robot_state.dimX_i();
        int dimX_frak = state_server.robot_state.dimX_frak();
        int dimP_i = state_server.robot_state.dimP_i();

        Eigen::MatrixXd X_i = state_server.robot_state.getX_i();
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(dimP_i + 12, dimP_i + 12);
        B.block(0, 0, dimP_i - dimX_frak, dimP_i - dimX_frak) = Adjoint_SEK3(X_i);
        B.block<3, 3>(dimP_i - dimX_frak, dimP_i - dimX_frak) = Matrix3d::Identity();
        B.block<3, 3>(dimP_i - dimX_frak + 3, dimP_i - dimX_frak + 3) = Matrix3d::Identity();

        Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(dimP_i + 12, dimP_i + 12);
        Cov.block<3, 3>(0, 0) = state_server.Qg;                                            // Qg
        Cov.block<3, 3>(3, 3) = state_server.Qa;                                            // Qa
        Cov.block<3, 3>(dimP_i - dimX_frak, dimP_i - dimX_frak) = state_server.Qbg;         // Qbg
        Cov.block<3, 3>(dimP_i - dimX_frak + 3, dimP_i - dimX_frak + 3) = state_server.Qba; // Qba
        // Qc
        for (auto it = state_server.robot_state.estimated_contact_position.begin();
             it != state_server.robot_state.estimated_contact_position.end(); ++it)
        {
            // TODO? 没有乘FkR, 或者保持为0？
            Cov.block<3, 3>(3 + 3 * (it->second - 3), 3 + 3 * (it->second - 3)) = state_server.Qc;
        }

        Eigen::MatrixXd PhiB = Phi * B;
        Eigen::MatrixXd Qd = PhiB * Cov * PhiB.transpose() * dt;
        ROS_DEBUG("DiscreteNoiseMatrix Out");
        return Qd;
    }

    void MsckfVio::InEKF_Correct(const Eigen::MatrixXd &Z, const Eigen::MatrixXd &H, const Eigen::MatrixXd &N)
    {
        ROS_DEBUG("InEKF_Correct In");
        int dimP_i = state_server.robot_state.dimP_i();
        int dimX_frak = state_server.robot_state.dimX_frak();
        const Eigen::MatrixXd &P = state_server.state_cov.block(0, 0, dimP_i + 12, dimP_i + 12);
        ROS_DEBUG_STREAM("state_cov size: " << state_server.state_cov.rows() << " " << state_server.state_cov.cols());
        ROS_DEBUG_STREAM("P size: " << P.rows() << " " << P.cols());
        Eigen::MatrixXd PHT = P * H.transpose();
        ROS_DEBUG_STREAM("PHT size: " << PHT.rows() << " " << PHT.cols());
        Eigen::MatrixXd S = H * PHT + N;
        ROS_DEBUG_STREAM("S size: " << S.rows() << " " << S.cols());
        Eigen::MatrixXd K = PHT * S.inverse();
        ROS_DEBUG_STREAM("K size: " << K.rows() << " " << K.cols());

        Eigen::VectorXd delta = K * Z;
        const Eigen::VectorXd &delta_X_i = delta.head(dimP_i - dimX_frak);
        const Eigen::VectorXd &delta_X_frak = delta.segment(dimP_i - dimX_frak, dimX_frak);
        // TODO: 测试两个外参有没有更新量
        const Eigen::VectorXd &delta_X_ext = delta.segment(dimP_i, 6);
        const Eigen::VectorXd &delta_X_stereo = delta.tail(6);
        Eigen::MatrixXd dX_i = Exp_SEK3(delta_X_i);
        Eigen::VectorXd dX_frak = delta_X_frak;
        Eigen::Matrix4d dT_ext = Exp_SEK3(delta_X_ext);
        Eigen::Matrix4d dT_stereo = Exp_SEK3(delta_X_stereo);
        Eigen::MatrixXd X_i_pred = dX_i * state_server.robot_state.getX_i();
        RobotState::Vector6d X_frak_pred = dX_frak + state_server.robot_state.getX_frak();
        state_server.robot_state.setX_i(X_i_pred);
        // NOTE: 这里drift中把yaw的bias设置为0了, 测试一下
        state_server.robot_state.setX_frak(X_frak_pred);
        // NOTE: 这里就不更新外参了，感觉不会准

        // 更新协方差   // TODO 不更新左上角和右下角和相机关联的部分可以吗
        Eigen::MatrixXd IKH = MatrixXd::Identity(dimP_i + 12, dimP_i + 12) - K * H;
        ROS_DEBUG_STREAM("IKH size: " << IKH.rows() << " " << IKH.cols());
        Eigen::MatrixXd P_new = IKH * P * IKH.transpose() + K * N * K.transpose();
        ROS_DEBUG_STREAM("P_new size: " << P_new.rows() << " " << P_new.cols());
        // NOTE: drift中这里还修改了yaw对应的协方差矩阵，不更新yaw
        state_server.state_cov.block(0, 0, dimP_i + 12, dimP_i + 12) = P_new;

        ROS_DEBUG_STREAM("X_i \n " << state_server.robot_state.getX_i() << endl);

        ROS_DEBUG("InEKF_Correct Out");
    }

    void MsckfVio::RemoveRowAndColumn(Eigen::MatrixXd &M, int index, int remove_dim)
    {
        ROS_DEBUG("RemoveRowAndColumn In");
        unsigned int dimX = M.cols();
        M.block(index, 0, dimX - index - remove_dim, dimX) = M.bottomRows(dimX - index - remove_dim).eval();
        M.block(0, index, dimX, dimX - index - remove_dim) = M.rightCols(dimX - index - remove_dim).eval();
        M.conservativeResize(dimX - remove_dim, dimX - remove_dim);
        ROS_DEBUG("RemoveRowAndColumn Out");
    }

    void MsckfVio::WheellegStateCallback(const dog_msg::WheellegState::ConstPtr &msg)
    {
        ROS_DEBUG("WheellegStateCallback In");
        webotsRealState.time = msg->time.data;
        tf::vectorMsgToEigen(msg->body_acc_in_IMU, webotsRealState.body_acc_in_IMU);
        tf::vectorMsgToEigen(msg->body_gyro_in_IMU, webotsRealState.body_gyro_in_IMU);
        webotsRealState.body_rotation_in_world << msg->body_rotation_in_world[0].data, msg->body_rotation_in_world[1].data, msg->body_rotation_in_world[2].data,
            msg->body_rotation_in_world[3].data, msg->body_rotation_in_world[4].data, msg->body_rotation_in_world[5].data,
            msg->body_rotation_in_world[6].data, msg->body_rotation_in_world[7].data, msg->body_rotation_in_world[8].data;
        tf::vectorMsgToEigen(msg->body_position_in_world, webotsRealState.body_position_in_world);
        tf::vectorMsgToEigen(msg->body_velocity_in_world, webotsRealState.body_velocity_in_world);
        tf::vectorMsgToEigen(msg->FLlun_position_in_world, webotsRealState.FLlun_position_in_world);
        tf::vectorMsgToEigen(msg->RLlun_position_in_world, webotsRealState.RLlun_position_in_world);
        tf::vectorMsgToEigen(msg->RRlun_position_in_world, webotsRealState.RRlun_position_in_world);
        tf::vectorMsgToEigen(msg->FRlun_position_in_world, webotsRealState.FRlun_position_in_world);
        tf::vectorMsgToEigen(msg->FLlun_velocity_in_world, webotsRealState.FLlun_velocity_in_world);
        tf::vectorMsgToEigen(msg->RLlun_velocity_in_world, webotsRealState.RLlun_velocity_in_world);
        tf::vectorMsgToEigen(msg->RRlun_velocity_in_world, webotsRealState.RRlun_velocity_in_world);
        tf::vectorMsgToEigen(msg->FRlun_velocity_in_world, webotsRealState.FRlun_velocity_in_world);
        tf::vectorMsgToEigen(msg->FLlun_velocity_in_body, webotsRealState.FLlun_velocity_in_body);
        tf::vectorMsgToEigen(msg->RLlun_velocity_in_body, webotsRealState.RLlun_velocity_in_body);
        tf::vectorMsgToEigen(msg->RRlun_velocity_in_body, webotsRealState.RRlun_velocity_in_body);
        tf::vectorMsgToEigen(msg->FRlun_velocity_in_body, webotsRealState.FRlun_velocity_in_body);
        webotsRealState.FLlun_Transform_to_body << msg->FLlun_Transform_to_body[0].data, msg->FLlun_Transform_to_body[1].data, msg->FLlun_Transform_to_body[2].data,
            msg->FLlun_Transform_to_body[3].data, msg->FLlun_Transform_to_body[4].data, msg->FLlun_Transform_to_body[5].data,
            msg->FLlun_Transform_to_body[6].data, msg->FLlun_Transform_to_body[7].data, msg->FLlun_Transform_to_body[8].data,
            msg->FLlun_Transform_to_body[9].data, msg->FLlun_Transform_to_body[10].data, msg->FLlun_Transform_to_body[11].data,
            msg->FLlun_Transform_to_body[12].data, msg->FLlun_Transform_to_body[13].data, msg->FLlun_Transform_to_body[14].data,
            msg->FLlun_Transform_to_body[15].data;
        webotsRealState.RLlun_Transform_to_body << msg->RLlun_Transform_to_body[0].data, msg->RLlun_Transform_to_body[1].data, msg->RLlun_Transform_to_body[2].data,
            msg->RLlun_Transform_to_body[3].data, msg->RLlun_Transform_to_body[4].data, msg->RLlun_Transform_to_body[5].data,
            msg->RLlun_Transform_to_body[6].data, msg->RLlun_Transform_to_body[7].data, msg->RLlun_Transform_to_body[8].data,
            msg->RLlun_Transform_to_body[9].data, msg->RLlun_Transform_to_body[10].data, msg->RLlun_Transform_to_body[11].data,
            msg->RLlun_Transform_to_body[12].data, msg->RLlun_Transform_to_body[13].data, msg->RLlun_Transform_to_body[14].data,
            msg->RLlun_Transform_to_body[15].data;
        webotsRealState.RRlun_Transform_to_body << msg->RRlun_Transform_to_body[0].data, msg->RRlun_Transform_to_body[1].data, msg->RRlun_Transform_to_body[2].data,
            msg->RRlun_Transform_to_body[3].data, msg->RRlun_Transform_to_body[4].data, msg->RRlun_Transform_to_body[5].data,
            msg->RRlun_Transform_to_body[6].data, msg->RRlun_Transform_to_body[7].data, msg->RRlun_Transform_to_body[8].data,
            msg->RRlun_Transform_to_body[9].data, msg->RRlun_Transform_to_body[10].data, msg->RRlun_Transform_to_body[11].data,
            msg->RRlun_Transform_to_body[12].data, msg->RRlun_Transform_to_body[13].data, msg->RRlun_Transform_to_body[14].data,
            msg->RRlun_Transform_to_body[15].data;
        webotsRealState.FRlun_Transform_to_body << msg->FRlun_Transform_to_body[0].data, msg->FRlun_Transform_to_body[1].data, msg->FRlun_Transform_to_body[2].data,
            msg->FRlun_Transform_to_body[3].data, msg->FRlun_Transform_to_body[4].data, msg->FRlun_Transform_to_body[5].data,
            msg->FRlun_Transform_to_body[6].data, msg->FRlun_Transform_to_body[7].data, msg->FRlun_Transform_to_body[8].data,
            msg->FRlun_Transform_to_body[9].data, msg->FRlun_Transform_to_body[10].data, msg->FRlun_Transform_to_body[11].data,
            msg->FRlun_Transform_to_body[12].data, msg->FRlun_Transform_to_body[13].data, msg->FRlun_Transform_to_body[14].data,
            msg->FRlun_Transform_to_body[15].data;
        webotsRealState.FLlun_DHJacobian_to_body << msg->FLlun_DHJacobian_to_body[0].data, msg->FLlun_DHJacobian_to_body[1].data, msg->FLlun_DHJacobian_to_body[2].data,
            msg->FLlun_DHJacobian_to_body[3].data, msg->FLlun_DHJacobian_to_body[4].data, msg->FLlun_DHJacobian_to_body[5].data,
            msg->FLlun_DHJacobian_to_body[6].data, msg->FLlun_DHJacobian_to_body[7].data, msg->FLlun_DHJacobian_to_body[8].data,
            msg->FLlun_DHJacobian_to_body[9].data, msg->FLlun_DHJacobian_to_body[10].data, msg->FLlun_DHJacobian_to_body[11].data,
            msg->FLlun_DHJacobian_to_body[12].data, msg->FLlun_DHJacobian_to_body[13].data, msg->FLlun_DHJacobian_to_body[14].data,
            msg->FLlun_DHJacobian_to_body[15].data, msg->FLlun_DHJacobian_to_body[16].data, msg->FLlun_DHJacobian_to_body[17].data;
        webotsRealState.RLlun_DHJacobian_to_body << msg->RLlun_DHJacobian_to_body[0].data, msg->RLlun_DHJacobian_to_body[1].data, msg->RLlun_DHJacobian_to_body[2].data,
            msg->RLlun_DHJacobian_to_body[3].data, msg->RLlun_DHJacobian_to_body[4].data, msg->RLlun_DHJacobian_to_body[5].data,
            msg->RLlun_DHJacobian_to_body[6].data, msg->RLlun_DHJacobian_to_body[7].data, msg->RLlun_DHJacobian_to_body[8].data,
            msg->RLlun_DHJacobian_to_body[9].data, msg->RLlun_DHJacobian_to_body[10].data, msg->RLlun_DHJacobian_to_body[11].data,
            msg->RLlun_DHJacobian_to_body[12].data, msg->RLlun_DHJacobian_to_body[13].data, msg->RLlun_DHJacobian_to_body[14].data,
            msg->RLlun_DHJacobian_to_body[15].data, msg->RLlun_DHJacobian_to_body[16].data, msg->RLlun_DHJacobian_to_body[17].data;
        webotsRealState.RRlun_DHJacobian_to_body << msg->RRlun_DHJacobian_to_body[0].data, msg->RRlun_DHJacobian_to_body[1].data, msg->RRlun_DHJacobian_to_body[2].data,
            msg->RRlun_DHJacobian_to_body[3].data, msg->RRlun_DHJacobian_to_body[4].data, msg->RRlun_DHJacobian_to_body[5].data,
            msg->RRlun_DHJacobian_to_body[6].data, msg->RRlun_DHJacobian_to_body[7].data, msg->RRlun_DHJacobian_to_body[8].data,
            msg->RRlun_DHJacobian_to_body[9].data, msg->RRlun_DHJacobian_to_body[10].data, msg->RRlun_DHJacobian_to_body[11].data,
            msg->RRlun_DHJacobian_to_body[12].data, msg->RRlun_DHJacobian_to_body[13].data, msg->RRlun_DHJacobian_to_body[14].data,
            msg->RRlun_DHJacobian_to_body[15].data, msg->RRlun_DHJacobian_to_body[16].data, msg->RRlun_DHJacobian_to_body[17].data;
        webotsRealState.FRlun_DHJacobian_to_body << msg->FRlun_DHJacobian_to_body[0].data, msg->FRlun_DHJacobian_to_body[1].data, msg->FRlun_DHJacobian_to_body[2].data,
            msg->FRlun_DHJacobian_to_body[3].data, msg->FRlun_DHJacobian_to_body[4].data, msg->FRlun_DHJacobian_to_body[5].data,
            msg->FRlun_DHJacobian_to_body[6].data, msg->FRlun_DHJacobian_to_body[7].data, msg->FRlun_DHJacobian_to_body[8].data,
            msg->FRlun_DHJacobian_to_body[9].data, msg->FRlun_DHJacobian_to_body[10].data, msg->FRlun_DHJacobian_to_body[11].data,
            msg->FRlun_DHJacobian_to_body[12].data, msg->FRlun_DHJacobian_to_body[13].data, msg->FRlun_DHJacobian_to_body[14].data,
            msg->FRlun_DHJacobian_to_body[15].data, msg->FRlun_DHJacobian_to_body[16].data, msg->FRlun_DHJacobian_to_body[17].data;
        webotsRealState.Fourlun_contact = Eigen::Matrix<bool, 4, 1>(msg->Fourlun_contact[0].data, msg->Fourlun_contact[1].data,
                                                                    msg->Fourlun_contact[2].data, msg->Fourlun_contact[3].data);
        ROS_DEBUG("WheellegStateCallback Out");
    }

} // namespace msckf_vio
