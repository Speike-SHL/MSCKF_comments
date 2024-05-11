/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
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

using namespace std;
using namespace Eigen;

namespace msckf_vio
{
    // Static member variables in IMUState class.
    StateIDType IMUState::next_id = 0;
    double IMUState::gyro_noise = 0.001;
    double IMUState::acc_noise = 0.01;
    double IMUState::gyro_bias_noise = 0.001;
    double IMUState::acc_bias_noise = 0.01;
    Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
    Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

    // Static member variables in CAMState class.
    Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

    // Static member variables in Feature class.
    FeatureIDType Feature::next_id = 0;
    double Feature::observation_noise = 0.01;
    Feature::OptimizationConfig Feature::optimization_config;

    map<int, double> MsckfVio::chi_squared_test_table;

    /**
     * @brief MsckfVio构造函数
     * @param pnh Ros节点句柄
     * @param is_gravity_set False 设置未初始化重力
     * @param is_first_img True 设置是第一帧图像
     */
    MsckfVio::MsckfVio(ros::NodeHandle &pnh) : is_gravity_set(false), is_first_img(true), nh(pnh)
    {
        return;
    }

    /**
     * @brief 导入各种参数，包括阈值、传感器误差标准差等
     */
    bool MsckfVio::loadParameters()
    {
        // Frame id
        // 坐标系名字
        nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
        nh.param<string>("child_frame_id", child_frame_id, "robot");

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

        // Noise related parameters
        // imu参数
        nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
        nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
        nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
        nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
        // 特征的噪声
        nh.param<double>("noise/feature", Feature::observation_noise, 0.01);

        // Use variance instead of standard deviation.
        // 方差
        IMUState::gyro_noise *= IMUState::gyro_noise;
        IMUState::acc_noise *= IMUState::acc_noise;
        IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
        IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
        Feature::observation_noise *= Feature::observation_noise;

        // Set the initial IMU state.
        // The intial orientation and position will be set to the origin
        // implicitly. But the initial velocity and bias can be
        // set by parameters.
        // TODO: is it reasonable to set the initial bias to 0?
        // 设置初始化速度为0，那必须一开始处于静止了，也符合msckf的静态初始化了
        nh.param<double>("initial_state/velocity/x", state_server.imu_state.velocity(0), 0.0);
        nh.param<double>("initial_state/velocity/y", state_server.imu_state.velocity(1), 0.0);
        nh.param<double>("initial_state/velocity/z", state_server.imu_state.velocity(2), 0.0);

        // The initial covariance of orientation and position can be
        // set to 0. But for velocity, bias and extrinsic parameters,
        // there should be nontrivial uncertainty.
        // 初始协方差的赋值（误差状态的协方差）
        // 为什么旋转平移就可以是0？因为在正式开始之前我们通过初始化找好了重力方向，确定了第一帧的位姿
        double gyro_bias_cov, acc_bias_cov, velocity_cov;
        nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25);
        nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4);
        nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2);

        double extrinsic_rotation_cov, extrinsic_translation_cov;
        nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4);
        nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4);

        // 0~3 角度
        // 3~6 陀螺仪偏置
        // 6~9 速度
        // 9~12 加速度计偏置
        // 12~15 位移
        // 15~18 外参旋转
        // 18~21 外参平移
        state_server.state_cov = MatrixXd::Zero(21, 21);
        for (int i = 3; i < 6; ++i)
            state_server.state_cov(i, i) = gyro_bias_cov;
        for (int i = 6; i < 9; ++i)
            state_server.state_cov(i, i) = velocity_cov;
        for (int i = 9; i < 12; ++i)
            state_server.state_cov(i, i) = acc_bias_cov;
        for (int i = 15; i < 18; ++i)
            state_server.state_cov(i, i) = extrinsic_rotation_cov;
        for (int i = 18; i < 21; ++i)
            state_server.state_cov(i, i) = extrinsic_translation_cov;

        // Transformation offsets between the frames involved.
        // 外参注意还是从左往右那么看
        Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
        Isometry3d T_cam0_imu = T_imu_cam0.inverse();

        // 关于外参状态的初始值设置
        state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
        state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();

        // 一些其他外参
        CAMState::T_cam0_cam1 =
            utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
        IMUState::T_imu_body =
            utils::getTransformEigen(nh, "T_imu_body").inverse();

        // Maximum number of camera states to be stored
        nh.param<int>("max_cam_state_size", max_cam_state_size, 30);

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
        ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
        ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
        ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
        ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
        ROS_INFO("observation noise: %.10f", Feature::observation_noise);
        ROS_INFO("initial velocity: %f, %f, %f",
                 state_server.imu_state.velocity(0),
                 state_server.imu_state.velocity(1),
                 state_server.imu_state.velocity(2));
        ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
        ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
        ROS_INFO("initial velocity cov: %f", velocity_cov);
        ROS_INFO("initial extrinsic rotation cov: %f", extrinsic_rotation_cov);
        ROS_INFO("initial extrinsic translation cov: %f", extrinsic_translation_cov);

        cout << T_imu_cam0.linear() << endl;
        cout << T_imu_cam0.translation().transpose() << endl;

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

        /// 6. 接收 "mocap_odom"，真值数据，未用到
        mocap_odom_sub = nh.subscribe("mocap_odom", 10, &MsckfVio::mocapOdomCallback, this);
        /// 7. 发布 "gt_odom"，发布由mocap_odom 算出的真实位姿，未用到
        mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("gt_odom", 1);

        return true;
    }

    bool MsckfVio::initialize()
    {
        /// 1. 加载参数 @see MsckfVio::loadParameters()
        if (!loadParameters())
            return false;
        ROS_INFO("Finish loading ROS parameters...");

        /// 2. 设置imu观测的协方差
        state_server.continuous_noise_cov =
            Matrix<double, 12, 12>::Zero();
        state_server.continuous_noise_cov.block<3, 3>(0, 0) =
            Matrix3d::Identity() * IMUState::gyro_noise;
        state_server.continuous_noise_cov.block<3, 3>(3, 3) =
            Matrix3d::Identity() * IMUState::gyro_bias_noise;
        state_server.continuous_noise_cov.block<3, 3>(6, 6) =
            Matrix3d::Identity() * IMUState::acc_noise;
        state_server.continuous_noise_cov.block<3, 3>(9, 9) =
            Matrix3d::Identity() * IMUState::acc_bias_noise;

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

        // 2. 用200个imu数据做静止初始化，不够则不做
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
        state_server.imu_state.gyro_bias =
            sum_angular_vel / imu_msg_buffer.size();
        // IMUState::gravity =
        //   -sum_linear_acc / imu_msg_buffer.size();
        //  This is the gravity in the IMU frame.
        // 3. 计算重力，忽略加速度计的偏置，剩下的就只有重力了，a_measure = R(a_truth - g) + ba + na
        // 因为假设静止，a_truth = 0, 认为噪声抵消，同时ba为小量，因此a_measure = R(-g)
        Vector3d gravity_imu =
            sum_linear_acc / imu_msg_buffer.size();
        std::cout << "gravity_imu: " << gravity_imu.transpose() << std::endl;

        // Initialize the initial orientation, so that the estimation
        // is consistent with the inertial frame.
        // 重力的模长就是重力的大小
        double gravity_norm = gravity_imu.norm();
        // 重力本来的方向
        IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);
        std::cout << "gravity: " << IMUState::gravity.transpose() << std::endl;

        // 求出当前imu状态的重力方向与实际重力方向的旋转 R‘， 为什么加负号？因为测量值是R(-g)，而真实值是g
        Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
            gravity_imu, -IMUState::gravity);
        // 得出姿态
        state_server.imu_state.orientation =
            rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

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
        IMUState &imu_state = state_server.imu_state;
        imu_state.time = 0.0;
        imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
        imu_state.position = Vector3d::Zero();
        imu_state.velocity = Vector3d::Zero();
        imu_state.gyro_bias = Vector3d::Zero();
        imu_state.acc_bias = Vector3d::Zero();
        imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
        imu_state.position_null = Vector3d::Zero();
        imu_state.velocity_null = Vector3d::Zero();

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

        state_server.state_cov = MatrixXd::Zero(21, 21);
        for (int i = 3; i < 6; ++i)
            state_server.state_cov(i, i) = gyro_bias_cov;
        for (int i = 6; i < 9; ++i)
            state_server.state_cov(i, i) = velocity_cov;
        for (int i = 9; i < 12; ++i)
            state_server.state_cov(i, i) = acc_bias_cov;
        for (int i = 15; i < 18; ++i)
            state_server.state_cov(i, i) = extrinsic_rotation_cov;
        for (int i = 18; i < 21; ++i)
            state_server.state_cov(i, i) = extrinsic_translation_cov;

        // Clear all exsiting features in the map.
        map_server.clear();

        // Clear the IMU msg buffer.
        imu_msg_buffer.clear();

        // Reset the starting flags.
        is_gravity_set = false;
        is_first_img = true;

        // Restart the subscribers.
        imu_sub = nh.subscribe("imu", 100, &MsckfVio::imuCallback, this);
        feature_sub = nh.subscribe("features", 40, &MsckfVio::featureCallback, this);

        // TODO: When can the reset fail?
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
        /// 1. 必须经过imu(重力)初始化才能继续进行
        if (!is_gravity_set)
        {
            ROS_WARN_THROTTLE(1, "Wait for the IMU initialization!");
            return;
        }

        /// 2. 接收到第一帧图像后，记录时间，后端开始工作
        if (is_first_img)
        {
            is_first_img = false;
            state_server.imu_state.time = msg->header.stamp.toSec();
        }

        // 调试使用
        static double max_processing_time = 0.0;
        static int critical_time_cntr = 0;
        double processing_start_time = ros::Time::now().toSec();
        
        /// 3. 批量IMU积分，取出上次IMU积分时间到当前特征接收时间中的IMU数据进行积分
        /// @see MsckfVio::batchImuProcessing(const double &time_bound)
        ros::Time start_time = ros::Time::now();
        batchImuProcessing(msg->header.stamp.toSec());
        double imu_processing_time = (ros::Time::now() - start_time).toSec();

        /// 4. 状态增广，包括名义状态增广和误差协方差矩阵P的增广，主要是增广新的相机状态
        /// @see MsckfVio::stateAugmentation(const double &time)
        start_time = ros::Time::now();
        stateAugmentation(msg->header.stamp.toSec());
        double state_augmentation_time = (ros::Time::now() - start_time).toSec();

        /// 5. 向map_server中添加新的特征，和旧特征在新相机帧上的观测
        /// @see MsckfVio::addFeatureObservations(const CameraMeasurementConstPtr &msg)
        start_time = ros::Time::now();
        addFeatureObservations(msg);
        double add_observations_time = (ros::Time::now() - start_time).toSec();

        /// 6. 使用当前帧没有跟踪上的点进行ESKF的状态更新
        /// @see MsckfVio::removeLostFeatures()
        start_time = ros::Time::now();
        removeLostFeatures();
        double remove_lost_features_time = (ros::Time::now() - start_time).toSec();

        /// 7. 当状态量中的相机状态数达到最大值时，挑出若干cam状态进行滑窗删除
        /// 同时删除前再利用有效的特征点进行一次更新
        /// @see MsckfVio::pruneCamStateBuffer()
        start_time = ros::Time::now();
        pruneCamStateBuffer();
        double prune_cam_states_time = (ros::Time::now() - start_time).toSec();

        /// 8. 发布位姿，点云，tf以及协方差
        /// @see MsckfVio::publish(const ros::Time &time)
        start_time = ros::Time::now();
        publish(msg->header.stamp);
        double publish_time = (ros::Time::now() - start_time).toSec();

        /// 9. 根据IMU状态位置协方差判断是否重置整个系统
        /// @see MsckfVio::onlineReset()
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
            // printf("IMU processing time: %f/%f\n",
            //     imu_processing_time, imu_processing_time/processing_time);
            // printf("State augmentation time: %f/%f\n",
            //     state_augmentation_time, state_augmentation_time/processing_time);
            // printf("Add observations time: %f/%f\n",
            //     add_observations_time, add_observations_time/processing_time);
            printf("Remove lost features time: %f/%f\n",
                   remove_lost_features_time, remove_lost_features_time / processing_time);
            printf("Remove camera states time: %f/%f\n",
                   prune_cam_states_time, prune_cam_states_time / processing_time);
            // printf("Publish time: %f/%f\n",
            //     publish_time, publish_time/processing_time);
        }

        return;
    }

    /**
     * @brief 没用，暂时不看
     */
    void MsckfVio::mocapOdomCallback(
        const nav_msgs::OdometryConstPtr &msg)
    {
        static bool first_mocap_odom_msg = true;

        // If this is the first mocap odometry messsage, set
        // the initial frame.
        if (first_mocap_odom_msg)
        {
            Quaterniond orientation;
            Vector3d translation;
            tf::pointMsgToEigen(
                msg->pose.pose.position, translation);
            tf::quaternionMsgToEigen(
                msg->pose.pose.orientation, orientation);
            // tf::vectorMsgToEigen(
            //     msg->transform.translation, translation);
            // tf::quaternionMsgToEigen(
            //     msg->transform.rotation, orientation);
            mocap_initial_frame.linear() = orientation.toRotationMatrix();
            mocap_initial_frame.translation() = translation;
            first_mocap_odom_msg = false;
        }

        // Transform the ground truth.
        Quaterniond orientation;
        Vector3d translation;
        // tf::vectorMsgToEigen(
        //     msg->transform.translation, translation);
        // tf::quaternionMsgToEigen(
        //     msg->transform.rotation, orientation);
        tf::pointMsgToEigen(
            msg->pose.pose.position, translation);
        tf::quaternionMsgToEigen(
            msg->pose.pose.orientation, orientation);

        Eigen::Isometry3d T_b_v_gt;
        T_b_v_gt.linear() = orientation.toRotationMatrix();
        T_b_v_gt.translation() = translation;
        Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

        // Eigen::Vector3d body_velocity_gt;
        // tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
        // body_velocity_gt = mocap_initial_frame.linear().transpose() *
        //   body_velocity_gt;

        // Ground truth tf.
        if (publish_tf)
        {
            tf::Transform T_b_w_gt_tf;
            tf::transformEigenToTF(T_b_w_gt, T_b_w_gt_tf);
            tf_pub.sendTransform(tf::StampedTransform(
                T_b_w_gt_tf, msg->header.stamp, fixed_frame_id, child_frame_id + "_mocap"));
        }

        // Ground truth odometry.
        nav_msgs::Odometry mocap_odom_msg;
        mocap_odom_msg.header.stamp = msg->header.stamp;
        mocap_odom_msg.header.frame_id = fixed_frame_id;
        mocap_odom_msg.child_frame_id = child_frame_id + "_mocap";

        tf::poseEigenToMsg(T_b_w_gt, mocap_odom_msg.pose.pose);
        // tf::vectorEigenToMsg(body_velocity_gt,
        //     mocap_odom_msg.twist.twist.linear);

        mocap_odom_pub.publish(mocap_odom_msg);
        return;
    }

    /**
     * @brief imu积分，批量处理imu数据
     * @param  time_bound 从state_server.imu_state.time上次处理到这个时间
     */
    void MsckfVio::batchImuProcessing(const double &time_bound)
    {
        int used_imu_msg_cntr = 0;

        /// 1. 遍历imu_msg_buffer中的所有imu数据，找到state_server.imu_state.time到
        /// 当前前端特征时间之间的imu数据，然后进行状态递推
        /// @see MsckfVio::processModel(const double &time, const Eigen::Vector3d &m_gyro, const Eigen::Vector3d &m_acc)
        for (const auto &imu_msg : imu_msg_buffer)
        {
            double imu_time = imu_msg.header.stamp.toSec();
            // 小于，说明这个数据比较旧，因为state_server.imu_state.time代表已经处理过的imu数据的时间
            if (imu_time < state_server.imu_state.time)
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

        /// 2. 更新IMU状态的id  state_server.imu_state.id, 相机状态id也根据这个赋值
        state_server.imu_state.id = IMUState::next_id++;

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
        IMUState &imu_state = state_server.imu_state;

        /// 2. 角速度和加速度减去偏置，计算dt
        Vector3d gyro = m_gyro - imu_state.gyro_bias;
        Vector3d acc = m_acc - imu_state.acc_bias; // acc_bias 初始值是0
        double dtime = time - imu_state.time;

        /// 3. 计算F阵和G阵，见笔记pdf中《IMU误差状态方程总结》
        // Compute discrete transition and noise covariance matrix
        Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
        Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();

        // 误差为真值（观测） - 预测
        // F矩阵表示的是误差的导数的微分方程，其实就是想求δ的递推公式
        // δ`= F · δ + G    δ表示误差
        // δn+1 = (I + F·dt)·δn + G·dt·Q    Q表示imu噪声

        // 两种推法，一种是通过论文中四元数的推，这个过程不再重复
        // 下面给出李代数推法，我们知道msckf的四元数使用的是反人类的jpl
        // 也就是同一个数值的旋转四元数经过两种不同定义得到的旋转是互为转置的
        // 这里面的四元数转成的旋转表示的是Riw，所以要以李代数推的话也要按照Riw推
        // 按照下面的旋转更新方式为左乘，因此李代数也要用左乘，且jpl模式下左乘一个δq = Exp(-δθ)
        // δQj * Qjw = Exp(-(w - b) * t) * δQi * Qiw
        // Exp(-δθj) * Qjw = Exp(-(w - b) * t) * Exp(-δθi) * Qiw
        // 其中Qjw = Exp(-(w - b) * t) * Qiw
        // 两边除得到 Exp(-δθj) = Exp(-(w - b) * t) * Exp(-δθi) * Exp(-(w - b) * t).t()
        // -δθj = - Exp(-(w - b) * t) * δθi
        // δθj = (I - (w - b)^ * t)  * δθi  得证

        // 关于偏置一样可以这么算，只不过式子变成了
        // δQj * Qjw = Exp(-(w - b - δb) * t) * Qiw
        // 上式使用bch近似公式可以得到 δθj = -t * δb
        // 其他也可以通过这个方法推出，是正确的

        F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
        F.block<3, 3>(0, 3) = -Matrix3d::Identity();

        F.block<3, 3>(6, 0) =
            -quaternionToRotation(imu_state.orientation).transpose() * skewSymmetric(acc);
        F.block<3, 3>(6, 9) = -quaternionToRotation(imu_state.orientation).transpose();
        F.block<3, 3>(12, 6) = Matrix3d::Identity();

        G.block<3, 3>(0, 0) = -Matrix3d::Identity();
        G.block<3, 3>(3, 3) = Matrix3d::Identity();
        G.block<3, 3>(6, 6) = -quaternionToRotation(imu_state.orientation).transpose();
        // G.block<3, 3>(6, 6) = -Matrix3d::Identity();
        G.block<3, 3>(9, 9) = Matrix3d::Identity();

        // Approximate matrix exponential to the 3rd order,
        // which can be considered to be accurate enough assuming
        // dtime is within 0.01s.
        Matrix<double, 21, 21> Fdt = F * dtime;
        Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
        Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;

        /// 3. 计算转移矩阵Phi矩阵，使用三阶近似，当dt<0.01s时，这个近似是足够准确的
        /// 见笔记pdf中《误差状态转移矩阵和过程噪声协方差矩阵》
        Matrix<double, 21, 21> Phi =
            Matrix<double, 21, 21>::Identity() + Fdt +
            0.5 * Fdt_square + (1.0 / 6.0) * Fdt_cube;

        /// 4. 四阶龙格库塔积分预测名义状态，旋转，速度，位置 
        /// @see MsckfVio::predictNewState(const double &dt, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc)
        predictNewState(dtime, gyro, acc);

        /// 5. 可观性约束OC，通过修改Phi阵，保证零空间秩为4。
        /// 见论文《Observability-constrained Vision-aided Inertial Navigation》中公式20~23
        // 5.1 修改phi_11
        // imu_state.orientation_null为上一个imu数据递推后保存的
        // 这块可能会有疑问，因为当上一个imu假如被观测更新了，
        // 导致当前的imu状态是由更新后的上一个imu状态递推而来，但是这里的值是没有更新的，这个有影响吗
        // 答案是没有的，因为我们更改了phi矩阵，保证了零空间
        // 并且这里必须这么处理，因为如果使用更新后的上一个imu状态构建上一时刻的零空间
        // 就破坏了上上一个跟上一个imu状态之间的0空间
        // Ni-1 = phi_[i-2] * Ni-2
        // Ni = phi_[i-1] * Ni-1^
        // 如果像上面这样约束，那么中间的0空间就“崩了”
        Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
        Phi.block<3, 3>(0, 0) =
            quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

        // 5.2 修改phi_31
        Vector3d u = R_kk_1 * IMUState::gravity;
        RowVector3d s = (u.transpose() * u).inverse() * u.transpose();
        Matrix3d A1 = Phi.block<3, 3>(6, 0);
        Vector3d w1 =
            skewSymmetric(imu_state.velocity_null - imu_state.velocity) * IMUState::gravity;
        Phi.block<3, 3>(6, 0) = A1 - (A1 * u - w1) * s;

        // 5.3 修改phi_51
        Matrix3d A2 = Phi.block<3, 3>(12, 0);
        Vector3d w2 =
            skewSymmetric(
                dtime * imu_state.velocity_null + imu_state.position_null -
                imu_state.position) *
            IMUState::gravity;
        Phi.block<3, 3>(12, 0) = A2 - (A2 * u - w2) * s;

        /// 6. 使用OC后的Phi阵计算过程噪声协方差矩阵Q, 见笔记pdf中《误差状态转移矩阵和过程噪声协方差矩阵》
        Matrix<double, 21, 21> Q =
            Phi * G * state_server.continuous_noise_cov * G.transpose() * Phi.transpose() * dtime;

        /// 7. 预测系统误差状态协方差矩阵P，如果有相机状态量，那么也更新imu状态量与相机状态量交叉的部分
        /// 见笔记pdf中《预测系统状态协方差矩阵P》
        state_server.state_cov.block<21, 21>(0, 0) =
            Phi * state_server.state_cov.block<21, 21>(0, 0) * Phi.transpose() + Q;
        if (state_server.cam_states.size() > 0)
        {
            // 起点是0 21  然后是21行 state_server.state_cov.cols() - 21 列的矩阵
            // 也就是整个协方差矩阵的右上角，这部分说白了就是imu状态量与相机状态量的协方差，imu更新了，这部分也需要更新
            state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21) =
                Phi * state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21);

            // 同理，这个是左下角
            state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) =
                state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) *
                Phi.transpose();
        }

        /// 8. 强制对称，因为协方差矩阵就是对称的
        MatrixXd state_cov_fixed =
            (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
        state_server.state_cov = state_cov_fixed;

        /// 9. 更新imu旋转，位置和速度的零空间，其实就是记录此次状态预估后的状态，用于下一次对Phi进行OC
        imu_state.orientation_null = imu_state.orientation;
        imu_state.position_null = imu_state.position;
        imu_state.velocity_null = imu_state.velocity;

        /// 10. 更新imu状态的时间state_server.imu_state.time
        state_server.imu_state.time = time;
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

        // TODO: Will performing the forward integration using
        //    the inverse of the quaternion give better accuracy?

        // 角速度，标量
        double gyro_norm = gyro.norm();
        Matrix4d Omega = Matrix4d::Zero();
        Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
        Omega.block<3, 1>(0, 3) = gyro;
        Omega.block<1, 3>(3, 0) = -gyro;

        Vector4d &q = state_server.imu_state.orientation;
        Vector3d &v = state_server.imu_state.velocity;
        Vector3d &p = state_server.imu_state.position;

        // Some pre-calculation
        // dq_dt表示积分n到n+1
        // dq_dt2表示积分n到n+0.5 算龙格库塔用的
        Vector4d dq_dt, dq_dt2;
        if (gyro_norm > 1e-5)
        {
            dq_dt =
                (cos(gyro_norm * dt * 0.5) * Matrix4d::Identity() +
                 1 / gyro_norm * sin(gyro_norm * dt * 0.5) * Omega) *
                q;
            dq_dt2 =
                (cos(gyro_norm * dt * 0.25) * Matrix4d::Identity() +
                 1 / gyro_norm * sin(gyro_norm * dt * 0.25) * Omega) *
                q;
        }
        else
        {
            // 当角增量很小时的近似
            dq_dt = (Matrix4d::Identity() + 0.5 * dt * Omega) * cos(gyro_norm * dt * 0.5) * q;
            dq_dt2 = (Matrix4d::Identity() + 0.25 * dt * Omega) * cos(gyro_norm * dt * 0.25) * q;
        }
        // Rwi
        Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
        Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

        // k1 = f(tn, yn)
        Vector3d k1_v_dot = quaternionToRotation(q).transpose() * acc + IMUState::gravity;
        Vector3d k1_p_dot = v;

        // k2 = f(tn+dt/2, yn+k1*dt/2)
        // 这里的4阶LK法用了匀加速度假设，即认为前一时刻的加速度和当前时刻相等
        Vector3d k1_v = v + k1_v_dot * dt / 2;
        Vector3d k2_v_dot = dR_dt2_transpose * acc + IMUState::gravity;
        Vector3d k2_p_dot = k1_v;

        // k3 = f(tn+dt/2, yn+k2*dt/2)
        Vector3d k2_v = v + k2_v_dot * dt / 2;
        Vector3d k3_v_dot = dR_dt2_transpose * acc + IMUState::gravity;
        Vector3d k3_p_dot = k2_v;

        // k4 = f(tn+dt, yn+k3*dt)
        Vector3d k3_v = v + k3_v_dot * dt;
        Vector3d k4_v_dot = dR_dt_transpose * acc + IMUState::gravity;
        Vector3d k4_p_dot = k3_v;

        // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        q = dq_dt;
        quaternionNormalize(q);
        v = v + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
        p = p + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);

        return;
    }

    /**
     * @brief 状态增广，向状态中增加新的相机状态，同时增广误差协方差矩阵P
     * @param  time 图片的时间戳
     */
    void MsckfVio::stateAugmentation(const double &time)
    {
        /// 1. 根据现在的IMU状态以及IMU和左目相机的外参，预估出当前相机的名义状态
        /// 见笔记pdf中《名义状态扩增》
        // 1.1 取出状态量中的外参，老规矩R_i_c 按照常理我们应该叫他 Rci imu到cam0的旋转
        const Matrix3d &R_i_c = state_server.imu_state.R_imu_cam0;
        const Vector3d &t_c_i = state_server.imu_state.t_cam0_imu;

        // 1.2 取出imu旋转平移，按照外参，将这个时刻cam0的位姿算出来
        Matrix3d R_w_i = quaternionToRotation(
            state_server.imu_state.orientation);
        Matrix3d R_w_c = R_i_c * R_w_i;
        Vector3d t_c_w = state_server.imu_state.position +
                         R_w_i.transpose() * t_c_i;

        /// 2. 注册新的相机状态到状态库state_server中, 
        /// 包括id(使用此时的imu状态id作为该帧相机的id), 时间戳，位姿
        /// QUERY 以及用于OC的零空间(第一次相机帧估计的数据)
        state_server.cam_states[state_server.imu_state.id] =
            CAMState(state_server.imu_state.id);
        CAMState &cam_state = state_server.cam_states[state_server.imu_state.id];

        cam_state.time = time;
        cam_state.orientation = rotationToQuaternion(R_w_c);
        cam_state.position = t_c_w;

        // 记录第一次被估计的数据，不能被改变，因为改变了就破坏了之前的0空间
        cam_state.orientation_null = cam_state.orientation;
        cam_state.position_null = cam_state.position;

        /// 3. 计算用于增广协方差矩阵P的雅可比矩阵J，见笔记pdf中《求雅可比矩阵 J_I》
        // 此时我们首先要知道相机位姿是 Rcw  twc
        // Rcw = Rci * Riw   twc = twi + Rwi * tic
        Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
        // Rcw对Riw的左扰动导数
        J.block<3, 3>(0, 0) = R_i_c;
        // Rcw对Rci的左扰动导数
        J.block<3, 3>(0, 15) = Matrix3d::Identity();

        // twc对Riw的左扰动导数
        // twc = twi + Rwi * Exp(φ) * tic
        //     = twi + Rwi * (I + φ^) * tic
        //     = twi + Rwi * tic + Rwi * φ^ * tic
        //     = twi + Rwi * tic - Rwi * tic^ * φ
        // 这部分的偏导为 -Rwi * tic^     与论文一致
        // TODO 试一下 -R_w_i.transpose() * skewSymmetric(t_c_i)
        // 其实这里可以反过来推一下当给的扰动是
        // twc = twi + Exp(-φ) * Rwi * tic
        //     = twi + (I - φ^) * Rwi * tic
        //     = twi + Rwi * tic - φ^ * Rwi * tic
        //     = twi + Rwi * tic + (Rwi * tic)^ * φ
        // 这样跟代码就一样了，但是上下定义的扰动方式就不同了
        J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose() * t_c_i);
        // 下面是代码里自带的，论文中给出的也是下面的结果
        // J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
        // J.block<3,3>(3,0) = R_w_i.transpose() * skewSymmetric(t_c_i);
        // std::cout << "(Rwi * tic)^" << skewSymmetric(R_w_i.transpose() * t_c_i) << std::endl;
        // std::cout << "-Rwi * tic^" << -R_w_i.transpose() * skewSymmetric(t_c_i) << std::endl;
        // std::cout << "Rwi * tic^" << R_w_i.transpose() * skewSymmetric(t_c_i) << std::endl;

        // twc对twi的左扰动导数
        J.block<3, 3>(3, 12) = Matrix3d::Identity();
        // twc对tic的左扰动导数
        J.block<3, 3>(3, 18) = R_w_i.transpose();

        /// 4. 增广误差协方差矩阵P，见笔记pdf中《误差协方差矩阵增广》
        // 简单地说就是原来的协方差是 21 + 6n 维的，现在新来了一个伙计，维度要扩了
        // 并且对应位置的值要根据雅可比跟这个时刻（也就是最新时刻）的imu协方差计算
        // 4.1 扩展矩阵大小 conservativeResize函数不改变原矩阵对应位置的数值
        // Resize the state covariance matrix.
        size_t old_rows = state_server.state_cov.rows();
        size_t old_cols = state_server.state_cov.cols();
        state_server.state_cov.conservativeResize(old_rows + 6, old_cols + 6);

        // imu的协方差矩阵
        const Matrix<double, 21, 21> &P11 =
            state_server.state_cov.block<21, 21>(0, 0);

        // imu相对于各个相机状态量的协方差矩阵（不包括最新的）
        const MatrixXd &P12 =
            state_server.state_cov.block(0, 21, 21, old_cols - 21);

        // Fill in the augmented state covariance.
        // 4.2 计算协方差矩阵
        // 左下角
        state_server.state_cov.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;

        // 右上角
        state_server.state_cov.block(0, old_cols, old_rows, 6) =
            state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();

        // 右下角，关于相机部分的J都是0所以省略了
        state_server.state_cov.block<6, 6>(old_rows, old_cols) =
            J * P11 * J.transpose();

        /// 5. 进行强制对称
        MatrixXd state_cov_fixed = (state_server.state_cov +
                                    state_server.state_cov.transpose()) /
                                   2.0;
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
        StateIDType state_id = state_server.imu_state.id;

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
     * @brief 使用当前帧没有跟踪上的点进行ESKF的状态更新，并从地图中删除无用的点
     */
    void MsckfVio::removeLostFeatures()
    {
        // Remove the features that lost track.
        // BTW, find the size the final Jacobian matrix and residual vector.
        int jacobian_row_size = 0;
        vector<FeatureIDType> invalid_feature_ids(0);   // 无效点，最后要删的
        vector<FeatureIDType> processed_feature_ids(0); // 待参与更新的点，用完也被无情的删掉

        /// 1. 遍历地图(map_server)中的所有特征点，先找到当前帧中跟踪丢失的点，再从这些点中
        /// 从地图中删除跟踪小于3帧的点，再从剩下点中尝试利用多帧观测三角化其三维坐标，删除三角化失败的点，
        /// 最后剩下的点就是可以用来更新的点，准备进行更新。(如果没有点可以用来更新，那么就直接返回)
        for (auto iter = map_server.begin();
             iter != map_server.end(); ++iter)
        {
            // 引用，改变feature相当于改变iter->second，类似于指针的效果
            auto &feature = iter->second;

            // 这个点被当前状态观测到，说明这个点后面还有可能被跟踪，跳过这些点
            if (feature.observations.find(state_server.imu_state.id) !=
                feature.observations.end())
                continue;

            // 跟踪小于3帧的点，认为是质量不高的点，也好理解，三角化起码要两个观测，但是只有两个没有其他观测来验证
            if (feature.observations.size() < 3)
            {
                invalid_feature_ids.push_back(feature.id);
                continue;
            }

            // 如果这个特征没有被初始化，尝试去初始化，初始化就是三角化
            if (!feature.is_initialized)
            {
                // 看看运动是否足够，没有足够视差或者平移小旋转多这种不符合三角化，所以就不要这些点了
                if (!feature.checkMotion(state_server.cam_states))
                {
                    invalid_feature_ids.push_back(feature.id);
                    continue;
                }
                else
                {
                    // 尝试三角化，失败也不要了
                    if (!feature.initializePosition(state_server.cam_states))
                    {
                        invalid_feature_ids.push_back(feature.id);
                        continue;
                    }
                }
            }

            // 到这里表示这个点能用于更新，所以准备下一步计算
            // 一个观测代表一帧，一帧有左右两个观测
            // 也就是算重投影误差时维度将会是4 * feature.observations.size()
            // 这里为什么减3下面会提到
            jacobian_row_size += 4 * feature.observations.size() - 3;
            // 接下来要参与优化的点加入到这个变量中
            processed_feature_ids.push_back(feature.id);
        }

        // cout << "invalid/processed feature #: " <<
        //   invalid_feature_ids.size() << "/" <<
        //   processed_feature_ids.size() << endl;
        // cout << "jacobian row #: " << jacobian_row_size << endl;

        // 删掉非法点
        for (const auto &feature_id : invalid_feature_ids)
            map_server.erase(feature_id);

        // Return if there is no lost feature to be processed.
        if (processed_feature_ids.size() == 0)
            return;

        /// 2. 根据待更新的特征点和相机观测数量，准备好总的重投影误差和观测雅可比矩阵
        /// \f$\mathbf{r}_o\f$ 和 \f$\mathbf{H}_{\mathbf{x}o}\f$ 的维度
        /// 见笔记pdf中《累积所有特征点的所有观测》
        MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                          21 + 6 * state_server.cam_states.size());
        VectorXd r = VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;

        /// 3. 遍历所有待更新的特征点，找到其对应的相机观测，并计算一个特征点的所有相机观测
        /// @see MsckfVio::featureJacobian(const FeatureIDType &feature_id, const std::vector<StateIDType> &cam_state_ids, Eigen::MatrixXd &H_x, Eigen::VectorXd &r)
        for (const auto &feature_id : processed_feature_ids)
        {
            auto &feature = map_server[feature_id];

            vector<StateIDType> cam_state_ids(0);
            for (const auto &measurement : feature.observations)
                cam_state_ids.push_back(measurement.first);

            MatrixXd H_xj;
            VectorXd r_j;
            // 计算雅可比，计算重投影误差
            featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

            // 卡方检验，剔除错误点，并不是所有点都用
            if (gatingTest(H_xj, r_j, cam_state_ids.size() - 1))
            {
                /// 4. 组合所有点的观测为最终的\f$\mathbf{r}_o\f$ 和 \f$\mathbf{H}_{\mathbf{x}o}\f$
                H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
                r.segment(stack_cntr, r_j.rows()) = r_j;
                stack_cntr += H_xj.rows();
            }

            /// 5. 为了保证速度，限制重投影误差和观测雅可比矩阵的行数不超过1500行，resize为1500行
            if (stack_cntr > 1500)
                break;
        }

        // resize成实际大小
        H_x.conservativeResize(stack_cntr, H_x.cols());
        r.conservativeResize(stack_cntr);

        /// 6. 使用\f$\mathbf{r}_o\f$ 和 \f$\mathbf{H}_{\mathbf{x}o}\f$进行ESKF更新
        /// @see MsckfVio::measurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r)
        measurementUpdate(H_x, r);

        /// 7. 从地图中删除更新后的点
        for (const auto &feature_id : processed_feature_ids)
            map_server.erase(feature_id);

        return;
    }

    /**
     * @brief 使用ESKF进行最终的状态更新，更新前还使用了QR分解简化雅可比矩阵和重投影误差
     * 见笔记pdf中《QR分解简化最终观测雅可比和误差》以及 《误差扩展卡尔曼滤波进行更新》
     * @param  H 雅可比
     * @param  r 误差
     */
    void MsckfVio::measurementUpdate(
        const MatrixXd &H, const VectorXd &r)
    {
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

            H_thin = H_temp.topRows(21 + state_server.cam_states.size() * 6);
            r_thin = r_temp.head(21 + state_server.cam_states.size() * 6);

            // HouseholderQR<MatrixXd> qr_helper(H);
            // MatrixXd Q = qr_helper.householderQ();
            // MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

            // H_thin = Q1.transpose() * H;
            // r_thin = Q1.transpose() * r;
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
        // MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
        MatrixXd K_transpose = S.ldlt().solve(H_thin * P);
        MatrixXd K = K_transpose.transpose();

        // Compute the error of the state.
        VectorXd delta_x = K * r_thin;

        // Update the IMU state.
        const VectorXd &delta_x_imu = delta_x.head<21>();

        if ( // delta_x_imu.segment<3>(0).norm() > 0.15 ||
             // delta_x_imu.segment<3>(3).norm() > 0.15 ||
            delta_x_imu.segment<3>(6).norm() > 0.5 ||
            // delta_x_imu.segment<3>(9).norm() > 0.5 ||
            delta_x_imu.segment<3>(12).norm() > 1.0)
        {
            printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
            printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
            ROS_WARN("Update change is too large.");
            // return;
        }

        // 3. 更新到imu状态量
        const Vector4d dq_imu =
            smallAngleQuaternion(delta_x_imu.head<3>());
        // 相当于左乘dq_imu
        state_server.imu_state.orientation = quaternionMultiplication(
            dq_imu, state_server.imu_state.orientation);
        state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
        state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
        state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
        state_server.imu_state.position += delta_x_imu.segment<3>(12);

        // 外参
        const Vector4d dq_extrinsic =
            smallAngleQuaternion(delta_x_imu.segment<3>(15));
        state_server.imu_state.R_imu_cam0 =
            quaternionToRotation(dq_extrinsic) * state_server.imu_state.R_imu_cam0;
        state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

        // 更新相机姿态
        auto cam_state_iter = state_server.cam_states.begin();
        for (int i = 0; i < state_server.cam_states.size(); ++i, ++cam_state_iter)
        {
            const VectorXd &delta_x_cam = delta_x.segment<6>(21 + i * 6);
            const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
            cam_state_iter->second.orientation = quaternionMultiplication(
                dq_cam, cam_state_iter->second.orientation);
            cam_state_iter->second.position += delta_x_cam.tail<3>();
        }

        // Update state covariance.
        // 4. 更新协方差
        MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
        // state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
        //   K*K.transpose()*Feature::observation_noise;
        state_server.state_cov = I_KH * state_server.state_cov;

        // Fix the covariance to be symmetric
        MatrixXd state_cov_fixed = (state_server.state_cov +
                                    state_server.state_cov.transpose()) /
                                   2.0;
        state_server.state_cov = state_cov_fixed;

        return;
    }

    /**
     * @brief 计算一个特征点的所有相机观测的雅可比和重投影误差。还包括左零空间投影(边缘化)
     * 见笔记pdf中《累积一个特征点的所有相机观测并进行左零空间投影(边缘化)》
     * @param  [in]  feature_id 特征点id
     * @param  [in]  cam_state_ids 观测到该特征点的所有相机状态id
     * @param  [out] H_x 观测雅可比
     * @param  [out] r 重投影误差
     */
    void MsckfVio::featureJacobian(
        const FeatureIDType &feature_id,
        const std::vector<StateIDType> &cam_state_ids,
        MatrixXd &H_x, VectorXd &r)
    {
        /// 1. 取出特征点和有效的相机观测
        const auto &feature = map_server[feature_id];
        vector<StateIDType> valid_cam_state_ids(0);
        for (const auto &cam_id : cam_state_ids)
        {
            if (feature.observations.find(cam_id) ==
                feature.observations.end())
                continue;
            valid_cam_state_ids.push_back(cam_id);
        }

        /// 2. 准备重投影误差矩阵和相对于相机状态与三维点的雅可比矩阵
        int jacobian_row_size = 0;
        jacobian_row_size = 4 * valid_cam_state_ids.size();
        MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
                                       21 + state_server.cam_states.size() * 6);
        MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
        VectorXd r_j = VectorXd::Zero(jacobian_row_size);
        int stack_cntr = 0;

        /// 3. 遍历该特征点的所有相机观测，先计算一个特征点相对于一个相机状态的雅可比和重投影误差
        /// \f$\mathbf{r}^j_i\f$, \f$\mathbf{H}^j_{C_i}\f$ 和 \f$\mathbf{H}^j_{f_i}\f$
        /// @see MsckfVio::measurementJacobian(const StateIDType &cam_state_id, const FeatureIDType &feature_id, Eigen::Matrix<double, 4, 6> &H_x, Eigen::Matrix<double, 4, 3> &H_f, Eigen::Vector4d &r)
        ///
        /// 再累积该特征点相对于所有相机状态的雅可比和重投影误差
        /// \f$\mathbf{r}^j\f$, \f$\mathbf{H}^j_x\f$ 和 \f$\mathbf{H}^j_f\f$
        for (const auto &cam_id : valid_cam_state_ids)
        {

            Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
            Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
            Vector4d r_i = Vector4d::Zero();
            // 计算一个左右目观测的雅可比
            measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

            // 计算这个cam_id在整个矩阵的列数，因为要在大矩阵里面放
            auto cam_state_iter = state_server.cam_states.find(cam_id);
            int cam_state_cntr = std::distance(
                state_server.cam_states.begin(), cam_state_iter);

            // Stack the Jacobians.
            H_xj.block<4, 6>(stack_cntr, 21 + 6 * cam_state_cntr) = H_xi;
            H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
            r_j.segment<4>(stack_cntr) = r_i;
            stack_cntr += 4;
        }

        /// 4. 使用SVD分解进行边缘化(左零空间投影)，QR分解也可以
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
     * @brief 计算一个特征点相对于一个相机帧的雅可比和重投影误差
     * 见笔记pdf中《观测误差的雅可比》
     * @param  [in] cam_state_id 观测到该特征点的一个相机状态id
     * @param  [in] feature_id 特征点id
     * @param  [out] H_x 重投影误差相对于位姿的雅可比
     * @param  [out] H_f 重投影误差相对于三维点的雅可比
     * @param  [out] r 重投影误差
     */
    void MsckfVio::measurementJacobian(
        const StateIDType &cam_state_id,
        const FeatureIDType &feature_id,
        Matrix<double, 4, 6> &H_x, Matrix<double, 4, 3> &H_f, Vector4d &r)
    {

        // Prepare all the required data.
        // 1. 取出相机状态与特征
        const CAMState &cam_state = state_server.cam_states[cam_state_id];
        const Feature &feature = map_server[feature_id];

        // 2. 取出左目位姿，根据外参计算右目位姿
        // Cam0 pose.
        Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
        const Vector3d &t_c0_w = cam_state.position;

        // Cam1 pose.
        Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
        Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
        Vector3d t_c1_w = t_c0_w - R_w_c1.transpose() * CAMState::T_cam0_cam1.translation();

        // 3. 取出三维点坐标与归一化的坐标点，因为前端发来的是归一化坐标的
        // 3d feature position in the world frame.
        // And its observation with the stereo cameras.
        const Vector3d &p_w = feature.position;
        const Vector4d &z = feature.observations.find(cam_state_id)->second;

        // 4. 转到左右目相机坐标系下
        // Convert the feature position from the world frame to
        // the cam0 and cam1 frame.
        Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);
        Vector3d p_c1 = R_w_c1 * (p_w - t_c1_w);
        // p_c1 = R_c0_c1 * R_w_c0 * (p_w - t_c0_w + R_w_c1.transpose() * t_cam0_cam1)
        //      = R_c0_c1 * (p_c0 + R_w_c0 * R_w_c1.transpose() * t_cam0_cam1)
        //      = R_c0_c1 * (p_c0 + R_c0_c1 * t_cam0_cam1)

        // Compute the Jacobians.
        // 5. 计算雅可比
        // 左相机归一化坐标点相对于左相机坐标系下的点的雅可比
        // (x, y) = (X / Z, Y / Z)
        Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
        dz_dpc0(0, 0) = 1 / p_c0(2);
        dz_dpc0(1, 1) = 1 / p_c0(2);
        dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
        dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

        // 与上同理
        Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
        dz_dpc1(2, 0) = 1 / p_c1(2);
        dz_dpc1(3, 1) = 1 / p_c1(2);
        dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2) * p_c1(2));
        dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2) * p_c1(2));

        // 左相机坐标系下的三维点相对于左相机位姿的雅可比 先r后t
        Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
        dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
        dpc0_dxc.rightCols(3) = -R_w_c0;

        // 右相机坐标系下的三维点相对于左相机位姿的雅可比 先r后t
        Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
        dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
        dpc1_dxc.rightCols(3) = -R_w_c1;

        // Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);
        // Vector3d p_c1 = R_w_c1 * (p_w - t_c1_w);
        // p_c0 对 p_w
        Matrix3d dpc0_dpg = R_w_c0;
        // p_c1 对 p_w
        Matrix3d dpc1_dpg = R_w_c1;

        // 两个雅可比
        H_x = dz_dpc0 * dpc0_dxc + dz_dpc1 * dpc1_dxc;
        H_f = dz_dpc0 * dpc0_dpg + dz_dpc1 * dpc1_dpg;

        // Modifty the measurement Jacobian to ensure
        // observability constrain.
        // 6. OC
        Matrix<double, 4, 6> A = H_x;
        Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
        u.block<3, 1>(0, 0) =
            quaternionToRotation(cam_state.orientation_null) * IMUState::gravity;
        u.block<3, 1>(3, 0) =
            skewSymmetric(p_w - cam_state.position_null) * IMUState::gravity;
        H_x = A - A * u * (u.transpose() * u).inverse() * u.transpose();
        H_f = -H_x.block<4, 3>(0, 3);

        // Compute the residual.
        // 7. 计算归一化平面坐标误差
        r = z - Vector4d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2),
                         p_c1(0) / p_c1(2), p_c1(1) / p_c1(2));

        return;
    }

    /**
     * @brief 当状态量中的相机状态量过多时，删除一些相机状态量。同时使用待删除的相机帧
     * 中共同观测到的特征点再进行一次计算重投影误差，观测雅可比矩阵和ESKF更新的操作
     * 见笔记pdf中《滑窗删除冗余帧》
     */
    void MsckfVio::pruneCamStateBuffer()
    {
        /// 1. 状态量中相机状态小于20个，直接return
        if (state_server.cam_states.size() < max_cam_state_size)
            return;

        /// 2. 从所有相机状态中挑选出两个该删的相机帧，挑选策略见笔记pdf《找出应该删除的相机帧》
        /// @see MsckfVio::findRedundantCamStates(std::vector<StateIDType> &rm_cam_state_ids)
        vector<StateIDType> rm_cam_state_ids(0);
        findRedundantCamStates(rm_cam_state_ids);

        /// 3. 遍历所有特征点，查找被带删除相机帧共同观测过的特征点(后面的步骤看代码，暂时不写了)
        int jacobian_row_size = 0;
        for (auto &item : map_server)
        {
            auto &feature = item.second;
            // 3.1 查找该特征点中储存的观测中(特征点中储存了哪些帧观测到了该特征点)是否有待删除的帧
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
            // 3.2 如果只有一个待删除的帧观测到了该特征点，那么将该特征点中关于该待删除帧的观测删除(即该特征点不再被该帧观测到，因为该帧删除了)
            if (involved_cam_state_ids.size() == 1)
            {
                feature.observations.erase(involved_cam_state_ids[0]);
                continue;
            }
            // 程序到这里说明involved_cam_state_ids至少大于等于2
            // 说明该特征点记录的相机帧id中至少有两个帧是待删除的帧
            // 3.3 如果该特征点没有做过三角化，做一下三角化，如果失败直接从 该特征点记录的 观测到该特征点的相机帧id 中删除该待删除的帧
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
                    if (!feature.initializePosition(state_server.cam_states))
                    {
                        for (const auto &cam_id : involved_cam_state_ids)
                            feature.observations.erase(cam_id);
                        continue;
                    }
                }
            }

            // 3.4 计算出雅可比与误差的行数
            // 因为删除这些帧不能只删除，其中还包含有用的观测信息，最终还可以做一次更新
            jacobian_row_size += 4 * involved_cam_state_ids.size() - 3;
        }

        // cout << "jacobian row #: " << jacobian_row_size << endl;

        // 计算待删掉的这部分观测的雅可比与误差
        // 预设大小
        MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                      21 + 6 * state_server.cam_states.size());
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

        // 用上述计算的雅可比与误差更新状态
        measurementUpdate(H_x, r);

        // 删除相机状态和协方差矩阵中对应的行列
        for (const auto &cam_id : rm_cam_state_ids)
        {
            // 找到相机状态在状态向量中的位置
            int cam_sequence = std::distance(
                state_server.cam_states.begin(), state_server.cam_states.find(cam_id));
            int cam_state_start = 21 + 6 * cam_sequence;
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
     * 见笔记pdf中《找出应该删除的相机帧》流程图
     * @param  rm_cam_state_ids 要删除的相机状态id
     */
    void MsckfVio::findRedundantCamStates(
        vector<StateIDType> &rm_cam_state_ids)
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
        const Vector3d key_position =
            key_cam_state_iter->second.position;
        const Matrix3d key_rotation = quaternionToRotation(
            key_cam_state_iter->second.orientation);

        // 3. 遍历两次，必然删掉两个状态，有可能是相对新的，有可能是最旧的
        // 但是永远删不到最新的
        for (int i = 0; i < 2; ++i)
        {
            // 从倒数第三个开始，取出位姿
            const Vector3d position =
                cam_state_iter->second.position;
            const Matrix3d rotation = quaternionToRotation(
                cam_state_iter->second.orientation);

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
        // 输入的dof的值是所有相机观测，且没有去掉滑窗的
        // 而且按照维度这个卡方的维度也不对
        //
        MatrixXd P1 = H * state_server.state_cov * H.transpose();
        MatrixXd P2 = Feature::observation_noise *
                      MatrixXd::Identity(H.rows(), H.rows());
        double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

        // cout << dof << " " << gamma << " " <<
        //   chi_squared_test_table[dof] << " ";

        if (gamma < chi_squared_test_table[dof])
        {
            // cout << "passed" << endl;
            return true;
        }
        else
        {
            // cout << "failed" << endl;
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
        double position_x_std = std::sqrt(state_server.state_cov(12, 12));
        double position_y_std = std::sqrt(state_server.state_cov(13, 13));
        double position_z_std = std::sqrt(state_server.state_cov(14, 14));

        if (position_x_std < position_std_threshold &&
            position_y_std < position_std_threshold &&
            position_z_std < position_std_threshold)
            return;

        ROS_WARN("Start %lld online reset procedure...",
                 ++online_reset_counter);
        ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
                 position_x_std, position_y_std, position_z_std);

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

        state_server.state_cov = MatrixXd::Zero(21, 21);
        for (int i = 3; i < 6; ++i)
            state_server.state_cov(i, i) = gyro_bias_cov;
        for (int i = 6; i < 9; ++i)
            state_server.state_cov(i, i) = velocity_cov;
        for (int i = 9; i < 12; ++i)
            state_server.state_cov(i, i) = acc_bias_cov;
        for (int i = 15; i < 18; ++i)
            state_server.state_cov(i, i) = extrinsic_rotation_cov;
        for (int i = 18; i < 21; ++i)
            state_server.state_cov(i, i) = extrinsic_translation_cov;

        ROS_WARN("%lld online reset complete...", online_reset_counter);
        return;
    }

    /**
     * @brief 发布位姿，点云，tf以及协方差
     */
    void MsckfVio::publish(const ros::Time &time)
    {

        // Convert the IMU frame to the body frame.
        // 1. 计算body坐标，因为imu与body相对位姿是单位矩阵，所以就是imu的坐标
        const IMUState &imu_state = state_server.imu_state;
        Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
        T_i_w.linear() = quaternionToRotation(
                             imu_state.orientation)
                             .transpose();
        T_i_w.translation() = imu_state.position;

        Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
                                  IMUState::T_imu_body.inverse();
        Eigen::Vector3d body_velocity =
            IMUState::T_imu_body.linear() * imu_state.velocity;

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
        Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
        Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
        Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
        Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
        P_imu_pose << P_pp, P_po, P_op, P_oo;

        // 转下坐标，但是这里都是单位矩阵
        Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
        H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
        H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
        Matrix<double, 6, 6> P_body_pose = H_pose *
                                           P_imu_pose * H_pose.transpose();

        // 填充协方差
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                odom_msg.pose.covariance[6 * i + j] = P_body_pose(i, j);

        // Construct the covariance for the velocity.
        // 速度协方差
        Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
        Matrix3d H_vel = IMUState::T_imu_body.linear();
        Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                odom_msg.twist.covariance[i * 6 + j] = P_body_vel(i, j);

        // 发布位姿
        odom_pub.publish(odom_msg);

        // Publish the 3D positions of the features that
        // has been initialized.
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
                    IMUState::T_imu_body.linear() * feature.position;
                feature_msg_ptr->points.push_back(pcl::PointXYZ(
                    feature_position(0), feature_position(1), feature_position(2)));
            }
        }
        feature_msg_ptr->width = feature_msg_ptr->points.size();

        feature_pub.publish(feature_msg_ptr);

        return;
    }

} // namespace msckf_vio
