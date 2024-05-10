/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <algorithm>
#include <set>
#include <Eigen/Dense>

#include <sensor_msgs/image_encodings.h>
#include <random_numbers/random_numbers.h>

#include <msckf_vio/CameraMeasurement.h>
#include <msckf_vio/TrackingInfo.h>
#include <msckf_vio/image_processor.h>
#include <msckf_vio/utils.h>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace msckf_vio
{
    /**
     * @brief ImageProcessor构造函数
     * @param[in] n ros节点句柄
     * @param 列表初始化 is_first_img = true, 设置第一帧图像标志位
     */
    ImageProcessor::ImageProcessor(ros::NodeHandle &n)
        : nh(n), is_first_img(true),
          stereo_sub(10), prev_features_ptr(new GridFeatures()),
          curr_features_ptr(new GridFeatures())
    {
        return;
    }

    ImageProcessor::~ImageProcessor()
    {
        destroyAllWindows();
        // ROS_INFO("Feature lifetime statistics:");
        // featureLifetimeStatistics();
        return;
    }

    /**
     * @brief 导入节点launch时提供的各种参数
     * @return 成功或失败 一直为true
     * @see ImageProcessor::initialize()
     */
    bool ImageProcessor::loadParameters()
    {
        // Camera calibration parameters
        // 1. 畸变模型，默认都是用的radtan模型
        nh.param<string>("cam0/distortion_model", cam0_distortion_model, string("radtan"));
        nh.param<string>("cam1/distortion_model", cam1_distortion_model, string("radtan"));

        // 2. 像素分辨率，没用
        vector<int> cam0_resolution_temp(2);
        nh.getParam("cam0/resolution", cam0_resolution_temp);
        cam0_resolution[0] = cam0_resolution_temp[0];
        cam0_resolution[1] = cam0_resolution_temp[1];

        vector<int> cam1_resolution_temp(2);
        nh.getParam("cam1/resolution", cam1_resolution_temp);
        cam1_resolution[0] = cam1_resolution_temp[0];
        cam1_resolution[1] = cam1_resolution_temp[1];

        // 3. 左右目内参
        vector<double> cam0_intrinsics_temp(4);
        nh.getParam("cam0/intrinsics", cam0_intrinsics_temp);
        cam0_intrinsics[0] = cam0_intrinsics_temp[0];
        cam0_intrinsics[1] = cam0_intrinsics_temp[1];
        cam0_intrinsics[2] = cam0_intrinsics_temp[2];
        cam0_intrinsics[3] = cam0_intrinsics_temp[3];

        vector<double> cam1_intrinsics_temp(4);
        nh.getParam("cam1/intrinsics", cam1_intrinsics_temp);
        cam1_intrinsics[0] = cam1_intrinsics_temp[0];
        cam1_intrinsics[1] = cam1_intrinsics_temp[1];
        cam1_intrinsics[2] = cam1_intrinsics_temp[2];
        cam1_intrinsics[3] = cam1_intrinsics_temp[3];

        // 4. 左右目畸变
        vector<double> cam0_distortion_coeffs_temp(4);
        nh.getParam("cam0/distortion_coeffs",
                    cam0_distortion_coeffs_temp);
        cam0_distortion_coeffs[0] = cam0_distortion_coeffs_temp[0];
        cam0_distortion_coeffs[1] = cam0_distortion_coeffs_temp[1];
        cam0_distortion_coeffs[2] = cam0_distortion_coeffs_temp[2];
        cam0_distortion_coeffs[3] = cam0_distortion_coeffs_temp[3];

        vector<double> cam1_distortion_coeffs_temp(4);
        nh.getParam("cam1/distortion_coeffs",
                    cam1_distortion_coeffs_temp);
        cam1_distortion_coeffs[0] = cam1_distortion_coeffs_temp[0];
        cam1_distortion_coeffs[1] = cam1_distortion_coeffs_temp[1];
        cam1_distortion_coeffs[2] = cam1_distortion_coeffs_temp[2];
        cam1_distortion_coeffs[3] = cam1_distortion_coeffs_temp[3];

        // 5. 左右目外参
        // 这里面的T_imu_cam0 表示 imu 到 cam0 的外参
        cv::Mat T_imu_cam0 = utils::getTransformCV(nh, "cam0/T_cam_imu");
        cv::Matx33d R_imu_cam0(T_imu_cam0(cv::Rect(0, 0, 3, 3)));
        cv::Vec3d t_imu_cam0 = T_imu_cam0(cv::Rect(3, 0, 1, 3));
        R_cam0_imu = R_imu_cam0.t();
        t_cam0_imu = -R_imu_cam0.t() * t_imu_cam0;

        cv::Mat T_cam0_cam1 = utils::getTransformCV(nh, "cam1/T_cn_cnm1");
        cv::Mat T_imu_cam1 = T_cam0_cam1 * T_imu_cam0;
        cv::Matx33d R_imu_cam1(T_imu_cam1(cv::Rect(0, 0, 3, 3)));
        cv::Vec3d t_imu_cam1 = T_imu_cam1(cv::Rect(3, 0, 1, 3));
        R_cam1_imu = R_imu_cam1.t();
        t_cam1_imu = -R_imu_cam1.t() * t_imu_cam1;

        // Processor parameters
        // 6.处理图像线程所用的参数
        // 网格数量
        nh.param<int>("grid_row", processor_config.grid_row, 4);
        nh.param<int>("grid_col", processor_config.grid_col, 4);
        // 每一格内最小与最大的特征点数量
        nh.param<int>("grid_min_feature_num",
                      processor_config.grid_min_feature_num, 2);
        nh.param<int>("grid_max_feature_num",
                      processor_config.grid_max_feature_num, 4);

        // 特征提取参数，包括金字塔层数，fast响应值等等
        nh.param<int>("pyramid_levels",
                      processor_config.pyramid_levels, 3);
        nh.param<int>("patch_size",
                      processor_config.patch_size, 31);
        nh.param<int>("fast_threshold",
                      processor_config.fast_threshold, 20);
        nh.param<int>("max_iteration",
                      processor_config.max_iteration, 30);

        // 光流提取所用的参数
        nh.param<double>("track_precision",
                         processor_config.track_precision, 0.01);
        nh.param<double>("ransac_threshold",
                         processor_config.ransac_threshold, 3);
        nh.param<double>("stereo_threshold",
                         processor_config.stereo_threshold, 3);

        // 剩下的都是显示了
        ROS_INFO("===========================================");
        ROS_INFO("cam0_resolution: %d, %d",
                 cam0_resolution[0], cam0_resolution[1]);
        ROS_INFO("cam0_intrinscs: %f, %f, %f, %f",
                 cam0_intrinsics[0], cam0_intrinsics[1],
                 cam0_intrinsics[2], cam0_intrinsics[3]);
        ROS_INFO("cam0_distortion_model: %s",
                 cam0_distortion_model.c_str());
        ROS_INFO("cam0_distortion_coefficients: %f, %f, %f, %f",
                 cam0_distortion_coeffs[0], cam0_distortion_coeffs[1],
                 cam0_distortion_coeffs[2], cam0_distortion_coeffs[3]);

        ROS_INFO("cam1_resolution: %d, %d",
                 cam1_resolution[0], cam1_resolution[1]);
        ROS_INFO("cam1_intrinscs: %f, %f, %f, %f",
                 cam1_intrinsics[0], cam1_intrinsics[1],
                 cam1_intrinsics[2], cam1_intrinsics[3]);
        ROS_INFO("cam1_distortion_model: %s",
                 cam1_distortion_model.c_str());
        ROS_INFO("cam1_distortion_coefficients: %f, %f, %f, %f",
                 cam1_distortion_coeffs[0], cam1_distortion_coeffs[1],
                 cam1_distortion_coeffs[2], cam1_distortion_coeffs[3]);

        cout << R_imu_cam0 << endl;
        cout << t_imu_cam0.t() << endl;

        ROS_INFO("grid_row: %d",
                 processor_config.grid_row);
        ROS_INFO("grid_col: %d",
                 processor_config.grid_col);
        ROS_INFO("grid_min_feature_num: %d",
                 processor_config.grid_min_feature_num);
        ROS_INFO("grid_max_feature_num: %d",
                 processor_config.grid_max_feature_num);
        ROS_INFO("pyramid_levels: %d",
                 processor_config.pyramid_levels);
        ROS_INFO("patch_size: %d",
                 processor_config.patch_size);
        ROS_INFO("fast_threshold: %d",
                 processor_config.fast_threshold);
        ROS_INFO("max_iteration: %d",
                 processor_config.max_iteration);
        ROS_INFO("track_precision: %f",
                 processor_config.track_precision);
        ROS_INFO("ransac_threshold: %f",
                 processor_config.ransac_threshold);
        ROS_INFO("stereo_threshold: %f",
                 processor_config.stereo_threshold);
        ROS_INFO("===========================================");
        return true;
    }

    /**
     * @brief 创建前端ROS节点的发布和订阅话题
     */
    bool ImageProcessor::createRosIO()
    {
        /// 1. "features"话题，缓存长度为3，用于向后端发布特征点（header,点id,左右目归一化坐标）
        feature_pub = nh.advertise<CameraMeasurement>("features", 3);
        /// 2. "tracking_info"话题，缓存长度为1，用于发布跟踪信息，没有地方接收
        tracking_info_pub = nh.advertise<TrackingInfo>("tracking_info", 1);
        /// 3. "debug_stereo_image"话题，缓存长度为1，用于发布绘制的双目图像
        image_transport::ImageTransport it(nh);
        debug_stereo_pub = it.advertise("debug_stereo_image", 1);

        /// 4. "cam0_image"话题，缓存长度为10，用于订阅左目图像
        cam0_img_sub.subscribe(nh, "cam0_image", 10);
        /// 5. "cam1_image"话题，缓存长度为10，用于订阅右目图像
        cam1_img_sub.subscribe(nh, "cam1_image", 10);
        /// 6. ROS软件同步订阅双目图像消息，缓存长度为10，回调函数为ImageProcessor::stereoCallback
        stereo_sub.connectInput(cam0_img_sub, cam1_img_sub);
        stereo_sub.registerCallback(&ImageProcessor::stereoCallback, this);
        /// 7. "imu"话题，缓存长度为50，用于订阅IMU消息
        imu_sub = nh.subscribe("imu", 50, &ImageProcessor::imuCallback, this);

        return true;
    }

    /**
     * @brief 初始化
     * @return 成功或失败
     */
    bool ImageProcessor::initialize()
    {
        /// 1. 加载参数 @see ImageProcessor::loadParameters()
        if (!loadParameters())
            return false;
        ROS_INFO("Finish loading ROS parameters...");

        /// 2. 构造OpenCV的FAST特征提取器
        detector_ptr = FastFeatureDetector::create(processor_config.fast_threshold);

        /// 3. 构造ROS IO进行消息的订阅和发布 @see ImageProcessor::createRosIO() 
        if (!createRosIO())
            return false;
        ROS_INFO("Finish creating ROS IO...");

        return true;
    }

    /**
     * @brief 双目图像回调, 处理双目图像，前端主要流程函数
     * @param  cam0_img 左图消息
     * @param  cam1_img 右图消息
     */
    void ImageProcessor::stereoCallback(const sensor_msgs::ImageConstPtr &cam0_img, const sensor_msgs::ImageConstPtr &cam1_img)
    {

        // 将ros的image消息转换为opencv中的cv::Mat（在cami_curr_img_ptr的成员image中存储），其中cv::Mat为mono8格式
        cam0_curr_img_ptr = cv_bridge::toCvShare(cam0_img, sensor_msgs::image_encodings::MONO8);
        cam1_curr_img_ptr = cv_bridge::toCvShare(cam1_img, sensor_msgs::image_encodings::MONO8);

        /// 1. 创建图像金字塔， @see ImageProcessor::createImagePyramids()
        createImagePyramids();

        /// 2. 检测是否是第一帧图像, 如果是第一帧图像，初始化第一帧特征点并绘制发布双目图像
        /// @see ImageProcessor::initializeFirstFrame(), ImageProcessor::drawFeaturesStereo()
        if (is_first_img)
        {
            // ros::Time start_time = ros::Time::now();
            // 3.1 初始化第一批特征：
            // 利用了LKT光流法来寻找两帧间的匹配点
            // 利用了对极约束筛选外点
            // 将特征点划分到不同的图像grid中
            // 同时有数量限制
            initializeFirstFrame();
            // ROS_INFO("Detection time: %f",
            //     (ros::Time::now()-start_time).toSec());
            is_first_img = false;

            // Draw results.
            // start_time = ros::Time::now();
            // 3.2 当有其他节点订阅了debug_stereo_image消息时
            // 将双目图像拼接起来并画出特征点位置，作为消息发出去
            drawFeaturesStereo();
            // ROS_INFO("Draw features: %f",
            //     (ros::Time::now()-start_time).toSec());
        }
        else
        {
            /// 3. 如果不是第一帧图像，进行下述操作
            ///   -# 跟踪特征点 @see ImageProcessor::trackFeatures()
            ///   -# 在左目提取新的特征，通过左右目光流跟踪去外点，向变量添加新的特征 @see ImageProcessor::addNewFeatures()
            ///   -# 剔除每个格多余的点 @see ImageProcessor::pruneGridFeatures()
            ///   -# 当有其他节点订阅了debug_stereo_image消息时，将双目图像拼接起来并画出特征点位置，作为消息发送出去
            ///      @see ImageProcessor::drawFeaturesStereo()

            // ros::Time start_time = ros::Time::now();
            trackFeatures();
            // ROS_INFO("Tracking time: %f",
            //     (ros::Time::now()-start_time).toSec());

            // start_time = ros::Time::now();
            addNewFeatures();
            // ROS_INFO("Addition time: %f",
            //     (ros::Time::now()-start_time).toSec());

            // start_time = ros::Time::now();
            pruneGridFeatures();
            // ROS_INFO("Prune grid features: %f",
            //     (ros::Time::now()-start_time).toSec());

            // start_time = ros::Time::now();
            drawFeaturesStereo();
            // ROS_INFO("Draw features: %f",
            //     (ros::Time::now()-start_time).toSec());
        }

        // ros::Time start_time = ros::Time::now();
        // updateFeatureLifetime();
        // ROS_INFO("Statistics: %f",
        //     (ros::Time::now()-start_time).toSec());

        /// 5. 发布图片特征点跟踪的结果到后端 @see ImageProcessor::publish()
        // ros::Time start_time = ros::Time::now();
        publish();
        // ROS_INFO("Publishing: %f",
        //     (ros::Time::now()-start_time).toSec());

        // Update the previous image and previous features.
        // 保存的都是上一帧左目的，只有curr_features_ptr包含右目的点
        cam0_prev_img_ptr = cam0_curr_img_ptr;
        prev_features_ptr = curr_features_ptr;
        std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

        // Initialize the current features to empty vectors.
        // curr_features_ptr指向new GridFeatures()，分配网格
        // 每次都是新的
        curr_features_ptr.reset(new GridFeatures());
        for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
        {
            (*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
        }

        return;
    }

    /**
     * @brief 接受IMU数据并向imu_msg_buffer添加imu消息
     * @param msg imu消息
     */
    void ImageProcessor::imuCallback(const sensor_msgs::ImuConstPtr &msg)
    {
        // Wait for the first image to be set.
        if (is_first_img)
            return;
        imu_msg_buffer.push_back(*msg);
        return;
    }

    /**
     * @brief 左右目图像构建光流金字塔并保存在curr_cam0/1_pyramid_中，方便后续传递给calcOpticalFlowPyrLK进行光流跟踪
     * @see ImageProcessor::stereoCallback
     */
    void ImageProcessor::createImagePyramids()
    {
        const Mat &curr_cam0_img = cam0_curr_img_ptr->image;
        // 构建图像金字塔，方便后续传递给calcOpticalFlowPyrLK进行光流跟踪
        // @param img 8-bit 输入图像
        // @param pyramid 输出的金字塔
        // @param winSize 每一层金字塔的窗口大小，必须不少于calcOpticalFlowPyrLK的winSize参数
        // @param maxLevel 金字塔的层数
        // @param withDerivatives 是否预先计算每层金字塔的图像梯度，如果为False，那么calcOpticalFlowPyrLK将在内部对其进行计算
        // @param pyrBorder 金字塔图层的边界模式
        // @param derivBorder 梯度边界模式
        // @param tryReuseInputImage 如果可能，将输入图像的ROI放入金字塔中。您可以传递false来强制数据复制
        // @return 构建的金字塔中的级数。可以少于maxLevel
        // 边界模式选择：
        // 1)BORDER_REPLICATE:重复：                    aaaaaa|abcdefgh|hhhhhhh
        // 2)BORDER_REFLECT:反射:                       fedcba|abcdefgh|hgfedcb
        // 3)BORDER_REFLECT_101:反射101:                gfedcb|abcdefgh|gfedcba
        // 4)BORDER_WRAP:外包装：                       cdefgh|abcdefgh|abcdefg
        // 5)BORDER_CONSTANT:常量复制：                 iiiiii|abcdefgh|iiiiiii(i的值由后一个参数Scalar()确定，如Scalar::all(0) )
        // borderValue:若上一参数为BORDER_CONSTANT，则由此参数确定补充上去的像素值。可选用默认值。
        buildOpticalFlowPyramid(
            curr_cam0_img, curr_cam0_pyramid_,
            Size(processor_config.patch_size, processor_config.patch_size),
            processor_config.pyramid_levels, true, BORDER_REFLECT_101,
            BORDER_CONSTANT, false);

        const Mat &curr_cam1_img = cam1_curr_img_ptr->image;
        buildOpticalFlowPyramid(
            curr_cam1_img, curr_cam1_pyramid_,
            Size(processor_config.patch_size, processor_config.patch_size),
            processor_config.pyramid_levels, true, BORDER_REFLECT_101,
            BORDER_CONSTANT, false);
    }

    /**
     * @brief
     * 1. 从cam0的图像中提取FAST特征
     * 2. 利用cam0到cam1的外参, 将cam0中的特征点投影到cam1中, 作为光流法寻找cam1中匹配点的初始值
     * 3. 利用LKT光流法在cam1的图像中寻找匹配的像素点
     * 4. 利用双目外参构成的对极几何约束进行野点筛选。
     * 5. 然后根据cam0中所有匹配特征点的位置将它们分配到不同的grid中
     * 6. 按提取FAST特征时的response对每个grid中的特征进行排序
     * 7. 最后将它们存储到相应的类成员变量中（每个grid特征数有限制）。
     * 执行完后能在curr_features_ptr中找到第一帧图像中提取的左右目特征点
     * @see ImageProcessor::stereoCallback
     */
    void ImageProcessor::initializeFirstFrame()
    {
        // Size of each grid.
        // 1. 计算每一个格子的长宽
        const Mat &img = cam0_curr_img_ptr->image;
        static int grid_height = img.rows / processor_config.grid_row;
        static int grid_width = img.cols / processor_config.grid_col;

        // Detect new features on the frist image.
        // 2. 计算原图fast特征点
        vector<KeyPoint> new_features(0);
        detector_ptr->detect(img, new_features);

        // Find the stereo matched points for the newly detected features.
        // 格式转为Point2f
        vector<cv::Point2f> cam0_points(new_features.size());
        for (int i = 0; i < new_features.size(); ++i)
            cam0_points[i] = new_features[i].pt;

        // 3. 利用LKT光流法在cam1中找到和cam0中特征匹配的点，然后再利用双目相机约束筛选内点
        vector<cv::Point2f> cam1_points(0);
        vector<unsigned char> inlier_markers(0);
        stereoMatch(cam0_points, cam1_points, inlier_markers);

        vector<cv::Point2f> cam0_inliers(0);
        vector<cv::Point2f> cam1_inliers(0);
        vector<float> response_inliers(0);
        // 取出匹配的内点，分别存放至对应的变量中，并且对应位置是匹配的
        for (int i = 0; i < inlier_markers.size(); ++i)
        {
            if (inlier_markers[i] == 0)
                continue;
            cam0_inliers.push_back(cam0_points[i]);
            cam1_inliers.push_back(cam1_points[i]);
            // 响应程度，代表该点强壮大小，即该点是特征点的程度，用白话说就是值越大这个点看起来更像是角点
            response_inliers.push_back(new_features[i].response);
        }

        // Group the features into grids
        // 4. 分格存入

        // 预备好格子
        GridFeatures grid_new_features; // typedef std::map<int, std::vector<FeatureMetaData>>
        for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
            grid_new_features[code] = vector<FeatureMetaData>(0);

        // 遍历所有匹配的内点，根据像素坐标存放到对应的网格中
        for (int i = 0; i < cam0_inliers.size(); ++i)
        {
            const cv::Point2f &cam0_point = cam0_inliers[i];
            const cv::Point2f &cam1_point = cam1_inliers[i];
            const float &response = response_inliers[i];

            int row = static_cast<int>(cam0_point.y / grid_height);
            int col = static_cast<int>(cam0_point.x / grid_width);

            // 在第几个格
            int code = row * processor_config.grid_col + col;

            // 存放，双目点一起存放
            FeatureMetaData new_feature;
            new_feature.response = response;
            new_feature.cam0_point = cam0_point;
            new_feature.cam1_point = cam1_point;
            grid_new_features[code].push_back(new_feature);
        }

        // Sort the new features in each grid based on its response.
        // 5. 根据左目提取的点的响应值，每一格按照响应值的从大到小排序
        for (auto &item : grid_new_features)
            std::sort(item.second.begin(), item.second.end(), &ImageProcessor::featureCompareByResponse);

        // Collect new features within each grid with high response.
        // 6. 向已有的curr_features_ptr里面放入新的响应值高的特征点，同时也不是全部加入，有数量限制
        for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
        {
            // 获取引用，即使curr_features_ptr里面没有这个值也能获取，相当于直接创建
            vector<FeatureMetaData> &features_this_grid = (*curr_features_ptr)[code];
            vector<FeatureMetaData> &new_features_this_grid = grid_new_features[code];

            // 向 features_this_grid 放数据， 可以理解成 grid_new_features 是一个临时的，没有数量限制的
            // 这步就是根据相应排好之后按照数量存放
            for (int k = 0; k < processor_config.grid_min_feature_num && k < new_features_this_grid.size(); ++k)
            {
                features_this_grid.push_back(new_features_this_grid[k]);
                features_this_grid.back().id = next_feature_id++;
                features_this_grid.back().lifetime = 1;
            }
        }

        return;
    }

    /**
     * @brief 利用输入的前一帧特征点图像坐标、前一帧到当前帧的旋转矩阵以及相机内参，预测当前帧中的特征点图像坐标。
     * 作用是给LKT光流一个initial guess。
     * @param  input_pts 上一帧的像素点
     * @param  R_p_c 旋转，左相机的上一帧到当前帧
     * @param  intrinsics 内参
     * @param  compensated_pts 输出预测的点
     * @see ImageProcessor::trackFeatures()
     */
    void ImageProcessor::predictFeatureTracking(
        const vector<cv::Point2f> &input_pts, const cv::Matx33f &R_p_c,
        const cv::Vec4d &intrinsics, vector<cv::Point2f> &compensated_pts)
    {
        // 1. 如果输入的上一帧像素点为0，清空返回
        if (input_pts.size() == 0)
        {
            compensated_pts.clear();
            return;
        }
        compensated_pts.resize(input_pts.size());

        // 内参矩阵
        cv::Matx33f K(
            intrinsics[0], 0.0, intrinsics[2],
            0.0, intrinsics[1], intrinsics[3],
            0.0, 0.0, 1.0);

        // 2. 投到归一化坐标，旋转，再转回像素坐标
        // 没有去畸变，反正只是帮助预测，又不是定死了的位置，节省计算资源
        cv::Matx33f H = K * R_p_c * K.inv();
        for (int i = 0; i < input_pts.size(); ++i)
        {
            cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
            cv::Vec3f p2 = H * p1;
            compensated_pts[i].x = p2[0] / p2[2];
            compensated_pts[i].y = p2[1] / p2[2];
        }

        return;
    }

    /**
     * @brief 跟踪特征点
     * @details 1. 上一帧左目特征点 --(光流跟踪)-> 当前帧左目特征点
     * 2. 当前帧左目特征点 --(双目匹配)-> 当前帧右目特征点
     * 3. 当前帧左目特征点 & 前一帧左目特征点 --(RANSAC去除外点)
     * 4. 当前帧右目特征点 & 前一帧右目特征点 --(RANSAC去除外点)
     * 5. 保存当前帧左右目特征点到curr_features_ptr
     */
    void ImageProcessor::trackFeatures()
    {
        // 网格大小，在很下面才会用到
        static int grid_height = cam0_curr_img_ptr->image.rows / processor_config.grid_row;
        static int grid_width = cam0_curr_img_ptr->image.cols / processor_config.grid_col;

        // 1. 利用imu数据求前一帧(prev)到当前帧(curr)的旋转矩阵，左右分别计算
        Matx33f cam0_R_p_c;
        Matx33f cam1_R_p_c;
        integrateImuData(cam0_R_p_c, cam1_R_p_c);

        // 2. 把上一帧所有点的相关信息全取出来
        vector<FeatureIDType> prev_ids(0);
        vector<int> prev_lifetime(0);
        vector<Point2f> prev_cam0_points(0);
        vector<Point2f> prev_cam1_points(0);
        for (const auto &item : *prev_features_ptr)
        {
            for (const auto &prev_feature : item.second)
            {
                prev_ids.push_back(prev_feature.id);
                prev_lifetime.push_back(prev_feature.lifetime);
                prev_cam0_points.push_back(prev_feature.cam0_point);
                prev_cam1_points.push_back(prev_feature.cam1_point);
            }
        }

        // 跟踪前上一帧的点数
        before_tracking = prev_cam0_points.size();

        // 上一帧没有点，return
        if (prev_ids.size() == 0)
            return;

        // 3. 利用输入的前一帧特征点图像坐标、前一帧到当前帧的旋转矩阵以及相机内参，预测当前帧中的特征点图像坐标(只有左相机)
        vector<Point2f> curr_cam0_points(0);
        vector<unsigned char> track_inliers(0);
        predictFeatureTracking(prev_cam0_points, cam0_R_p_c, cam0_intrinsics, curr_cam0_points);

        // 4. 光流跟踪，都是在未去畸变的点上做的
        calcOpticalFlowPyrLK(
            prev_cam0_pyramid_, curr_cam0_pyramid_, // 左相机上一帧与当前帧的图片（多层金字塔）
            prev_cam0_points, curr_cam0_points,     // 前后特征点, 当前时刻的点可以通过上一时刻的点和IMU数据预测出来作为预输入
            track_inliers, noArray(),
            Size(processor_config.patch_size, processor_config.patch_size),
            processor_config.pyramid_levels,
            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, // 迭代终止条件，当达到最大次数或指定精度时停止
                         processor_config.max_iteration,
                         processor_config.track_precision),
            cv::OPTFLOW_USE_INITIAL_FLOW); // OPTFLOW_USE_INITIAL_FLOW使用存储在curr_cam0_points中的初始估计值

        // 去除边缘以外的点
        for (int i = 0; i < curr_cam0_points.size(); ++i)
        {
            if (track_inliers[i] == 0)
                continue;
            if (curr_cam0_points[i].y < 0 ||
                curr_cam0_points[i].y > cam0_curr_img_ptr->image.rows - 1 ||
                curr_cam0_points[i].x < 0 ||
                curr_cam0_points[i].x > cam0_curr_img_ptr->image.cols - 1)
                track_inliers[i] = 0;
        }

        // 收集成功跟踪到的点的信息
        vector<FeatureIDType> prev_tracked_ids(0);
        vector<int> prev_tracked_lifetime(0);
        vector<Point2f> prev_tracked_cam0_points(0);
        vector<Point2f> prev_tracked_cam1_points(0);
        vector<Point2f> curr_tracked_cam0_points(0);
        // 5. 根据track_inliers删除外点，且剩下的依然是一一对应的点
        removeUnmarkedElements(prev_ids, track_inliers, prev_tracked_ids);
        removeUnmarkedElements(prev_lifetime, track_inliers, prev_tracked_lifetime);
        removeUnmarkedElements(prev_cam0_points, track_inliers, prev_tracked_cam0_points);
        removeUnmarkedElements(prev_cam1_points, track_inliers, prev_tracked_cam1_points);
        removeUnmarkedElements(curr_cam0_points, track_inliers, curr_tracked_cam0_points);

        // 跟踪后当前帧点的数量
        after_tracking = curr_tracked_cam0_points.size();

        // Outlier removal involves three steps, which forms a close
        // loop between the previous and current frames of cam0 (left)
        // and cam1 (right). Assuming the stereo matching between the
        // previous cam0 and cam1 images are correct, the three steps are:
        //
        // prev frames cam0 ----------> cam1
        //              |                |
        //              |ransac          |ransac
        //              |   stereo match |
        // curr frames cam0 ----------> cam1
        //
        // 1) Stereo matching between current images of cam0 and cam1.
        // 2) RANSAC between previous and current images of cam0.
        // 3) RANSAC between previous and current images of cam1.
        //
        // For Step 3, tracking between the images is no longer needed.
        // The stereo matching results are directly used in the RANSAC.

        // Step 1: stereo matching.
        // 6. 进一步去除外点
        // 6.1 将上一步跟踪到当前帧的点通过左右目匹配去除外点
        vector<Point2f> curr_cam1_points(0);
        vector<unsigned char> match_inliers(0);
        // 利用LKT光流法和双目外参约束，在当前cam1上找到和当前cam0上匹配的特征点
        stereoMatch(curr_tracked_cam0_points, curr_cam1_points, match_inliers);

        vector<FeatureIDType> prev_matched_ids(0);
        vector<int> prev_matched_lifetime(0);
        vector<Point2f> prev_matched_cam0_points(0);
        vector<Point2f> prev_matched_cam1_points(0);
        vector<Point2f> curr_matched_cam0_points(0);
        vector<Point2f> curr_matched_cam1_points(0);

        // 左右光流删除外点
        removeUnmarkedElements(prev_tracked_ids, match_inliers, prev_matched_ids);
        removeUnmarkedElements(prev_tracked_lifetime, match_inliers, prev_matched_lifetime);
        removeUnmarkedElements(prev_tracked_cam0_points, match_inliers, prev_matched_cam0_points);
        removeUnmarkedElements(prev_tracked_cam1_points, match_inliers, prev_matched_cam1_points);
        removeUnmarkedElements(curr_tracked_cam0_points, match_inliers, curr_matched_cam0_points);
        removeUnmarkedElements(curr_cam1_points, match_inliers, curr_matched_cam1_points);

        // 经过双目除外点后
        after_matching = curr_matched_cam0_points.size();

        // Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
        // 6.2 进一步通过ransac在左目前后帧和右目前后帧中挑出内外点，并把标志存入cam0/cam1_ransac_inliers
        // prev_matched_cam0_points curr_matched_cam0_points curr_matched_cam1_points prev_matched_cam1_points内的点一一对应
        vector<int> cam0_ransac_inliers(0);
        twoPointRansac(prev_matched_cam0_points, curr_matched_cam0_points,
                       cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
                       cam0_distortion_coeffs, processor_config.ransac_threshold,
                       0.99, cam0_ransac_inliers);

        vector<int> cam1_ransac_inliers(0);
        twoPointRansac(prev_matched_cam1_points, curr_matched_cam1_points,
                       cam1_R_p_c, cam1_intrinsics, cam1_distortion_model,
                       cam1_distortion_coeffs, processor_config.ransac_threshold,
                       0.99, cam1_ransac_inliers);

        // 7. 将curr_matched_cam0_points和curr_matched_cam1_points中不是外点的点存入curr_features_ptr
        after_ransac = 0;
        for (int i = 0; i < cam0_ransac_inliers.size(); ++i)
        {
            if (cam0_ransac_inliers[i] == 0 || cam1_ransac_inliers[i] == 0)
                continue;
            // 确定当前帧的内点根据坐标确定在哪个格后分格存入，供下一帧使用
            int row = static_cast<int>(curr_matched_cam0_points[i].y / grid_height);
            int col = static_cast<int>(curr_matched_cam0_points[i].x / grid_width);
            int code = row * processor_config.grid_col + col;
            (*curr_features_ptr)[code].push_back(FeatureMetaData());

            FeatureMetaData &grid_new_feature = (*curr_features_ptr)[code].back();
            grid_new_feature.id = prev_matched_ids[i];
            grid_new_feature.lifetime = ++prev_matched_lifetime[i];
            grid_new_feature.cam0_point = curr_matched_cam0_points[i];
            grid_new_feature.cam1_point = curr_matched_cam1_points[i];

            ++after_ransac;
        }

        // 8. 每0.5s打印一次数据，包括跟踪前点数，跟踪后点数，双目匹配后点数，以及算法处理后当前帧与上一帧的点数和比例
        int prev_feature_num = 0;
        for (const auto &item : *prev_features_ptr)
            prev_feature_num += item.second.size();

        int curr_feature_num = 0;
        for (const auto &item : *curr_features_ptr)
            curr_feature_num += item.second.size();

        ROS_INFO_THROTTLE(
            0.5, "\033[0;32m candidates: %d; track: %d; match: %d; ransac: %d/%d=%f\033[0m",
            before_tracking, after_tracking, after_matching,
            curr_feature_num, prev_feature_num,
            static_cast<double>(curr_feature_num) /
                (static_cast<double>(prev_feature_num) + 1e-5));
        // printf(
        //     "\033[0;32m candidates: %d; raw track: %d; stereo match: %d; ransac: %d/%d=%f\033[0m\n",
        //     before_tracking, after_tracking, after_matching,
        //     curr_feature_num, prev_feature_num,
        //     static_cast<double>(curr_feature_num)/
        //     (static_cast<double>(prev_feature_num)+1e-5));

        return;
    }

    /**
     * @brief 在右相机cam1中找到与左相机cam0中特征点匹配的点
     *        1. 先利用双目的已知外参将左相机的点投影到右相机中，作为右相机中特征点的初值
     *        2. 利用LKT光流法在右相机中找到和左相机中特征点匹配的点
     *        3. 利用双目外参构成的对极几何约束进行野点筛选
     * @note  筛选的点是在inlier_markers中标记, 而不是将其在cam1_points中删除
     * @param  cam0_points 左图像的特征点，输入
     * @param  cam1_points 与cam0_points匹配的点
     * @param  inlier_markers 是否是内点 1是 0否
     * @see ImageProcessor::addNewFeatures()  ImageProcessor::trackFeatures()  ImageProcessor::initializeFirstFrame()
     */
    void ImageProcessor::stereoMatch(
        const vector<cv::Point2f> &cam0_points, vector<cv::Point2f> &cam1_points, vector<unsigned char> &inlier_markers)
    {

        if (cam0_points.size() == 0)
            return;

        // 刚进来cam1_points里面必然是空的，得到cam1_points的初值
        if (cam1_points.size() == 0)
        {
            // 1. 计算左相机到右相机的旋转，你没看错是左到右，MSCKF中都是这样定义的
            const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
            vector<cv::Point2f> cam0_points_undistorted;
            // 2. 左相机中带畸变的像素点 --(左右目外参)-> 右相机中无畸变的归一化坐标，相当于先进行一个预测，为后续的光流跟踪提供一个初值
            undistortPoints(cam0_points, cam0_intrinsics, cam0_distortion_model, cam0_distortion_coeffs, cam0_points_undistorted, R_cam0_cam1);

            // 右相机中无畸变的归一化坐标 -> 右相机中带畸变的像素点
            cam1_points = distortPoints(cam0_points_undistorted, cam1_intrinsics, cam1_distortion_model, cam1_distortion_coeffs);
        }

        // 3. 光流匹配找点，中间去除一些误匹配
        calcOpticalFlowPyrLK(
            curr_cam0_pyramid_, curr_cam1_pyramid_,                         // 左相机与右相机当前帧的图片（多层金字塔）
            cam0_points, cam1_points,                                       // 左右特征点，一一对应的，当OPTFLOW_USE_INITIAL_FLOW参数传入时，cam1_points传入的值作为cam1中特征位置初值，但它最终会用来存放cam1特征点输出值
            inlier_markers, noArray(),                                      // 内点， 每个点跟踪的误差
            Size(processor_config.patch_size, processor_config.patch_size), // 搜索范围
            processor_config.pyramid_levels,                                // 金字塔层数
            TermCriteria(
                TermCriteria::COUNT + TermCriteria::EPS,
                processor_config.max_iteration,    // 用来限制每层金字塔上LKT光流跟踪的迭代次数，即畸变光流未收敛，当到达迭代次数时也停止迭代。
                processor_config.track_precision), // track_precision用来限制每层金字塔LKT光流跟踪的迭代终止阈值，即当某次迭代光流小于该阈值时则停止迭代。
            cv::OPTFLOW_USE_INITIAL_FLOW);         // 使用初值

        // 去除落在图像像素坐标范围外的点
        for (int i = 0; i < cam1_points.size(); ++i)
        {
            if (inlier_markers[i] == 0) // 外点
                continue;
            if (cam1_points[i].y < 0 ||
                cam1_points[i].y > cam1_curr_img_ptr->image.rows - 1 ||
                cam1_points[i].x < 0 ||
                cam1_points[i].x > cam1_curr_img_ptr->image.cols - 1)
                inlier_markers[i] = 0; // 在图像外，设置为外点
        }

        // 相机0->相机1的旋转与平移(外参已知量)
        const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
        const cv::Vec3d t_cam0_cam1 = R_cam1_imu.t() * (t_cam0_imu - t_cam1_imu);
        // 4. 计算本质矩阵  [t]x * R， 根据本质矩阵的对极约束进一步去除外点
        const cv::Matx33d t_cam0_cam1_hat(
            0.0, -t_cam0_cam1[2], t_cam0_cam1[1],
            t_cam0_cam1[2], 0.0, -t_cam0_cam1[0],
            -t_cam0_cam1[1], t_cam0_cam1[0], 0.0);
        const cv::Matx33d E = t_cam0_cam1_hat * R_cam0_cam1;

        vector<cv::Point2f> cam0_points_undistorted(0);
        vector<cv::Point2f> cam1_points_undistorted(0);
        // 5. 计算左右目畸变点对应的归一化平面上的去畸变坐标
        undistortPoints(
            cam0_points, cam0_intrinsics, cam0_distortion_model,
            cam0_distortion_coeffs, cam0_points_undistorted);
        undistortPoints(
            cam1_points, cam1_intrinsics, cam1_distortion_model,
            cam1_distortion_coeffs, cam1_points_undistorted);
        // 计算一个1/f 的平均值
        double norm_pixel_unit = 4.0 / (cam0_intrinsics[0] + cam0_intrinsics[1] + cam1_intrinsics[0] + cam1_intrinsics[1]);

        // 6. 根据本质矩阵的对极约束进一步去除外点，十四讲eq.7-11
        for (int i = 0; i < cam0_points_undistorted.size(); ++i)
        {
            if (inlier_markers[i] == 0)
                continue;
            // 归一化坐标
            cv::Vec3d pt0(cam0_points_undistorted[i].x, cam0_points_undistorted[i].y, 1.0);
            cv::Vec3d pt1(cam1_points_undistorted[i].x, cam1_points_undistorted[i].y, 1.0);

            // 组成线
            cv::Vec3d epipolar_line = E * pt0;

            // 点到线的距离
            double error =
                fabs((pt1.t() * epipolar_line)[0]) /
                sqrt(epipolar_line[0] * epipolar_line[0] + epipolar_line[1] * epipolar_line[1]);
            if (error > processor_config.stereo_threshold * norm_pixel_unit) // 阈值 * 焦距的倒数的平均值
                inlier_markers[i] = 0;
        }

        return;
    }

    /**
     * @brief 在左目提取新的特征，通过左右目光流跟踪去外点，向变量添加新的特征
     * @details 1. 在左目图像上提取新的Fast特征点
     * 2. 将新提取的特征点按照网格分组, 保留响应值高的特征点
     * 3. 将新提取的特征点与右目图像上的特征点进行匹配(同initializeFirstFrame)
     * 4. 将特征点进行网格分组存入curr_features_ptr
     */
    void ImageProcessor::addNewFeatures()
    {
        // 取出当前左目图片
        const Mat &curr_img = cam0_curr_img_ptr->image;

        // Size of each grid.
        static int grid_height = cam0_curr_img_ptr->image.rows / processor_config.grid_row;
        static int grid_width = cam0_curr_img_ptr->image.cols / processor_config.grid_col;

        // Create a mask to avoid redetecting existing features.
        Mat mask(curr_img.rows, curr_img.cols, CV_8U, Scalar(1));
        // 1. 已经有特征的区域在mask上填充0，一定范围内，避免重复提取遍历所有跟踪成功的点
        for (const auto &features : *curr_features_ptr)
        {
            for (const auto &feature : features.second)
            {
                const int y = static_cast<int>(feature.cam0_point.y);
                const int x = static_cast<int>(feature.cam0_point.x);

                // 划片，就是这个点附近都不能再有点了
                int up_lim = y - 2, bottom_lim = y + 3,
                    left_lim = x - 2, right_lim = x + 3;
                if (up_lim < 0)
                    up_lim = 0;
                if (bottom_lim > curr_img.rows)
                    bottom_lim = curr_img.rows;
                if (left_lim < 0)
                    left_lim = 0;
                if (right_lim > curr_img.cols)
                    right_lim = curr_img.cols;

                Range row_range(up_lim, bottom_lim);
                Range col_range(left_lim, right_lim);
                mask(row_range, col_range) = 0;
            }
        }

        // Detect new features.
        vector<KeyPoint> new_features(0);
        // 2. 加入mask提取新的fast特征点
        detector_ptr->detect(curr_img, new_features, mask);

        // Collect the new detected features based on the grid.
        // Select the ones with top response within each grid afterwards.
        // 3. 新提取的特征点按照网格放入new_feature_sieve
        vector<vector<KeyPoint>> new_feature_sieve(processor_config.grid_row * processor_config.grid_col);
        for (const auto &feature : new_features)
        {
            int row = static_cast<int>(feature.pt.y / grid_height);
            int col = static_cast<int>(feature.pt.x / grid_width);
            new_feature_sieve[row * processor_config.grid_col + col].push_back(feature);
        }

        // 3.1 根据响应值排序并限制每个格子中点的数量，放入new_feature中
        new_features.clear();
        for (auto &item : new_feature_sieve)
        {
            // 删除超过数量的点
            if (item.size() > processor_config.grid_max_feature_num)
            {
                std::sort(item.begin(), item.end(), &ImageProcessor::keyPointCompareByResponse);
                item.erase(item.begin() + processor_config.grid_max_feature_num, item.end());
            }
            new_features.insert(new_features.end(), item.begin(), item.end());
        }

        int detected_new_features = new_features.size();
        // 4. 左右目匹配，步骤跟初始化基本一致 转成cv::Point2f
        vector<cv::Point2f> cam0_points(new_features.size());
        for (int i = 0; i < new_features.size(); ++i)
            cam0_points[i] = new_features[i].pt;

        // 4.1 左右目匹配
        vector<cv::Point2f> cam1_points(0);
        vector<unsigned char> inlier_markers(0);
        stereoMatch(cam0_points, cam1_points, inlier_markers);

        // 4.2 保留左右目的内点和响应值
        vector<cv::Point2f> cam0_inliers(0);
        vector<cv::Point2f> cam1_inliers(0);
        vector<float> response_inliers(0);
        for (int i = 0; i < inlier_markers.size(); ++i)
        {
            if (inlier_markers[i] == 0)
                continue;
            cam0_inliers.push_back(cam0_points[i]);
            cam1_inliers.push_back(cam1_points[i]);
            response_inliers.push_back(new_features[i].response);
        }

        int matched_new_features = cam0_inliers.size();

        if (matched_new_features < 5 && static_cast<double>(matched_new_features) / static_cast<double>(detected_new_features) < 0.1)
            ROS_WARN("Images at [%f] seems unsynced...", cam0_curr_img_ptr->header.stamp.toSec());

        // Group the features into grids
        // 5. 分格存入
        GridFeatures grid_new_features;
        for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
            grid_new_features[code] = vector<FeatureMetaData>(0);

        for (int i = 0; i < cam0_inliers.size(); ++i)
        {
            const cv::Point2f &cam0_point = cam0_inliers[i];
            const cv::Point2f &cam1_point = cam1_inliers[i];
            const float &response = response_inliers[i];

            int row = static_cast<int>(cam0_point.y / grid_height);
            int col = static_cast<int>(cam0_point.x / grid_width);
            int code = row * processor_config.grid_col + col;

            FeatureMetaData new_feature;
            new_feature.response = response;
            new_feature.cam0_point = cam0_point;
            new_feature.cam1_point = cam1_point;
            grid_new_features[code].push_back(new_feature);
        }

        // 6. 与本身跟踪的点共同放入curr_features_ptr，同时确保每个格里面的点数不超过设定值
        // 把grid_new_features中每个网格的点按照响应值排序
        for (auto &item : grid_new_features)
            std::sort(item.second.begin(), item.second.end(), &ImageProcessor::featureCompareByResponse);

        int new_added_feature_num = 0;
        // 遍历所有格，如果格内的点数不够，就把新提取的点按照响应值大小往里面填
        for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
        {
            vector<FeatureMetaData> &features_this_grid = (*curr_features_ptr)[code];  // 单个网格内跟踪成功的点
            vector<FeatureMetaData> &new_features_this_grid = grid_new_features[code]; // 单个网格内新提取的点

            // 如果这个格内跟踪的点已经够数了，就不要新的点了
            if (features_this_grid.size() >= processor_config.grid_min_feature_num)
                continue;

            // 否则用新点按照响应值大小往里面填
            int vacancy_num = processor_config.grid_min_feature_num - features_this_grid.size(); // 需要新点的数量
            for (int k = 0; k < vacancy_num && k < new_features_this_grid.size(); ++k)
            {
                features_this_grid.push_back(new_features_this_grid[k]);
                features_this_grid.back().id = next_feature_id++;
                features_this_grid.back().lifetime = 1;

                ++new_added_feature_num;
            }
        }

        // printf("\033[0;33m detected: %d; matched: %d; new added feature: %d\033[0m\n",
        //     detected_new_features, matched_new_features, new_added_feature_num);

        return;
    }

    /**
     * @brief 剔除每个格多余的点
     * @note  为什么addNewFeatures中明明每个格子都是按照最小需要的点数添加的，这里格子内的点还会超出最大值？
     *        因为trackFeatures中只按照每个格子最少的点补齐了， 但某个格子内的点可能移动到了另一个格子，
     * 所以超出了格子内点的最大值
     */
    void ImageProcessor::pruneGridFeatures()
    {
        for (auto &item : *curr_features_ptr)
        {
            auto &grid_features = item.second;
            // Continue if the number of features in this grid does
            // not exceed the upper bound.
            if (grid_features.size() <= processor_config.grid_max_feature_num)
                continue;
            // 这里不是用响应值排序，而是用lifetime排序，认为跟踪时间长的点更稳定
            std::sort(grid_features.begin(), grid_features.end(), &ImageProcessor::featureCompareByLifetime);
            grid_features.erase(grid_features.begin() + processor_config.grid_max_feature_num, grid_features.end());
        }
        return;
    }

    /**
     * @brief 畸变图像中的pts_in点 -> 通过旋转矩阵R转换 -> 另一个图像坐标系中的无畸变归一化坐标 -> 通过新的内参矩阵K_new转换到图像坐标系上
     *        (如果K_new是单位矩阵，那么最后的结果就是另一个图像坐标系的无畸变归一化坐标)
     * @param  pts_in 输入像素点(畸变图像中的)
     * @param  intrinsics 相机内参 fx, fy, cx, cy
     * @param  distortion_model 相机畸变模型
     * @param  distortion_coeffs 畸变系数 k1, k2, p1, p2
     * @param  pts_out 输出矫正后的像素点（归一化的点）
     * @param  rectification_matrix R矩阵，默认为单位阵
     * @param  new_intrinsics 矫正后的内参矩阵, 默认为cv::Vec4d(1, 1, 0, 0)
     * @note   不输入rectification_matrix和new_intrinsics时，相当于只是去畸变，得到的是无畸变归一化坐标
     */
    void ImageProcessor::undistortPoints(
        const vector<cv::Point2f> &pts_in, const cv::Vec4d &intrinsics, const string &distortion_model,
        const cv::Vec4d &distortion_coeffs, vector<cv::Point2f> &pts_out, const cv::Matx33d &rectification_matrix,
        const cv::Vec4d &new_intrinsics)
    {

        if (pts_in.size() == 0)
            return;

        // 构建内参矩阵K
        const cv::Matx33d K(
            intrinsics[0], 0.0, intrinsics[2],
            0.0, intrinsics[1], intrinsics[3],
            0.0, 0.0, 1.0);

        const cv::Matx33d K_new(
            new_intrinsics[0], 0.0, new_intrinsics[2],
            0.0, new_intrinsics[1], new_intrinsics[3],
            0.0, 0.0, 1.0);

        // undistortPoints 计算流程
        // pts_in(畸变图像上点的像素坐标u v) -> 由K到畸变归一化平面上点的归一化坐标(x_distorted, y_distorted，十四讲eq5.13) ->
        // 用畸变系数和畸变模型去畸变得到无畸变图像的归一化坐标(x,y 十四讲eq5.12) -> 通过R矩阵转到另一个图像坐标系上(这里是相机1的无畸变归一化坐标系中) ->
        // 用K_new将新图像坐标系的无畸变归一化坐标转到图像坐标系上，如果K_new是单位矩阵，所以最后的结果就是相机1的无畸变归一化坐标
        if (distortion_model == "radtan")
        {
            cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs, rectification_matrix, K_new);
        }
        else if (distortion_model == "equidistant")
        {
            cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs, rectification_matrix, K_new);
        }
        else
        {
            ROS_WARN_ONCE("The model %s is unrecognized, use radtan instead...", distortion_model.c_str());
            cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs, rectification_matrix, K_new);
        }

        return;
    }

    /**
     * @brief  归一化平面上的坐标 -> 带畸变图像的像素坐标
     * @param  pts_in 归一化坐标的前两维
     * @param  intrinsics 内参
     * @param  distortion_model 畸变模型
     * @param  distortion_coeffs 畸变系数
     * @return 像素坐标的点
     */
    vector<cv::Point2f> ImageProcessor::distortPoints(
        const vector<cv::Point2f> &pts_in, const cv::Vec4d &intrinsics, const string &distortion_model,
        const cv::Vec4d &distortion_coeffs)
    {

        // 1. 内参矩阵
        const cv::Matx33d K(
            intrinsics[0], 0.0, intrinsics[2],
            0.0, intrinsics[1], intrinsics[3],
            0.0, 0.0, 1.0);

        vector<cv::Point2f> pts_out;
        // 2. 不同模型的加畸变函数
        if (distortion_model == "radtan")
        {
            vector<cv::Point3f> homogenous_pts;
            // 转成齐次
            cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
            cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K, distortion_coeffs, pts_out);
        }
        else if (distortion_model == "equidistant")
        {
            cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
        }
        else
        {
            ROS_WARN_ONCE("The model %s is unrecognized, using radtan instead...", distortion_model.c_str());
            vector<cv::Point3f> homogenous_pts;
            cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
            cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K, distortion_coeffs, pts_out);
        }

        return pts_out;
    }

    /**
     * @brief 利用imu的角速度计算前后帧的旋转，左右相机分开计算
     * @param  cam0_R_p_c 左相机prev帧到curr帧的旋转
     * @param  cam1_R_p_c 右相机prev帧到curr帧的旋转
     * @see ImageProcessor::trackFeatures()
     */
    void ImageProcessor::integrateImuData(Matx33f &cam0_R_p_c, Matx33f &cam1_R_p_c)
    {
        // 1. 根据cam0上一帧和当前帧的时间戳，在imu_msg_buffer中寻找cam0上一帧提前0.01s到当前帧推迟0.005s的IMU数据，记录begin_iter和end_iter
        auto begin_iter = imu_msg_buffer.begin();
        while (begin_iter != imu_msg_buffer.end())
        {
            if ((begin_iter->header.stamp - cam0_prev_img_ptr->header.stamp).toSec() < -0.01)
                ++begin_iter;
            else
                break;
        }
        auto end_iter = begin_iter;
        while (end_iter != imu_msg_buffer.end())
        {
            if ((end_iter->header.stamp - cam0_curr_img_ptr->header.stamp).toSec() < 0.005)
                ++end_iter;
            else
                break;
        }

        // 2. 计算这段时间内的平均角速度，不需要严格积分，因为本身就只是提供一个初始值
        Vec3f mean_ang_vel(0.0, 0.0, 0.0);
        for (auto iter = begin_iter; iter < end_iter; ++iter)
            mean_ang_vel += Vec3f(iter->angular_velocity.x, iter->angular_velocity.y, iter->angular_velocity.z);

        if (end_iter - begin_iter > 0)
            mean_ang_vel *= 1.0f / (end_iter - begin_iter);

        // 3. 将IMU系下的平均角速度分别转到左右相机系下
        Vec3f cam0_mean_ang_vel = R_cam0_imu.t() * mean_ang_vel;
        Vec3f cam1_mean_ang_vel = R_cam1_imu.t() * mean_ang_vel;

        // 通过罗德里格斯公式计算旋转矩阵
        double dtime = (cam0_curr_img_ptr->header.stamp - cam0_prev_img_ptr->header.stamp).toSec();
        Rodrigues(cam0_mean_ang_vel * dtime, cam0_R_p_c);
        Rodrigues(cam1_mean_ang_vel * dtime, cam1_R_p_c);
        // 首先 假设Rwp 表示上一帧到世界的旋转 我们用imu积分了角度 得到了Rwc 表示 从c到w的旋转Rwc = Rwp * Rpc
        // 积分都是右乘所以Rwc = Rwp * Rpc
        // Pc表示当前坐标系下的点  Pp表示上一帧坐标系下的点
        // 有 Rwp * Rpc * Pc = Rwp * Pp
        // 推出Pc = Rpc.t * Pp
        // 因此需要在最后加个转置

        // Pc表示当前帧下的点 Pp表示上一帧下的点
        // Pc = Rot(wt) * Pp = Rcp * Pp = cam0_R_p_c * Pp
        // QUERY 为什么加转置
        cam0_R_p_c = cam0_R_p_c.t();
        cam1_R_p_c = cam1_R_p_c.t();

        // 4. 删除中间已用的数据
        imu_msg_buffer.erase(imu_msg_buffer.begin(), end_iter);
        return;
    }

    /**
     * @brief 估计系数，用于算像素单位与归一化平面单位的一个乘数
     * @param  pts1 非归一化坐标
     * @param  pts2 归一化坐标
     * @param  scaling_factor 尺度
     */
    void ImageProcessor::rescalePoints(vector<Point2f> &pts1, vector<Point2f> &pts2, float &scaling_factor)
    {

        scaling_factor = 0.0f;

        // 求所有点的sqrt(x^2 + y^2)和, 即求所有归一化平面上点的斜边和
        for (int i = 0; i < pts1.size(); ++i)
        {
            scaling_factor += sqrt(pts1[i].dot(pts1[i]));
            scaling_factor += sqrt(pts2[i].dot(pts2[i]));
        }
        // 点个数 / (斜边和 / sqrt(2)) = 点个数 / 斜边和 * sqrt(2)，
        // 这里面为什么还有个根号2呢，前面得出的尺度系数是在斜边的基础上求的，但实际应用的时候是长宽级别的，所以要除根号2
        scaling_factor = (pts1.size() + pts2.size()) / scaling_factor * sqrt(2.0f);
        // 每个点乘这个数
        for (int i = 0; i < pts1.size(); ++i)
        {
            pts1[i] *= scaling_factor;
            pts2[i] *= scaling_factor;
        }

        return;
    }

    /**
     * @brief 通过两点RANSAC进一步去除外点并把结果保存在inlier_markers中
     *        1. 在所有点中随机选择两个点
     *        2. 用这两个点利用对极约束，根据IMU预测的R，计算出t，然后用t判断每个点是否满足对极约束
     *        3. 利用上一步选出的所有内点计算一个综合的t，并计算误差
     *        4. 循环1-3，直到找到一个最优的t
     * @param  pts1 上一帧内图像上的点
     * @param  pts2 当前帧内图像上的点
     * @param  R_p_c 上一帧到当前帧的旋转,IMU预测的
     * @param  intrinsics 内参
     * @param  distortion_model 畸变模型
     * @param  distortion_coeffs 畸变参数
     * @param  inlier_error 内点误差，像素，也就是ransac的阈值
     * @param  success_probability 成功率, 即有一次选择到都是内点的概率，一般设为0.99
     * @param  inlier_markers 内外点标志
     * @see ImageProcessor::trackFeatures()
     */
    void ImageProcessor::twoPointRansac(
        const vector<Point2f> &pts1, const vector<Point2f> &pts2, const cv::Matx33f &R_p_c, const cv::Vec4d &intrinsics,
        const std::string &distortion_model, const cv::Vec4d &distortion_coeffs, const double &inlier_error,
        const double &success_probability, vector<int> &inlier_markers)
    {

        // Check the size of input point size.
        if (pts1.size() != pts2.size())
            ROS_ERROR("Sets of different size (%lu and %lu) are used...", pts1.size(), pts2.size());

        double norm_pixel_unit = 2.0 / (intrinsics[0] + intrinsics[1]);
        // ransac迭代次数， success_probability 表示希望RANSAC得到正确模型的概率，认为内点比率是0.7，每次取两对点
        int iter_num = static_cast<int>(ceil(log(1 - success_probability) / log(1 - 0.7 * 0.7)));

        // Initially, mark all points as inliers.
        inlier_markers.clear();
        inlier_markers.resize(pts1.size(), 1);

        // 1. 两批点全部投到矫正后的归一化平面坐标下
        vector<Point2f> pts1_undistorted(pts1.size());
        vector<Point2f> pts2_undistorted(pts2.size());
        undistortPoints(pts1, intrinsics, distortion_model, distortion_coeffs, pts1_undistorted);
        undistortPoints(pts2, intrinsics, distortion_model, distortion_coeffs, pts2_undistorted);

        // 2. 通过旋转将pts1_undistorted点转换到pts2_undistorted的相机坐标下，注意这里没做平移，且没有将第三维弄成1，即结果不是归一化平面了
        // 相当于十四讲eq7.8中R*x1
        for (auto &pt : pts1_undistorted)
        {
            Vec3f pt_h(pt.x, pt.y, 1.0f);
            // Vec3f pt_hc = dR * pt_h;
            Vec3f pt_hc = R_p_c * pt_h;
            pt.x = pt_hc[0];
            pt.y = pt_hc[1];
        }

        // Normalize the points to gain numerical stability.
        float scaling_factor = 0.0f;
        // 求出两波点的平均尺度，每个点再做尺度归一化
        rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
        // 焦距倒数乘尺度，阈值与点的坐标尺度保持一致
        norm_pixel_unit *= scaling_factor;

        // 3. 计算对应的两批点在尺度归一化的坐标差
        vector<Point2d> pts_diff(pts1_undistorted.size());
        for (int i = 0; i < pts1_undistorted.size(); ++i)
            pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];

        // Mark the point pairs with large difference directly.
        // BTW(by the way), the mean distance of the rest of the point pairs
        // are computed.
        // 4. 通过上面的差值剔除一些较大的，因为要做RANSAC，所以尽量的先把外点剔除一部分
        double mean_pt_distance = 0.0;
        int raw_inlier_cntr = 0;
        for (int i = 0; i < pts_diff.size(); ++i)
        {
            double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
            // 25 pixel distance is a pretty large tolerance for normal motion.
            // However, to be used with aggressive motion, this tolerance should
            // be increased significantly to match the usage.
            // 50 代表像素值, 50 * norm_pixel_unit 即 像素除以焦距，得到的是归一化平面上的距离
            // 然后因为norm_pixel_unit又乘了一个scaling_factor，所以代表在尺度归一化平面上的距离
            if (distance > 50.0 * norm_pixel_unit)
            {
                inlier_markers[i] = 0;
            }
            else
            {
                mean_pt_distance += distance;
                ++raw_inlier_cntr;
            }
        }
        // 计算保留下来的差值的均值
        mean_pt_distance /= raw_inlier_cntr;

        // If the current number of inliers is less than 3, just mark
        // all input as outliers. This case can happen with fast
        // rotation where very few features are tracked.
        // 5. 快速旋转将导致内点很少，返回。
        // 前端没有所谓的是否丢失，最多是没有点跟踪上，导致后端没办法更新而已，即如果前端没点来，后端会单纯用IMU一直递推
        if (raw_inlier_cntr < 3)
        {
            for (auto &marker : inlier_markers)
                marker = 0;
            return;
        }

        // Before doing 2-point RANSAC, we have to check if the motion
        // is degenerated, meaning that there is no translation between
        // the frames, in which case, the model of the RANSAC does not
        // work. If so, the distance between the matched points will
        // be almost 0.
        // if (mean_pt_distance < inlier_error*norm_pixel_unit) {
        // 6. 如果平均差较小， 一个像素 ，就不做ransac了，因为此时平移几乎是0，运动退化，没必要在估计t了，所以不用下面的RANSAC
        // 使用经验值进一步剔除外点
        if (mean_pt_distance < norm_pixel_unit)
        {
            // ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
            for (int i = 0; i < pts_diff.size(); ++i)
            {
                if (inlier_markers[i] == 0)
                    continue;
                if (sqrt(pts_diff[i].dot(pts_diff[i])) > inlier_error * norm_pixel_unit)
                    inlier_markers[i] = 0;
            }
            return;
        }

        // In the case of general motion, the RANSAC model can be applied.
        // The three column corresponds to tx, ty, and tz respectively.
        // 7. ransac
        MatrixXd coeff_t(pts_diff.size(), 3);
        // 7.1 构建方程 y1 - y2     -(x1 - x2)    x1y2 - x2y1，
        // 即十四讲eq7.8 x2^T * t^ * R * x1 = 0 --> - x2^T * (R * x1)^ * t = 0
        // 这里求的便是- x2^T * (R * x1)^
        for (int i = 0; i < pts_diff.size(); ++i)
        {
            coeff_t(i, 0) = pts_diff[i].y;
            coeff_t(i, 1) = -pts_diff[i].x;
            coeff_t(i, 2) = pts1_undistorted[i].x * pts2_undistorted[i].y - pts1_undistorted[i].y * pts2_undistorted[i].x;
        }

        // 7.2 记录剩下可用内点的索引
        vector<int> raw_inlier_idx;
        for (int i = 0; i < inlier_markers.size(); ++i)
        {
            if (inlier_markers[i] != 0)
                raw_inlier_idx.push_back(i);
        }

        // 开始RANSAC迭代
        vector<int> best_inlier_set;
        double best_error = 1e10;
        random_numbers::RandomNumberGenerator random_gen;
        for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx)
        {
            // Randomly select two point pairs.
            // Although this is a weird way of selecting two pairs, but it
            // is able to efficiently avoid selecting repetitive pairs.
            // 7.3 选择非重复的两个点
            int select_idx1 = random_gen.uniformInteger(0, raw_inlier_idx.size() - 1);
            int select_idx_diff = random_gen.uniformInteger(1, raw_inlier_idx.size() - 1);
            // 到这里上面两个数有可能出现重复的，经过下面处理实现获取两个不相等的id
            int select_idx2 =
                (select_idx1 + select_idx_diff) < raw_inlier_idx.size() ? select_idx1 + select_idx_diff : select_idx1 + select_idx_diff - raw_inlier_idx.size();
            // 取出两个点的索引值
            int pair_idx1 = raw_inlier_idx[select_idx1];
            int pair_idx2 = raw_inlier_idx[select_idx2];

            // | y1 - y2, -(x1 - x2), x1y2 - x2y1 |         tx       0
            // |                                  |    *    ty   ~=
            // | y3 - y4, -(x3 - x4), x3y4 - x4y3 |         tz       0

            // Construct the model;
            // coeff_tx = [y1 - y2, y3 - y4] = Ax.T
            // coeff_ty = [-(x1 - x2), -(x3 - x4)] = Ay.T
            // coeff_tz = [x1y2 - x2y1, x3y4 - x4y3] = Az.T
            Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
            Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
            Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
            vector<double> coeff_l1_norm(3);
            // 1范数，所有元素绝对值的合
            coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
            coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
            coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
            // min_element返回绝对值最小的元素的迭代器，然后减去begin的迭代器，得到的是两个迭代器间的距离
            // 当base_indicator=0即第一个元素，当base_indicator=1即第二个元素，当base_indicator=2即第三个元素
            int base_indicator = min_element(coeff_l1_norm.begin(), coeff_l1_norm.end()) - coeff_l1_norm.begin();

            // 这里想要取其中两个做成方阵，到底选哪两个？
            // 首先不管选哪两个，得出的方阵由于有误差的存在，它是绝对可逆的
            // 所以挑选了数值相对较大的两个，因为在相同误差下，数值大的误差比例小
            // 求出的逆更加“可靠”
            // 然后这里的model即求出了对极几何中的t
            Vector3d model(0.0, 0.0, 0.0);
            if (base_indicator == 0)
            {
                Matrix2d A;
                A << coeff_ty, coeff_tz;
                Vector2d solution = A.inverse() * (-coeff_tx);
                model(0) = 1.0;
                model(1) = solution(0);
                model(2) = solution(1);
            }
            else if (base_indicator == 1)
            {
                Matrix2d A;
                A << coeff_tx, coeff_tz;
                Vector2d solution = A.inverse() * (-coeff_ty);
                model(0) = solution(0);
                model(1) = 1.0;
                model(2) = solution(1);
            }
            else
            {
                Matrix2d A;
                A << coeff_tx, coeff_ty;
                Vector2d solution = A.inverse() * (-coeff_tz);
                model(0) = solution(0);
                model(1) = solution(1);
                model(2) = 1.0;
            }

            // 7.4 根据求出的t进行验证，看这个t能否让所有点都尽可能满足对极约束，即error尽可能为0
            // 即十四讲eq7.8 x2^T * t^ * R * x1 = 0 --> - x2^T * (R * x1)^ * t = 0
            VectorXd error = coeff_t * model;
            vector<int> inlier_set;
            for (int i = 0; i < error.rows(); ++i)
            {
                if (inlier_markers[i] == 0)
                    continue;

                // 点越准，这个error越接近0
                if (std::abs(error(i)) < inlier_error * norm_pixel_unit)
                    inlier_set.push_back(i);
            }

            // If the number of inliers is small, the current model is probably wrong.
            // 内点数量太少，跳过这组结果，换其他2个点再计算
            if (inlier_set.size() < 0.2 * pts1_undistorted.size())
                continue;

            // 7.5 下面这段类似于上面计算t的过程，只不过这里是用第一步RANSAC选出的所有内点再次计算t
            // Refit the model using all of the possible inliers.
            VectorXd coeff_tx_better(inlier_set.size());
            VectorXd coeff_ty_better(inlier_set.size());
            VectorXd coeff_tz_better(inlier_set.size());
            for (int i = 0; i < inlier_set.size(); ++i)
            {
                coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
                coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
                coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
            }

            Vector3d model_better(0.0, 0.0, 0.0);
            if (base_indicator == 0)
            {
                MatrixXd A(inlier_set.size(), 2);
                A << coeff_ty_better, coeff_tz_better;
                // 计算伪逆
                Vector2d solution =
                    (A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
                model_better(0) = 1.0;
                model_better(1) = solution(0);
                model_better(2) = solution(1);
            }
            else if (base_indicator == 1)
            {
                MatrixXd A(inlier_set.size(), 2);
                A << coeff_tx_better, coeff_tz_better;
                Vector2d solution =
                    (A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
                model_better(0) = solution(0);
                model_better(1) = 1.0;
                model_better(2) = solution(1);
            }
            else
            {
                MatrixXd A(inlier_set.size(), 2);
                A << coeff_tx_better, coeff_ty_better;
                Vector2d solution =
                    (A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
                model_better(0) = solution(0);
                model_better(1) = solution(1);
                model_better(2) = 1.0;
            }

            // Compute the error and upate the best model if possible.
            VectorXd new_error = coeff_t * model_better;

            // 用此次两点RANSAC筛选出的所有内点计算出t后，计算此次迭代的平均误差this_error
            // 如果内点数量大于之前的，就更新best_error和best_inlier_set
            double this_error = 0.0;
            for (const auto &inlier_idx : inlier_set)
                this_error += std::abs(new_error(inlier_idx));
            this_error /= inlier_set.size();
            if (inlier_set.size() > best_inlier_set.size())
            {
                best_error = this_error;
                best_inlier_set = inlier_set;
            }
        }

        // 8. 保存最后的内点标志
        inlier_markers.clear();
        inlier_markers.resize(pts1.size(), 0);
        for (const auto &inlier_idx : best_inlier_set)
            inlier_markers[inlier_idx] = 1;

.
        // printf("inlier ratio: %lu/%lu\n",
        //     best_inlier_set.size(), inlier_markers.size());

        return;
    }

    /**
     * @brief 发送前端提取的特征点和id到后端
     */
    void ImageProcessor::publish()
    {

        // Publish features.
        CameraMeasurementPtr feature_msg_ptr(new CameraMeasurement);
        feature_msg_ptr->header.stamp = cam0_curr_img_ptr->header.stamp;

        // 1. 取出当前帧所有跟踪上的+新提取的特征点， 的id和位置
        vector<FeatureIDType> curr_ids(0);
        vector<Point2f> curr_cam0_points(0);
        vector<Point2f> curr_cam1_points(0);
        for (const auto &grid_features : (*curr_features_ptr))
        {
            for (const auto &feature : grid_features.second)
            {
                curr_ids.push_back(feature.id);
                curr_cam0_points.push_back(feature.cam0_point);
                curr_cam1_points.push_back(feature.cam1_point);
            }
        }

        vector<Point2f> curr_cam0_points_undistorted(0);
        vector<Point2f> curr_cam1_points_undistorted(0);
        // 2. 去畸变，注意！！！！！这里的输入最后的相机内参矩阵没有，所以输出为归一化的坐标
        undistortPoints(
            curr_cam0_points, cam0_intrinsics, cam0_distortion_model,
            cam0_distortion_coeffs, curr_cam0_points_undistorted);
        undistortPoints(
            curr_cam1_points, cam1_intrinsics, cam1_distortion_model,
            cam1_distortion_coeffs, curr_cam1_points_undistorted);

        // 3. 发送消息，存放的是所有点以及id
        for (int i = 0; i < curr_ids.size(); ++i)
        {
            feature_msg_ptr->features.push_back(FeatureMeasurement());
            feature_msg_ptr->features[i].id = curr_ids[i];
            feature_msg_ptr->features[i].u0 = curr_cam0_points_undistorted[i].x;
            feature_msg_ptr->features[i].v0 = curr_cam0_points_undistorted[i].y;
            feature_msg_ptr->features[i].u1 = curr_cam1_points_undistorted[i].x;
            feature_msg_ptr->features[i].v1 = curr_cam1_points_undistorted[i].y;
        }

        feature_pub.publish(feature_msg_ptr);

        // Publish tracking info.
        // 4. 发一些个数类型的数据
        TrackingInfoPtr tracking_info_msg_ptr(new TrackingInfo());
        tracking_info_msg_ptr->header.stamp = cam0_curr_img_ptr->header.stamp;
        tracking_info_msg_ptr->before_tracking = before_tracking;
        tracking_info_msg_ptr->after_tracking = after_tracking;
        tracking_info_msg_ptr->after_matching = after_matching;
        tracking_info_msg_ptr->after_ransac = after_ransac;
        tracking_info_pub.publish(tracking_info_msg_ptr);

        return;
    }

    /**
     * @brief 做图相关，且没有用到
     */
    void ImageProcessor::drawFeaturesMono()
    {
        // Colors for different features.
        Scalar tracked(0, 255, 0);
        Scalar new_feature(0, 255, 255);

        static int grid_height =
            cam0_curr_img_ptr->image.rows / processor_config.grid_row;
        static int grid_width =
            cam0_curr_img_ptr->image.cols / processor_config.grid_col;

        // Create an output image.
        int img_height = cam0_curr_img_ptr->image.rows;
        int img_width = cam0_curr_img_ptr->image.cols;
        Mat out_img(img_height, img_width, CV_8UC3);
        cvtColor(cam0_curr_img_ptr->image, out_img, CV_GRAY2RGB);

        // Draw grids on the image.
        for (int i = 1; i < processor_config.grid_row; ++i)
        {
            Point pt1(0, i * grid_height);
            Point pt2(img_width, i * grid_height);
            line(out_img, pt1, pt2, Scalar(255, 0, 0));
        }
        for (int i = 1; i < processor_config.grid_col; ++i)
        {
            Point pt1(i * grid_width, 0);
            Point pt2(i * grid_width, img_height);
            line(out_img, pt1, pt2, Scalar(255, 0, 0));
        }

        // Collect features ids in the previous frame.
        vector<FeatureIDType> prev_ids(0);
        for (const auto &grid_features : *prev_features_ptr)
            for (const auto &feature : grid_features.second)
                prev_ids.push_back(feature.id);

        // Collect feature points in the previous frame.
        map<FeatureIDType, Point2f> prev_points;
        for (const auto &grid_features : *prev_features_ptr)
            for (const auto &feature : grid_features.second)
                prev_points[feature.id] = feature.cam0_point;

        // Collect feature points in the current frame.
        map<FeatureIDType, Point2f> curr_points;
        for (const auto &grid_features : *curr_features_ptr)
            for (const auto &feature : grid_features.second)
                curr_points[feature.id] = feature.cam0_point;

        // Draw tracked features.
        for (const auto &id : prev_ids)
        {
            if (prev_points.find(id) != prev_points.end() &&
                curr_points.find(id) != curr_points.end())
            {
                cv::Point2f prev_pt = prev_points[id];
                cv::Point2f curr_pt = curr_points[id];
                circle(out_img, curr_pt, 3, tracked);
                line(out_img, prev_pt, curr_pt, tracked, 1);

                prev_points.erase(id);
                curr_points.erase(id);
            }
        }

        // Draw new features.
        for (const auto &new_curr_point : curr_points)
        {
            cv::Point2f pt = new_curr_point.second;
            circle(out_img, pt, 3, new_feature, -1);
        }

        imshow("Feature", out_img);
        waitKey(5);
    }

    /**
     * @brief 当有其他节点订阅了debug_stereo_image话题时，将双目图像拼接起来并画出特征点位置，作为消息发送出去
     * @note  绿色: 跟踪成功的点 金色: 新添加的点
     */
    void ImageProcessor::drawFeaturesStereo()
    {
        // 如果有节点订阅了debug_stereo_image消息
        if (debug_stereo_pub.getNumSubscribers() > 0)
        {
            // 跟踪的成功的点是绿色的
            Scalar tracked(0, 255, 0);
            // 新点是金色的
            Scalar new_feature(0, 255, 255);

            // 网格大小
            static int grid_height =
                cam0_curr_img_ptr->image.rows / processor_config.grid_row;
            static int grid_width =
                cam0_curr_img_ptr->image.cols / processor_config.grid_col;

            // 1. 把左右目图片拼一块
            int img_height = cam0_curr_img_ptr->image.rows;
            int img_width = cam0_curr_img_ptr->image.cols;
            Mat out_img(img_height, img_width * 2, CV_8UC3);
            cvtColor(cam0_curr_img_ptr->image,
                     out_img.colRange(0, img_width), CV_GRAY2RGB);
            cvtColor(cam1_curr_img_ptr->image,
                     out_img.colRange(img_width, img_width * 2), CV_GRAY2RGB);

            // 2. 画网格
            for (int i = 1; i < processor_config.grid_row; ++i) // 行
            {
                Point pt1(0, i * grid_height);
                Point pt2(img_width * 2, i * grid_height);
                line(out_img, pt1, pt2, Scalar(255, 0, 0), 2);
            }
            for (int i = 1; i < processor_config.grid_col; ++i) // 左图列
            {
                Point pt1(i * grid_width, 0);
                Point pt2(i * grid_width, img_height);
                line(out_img, pt1, pt2, Scalar(255, 0, 0), 2);
            }
            for (int i = 1; i < processor_config.grid_col; ++i) // 右图列
            {
                Point pt1(i * grid_width + img_width, 0);
                Point pt2(i * grid_width + img_width, img_height);
                line(out_img, pt1, pt2, Scalar(255, 0, 0), 2);
            }

            // 3. 取出上一帧特征点id
            vector<FeatureIDType> prev_ids(0);
            for (const auto &grid_features : *prev_features_ptr)
                for (const auto &feature : grid_features.second)
                    prev_ids.push_back(feature.id);

            // 4. 取出上一帧特征点id与对应的左右目特征点
            map<FeatureIDType, Point2f> prev_cam0_points;
            map<FeatureIDType, Point2f> prev_cam1_points;
            for (const auto &grid_features : *prev_features_ptr)
                for (const auto &feature : grid_features.second)
                {
                    prev_cam0_points[feature.id] = feature.cam0_point;
                    prev_cam1_points[feature.id] = feature.cam1_point;
                }

            // 5. 取出当前帧特征点id与对应的左右目特征点
            map<FeatureIDType, Point2f> curr_cam0_points;
            map<FeatureIDType, Point2f> curr_cam1_points;
            for (const auto &grid_features : *curr_features_ptr)
                for (const auto &feature : grid_features.second)
                {
                    curr_cam0_points[feature.id] = feature.cam0_point;
                    curr_cam1_points[feature.id] = feature.cam1_point;
                }

            // 6. 绘制跟踪成功的点，左右图像上当前帧的点会画出来，然后画出上一帧到当前帧的连线，视觉上展示就是一个拖尾
            for (const auto &id : prev_ids)
            {
                if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
                    curr_cam0_points.find(id) != curr_cam0_points.end()) // 如果找不到id, find会返回end()迭代器
                {
                    // 计算在拼接图像中的坐标
                    cv::Point2f prev_pt0 = prev_cam0_points[id];
                    cv::Point2f prev_pt1 = prev_cam1_points[id] + Point2f(img_width, 0.0);
                    cv::Point2f curr_pt0 = curr_cam0_points[id];
                    cv::Point2f curr_pt1 = curr_cam1_points[id] + Point2f(img_width, 0.0);

                    // 当前帧左右目特征点
                    circle(out_img, curr_pt0, 3, tracked, -1);
                    circle(out_img, curr_pt1, 3, tracked, -1);
                    // 上一帧到当前帧的连线，视觉上是一个拖尾
                    line(out_img, prev_pt0, curr_pt0, tracked, 1);
                    line(out_img, prev_pt1, curr_pt1, tracked, 1);

                    // 删除绘制过的跟踪成功的点
                    prev_cam0_points.erase(id);
                    prev_cam1_points.erase(id);
                    curr_cam0_points.erase(id);
                    curr_cam1_points.erase(id);
                }
            }

            // 7. 绘制新添加的点
            for (const auto &new_cam0_point : curr_cam0_points)
            {
                cv::Point2f pt0 = new_cam0_point.second;
                cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] +
                                  Point2f(img_width, 0.0);

                circle(out_img, pt0, 3, new_feature, -1);
                circle(out_img, pt1, 3, new_feature, -1);
            }

            // 8. 格式转换
            cv_bridge::CvImage debug_image(cam0_curr_img_ptr->header, "bgr8", out_img);
            debug_stereo_pub.publish(debug_image.toImageMsg());
        }
        // imshow("Feature", out_img);
        // waitKey(5);

        return;
    }

    /**
     * @brief 没用到，计算特征点的生命周期
     */
    void ImageProcessor::updateFeatureLifetime()
    {
        for (int code = 0; code < processor_config.grid_row * processor_config.grid_col; ++code)
        {
            vector<FeatureMetaData> &features = (*curr_features_ptr)[code];
            for (const auto &feature : features)
            {
                if (feature_lifetime.find(feature.id) == feature_lifetime.end())
                    feature_lifetime[feature.id] = 1;
                else
                    ++feature_lifetime[feature.id];
            }
        }
        return;
    }

    /**
     * @brief 没用到，输出每个点的生命周期
     */
    void ImageProcessor::featureLifetimeStatistics()
    {

        map<int, int> lifetime_statistics;
        for (const auto &data : feature_lifetime) // std::map<FeatureIDType, int>
        {
            if (lifetime_statistics.find(data.second) == lifetime_statistics.end())
                lifetime_statistics[data.second] = 1;
            else
                ++lifetime_statistics[data.second];
        }

        for (const auto &data : lifetime_statistics)
            cout << data.first << " : " << data.second << endl;

        return;
    }

} // end namespace msckf_vio
