/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <msckf_vio/image_processor_nodelet.h>

namespace msckf_vio
{
    /// @brief 创建前端ImageProcessor节点并初始化ImageProcessor类
    /// @see ImageProcessor::ImageProcessor(ros::NodeHandle &n)
    /// @see ImageProcessor::initialize()
    void ImageProcessorNodelet::onInit()
    {
        setlocale(LC_ALL, "");
        img_processor_ptr.reset(new ImageProcessor(getPrivateNodeHandle()));
        auto nh = getPrivateNodeHandle();
        std::string logger_level;
        if(!nh.param<std::string>("logger_level", logger_level, "Info"))
            ROS_WARN("ip Cannot find logger_level parameter, use default value: Info");
        if(logger_level == "Debug")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
        else if(logger_level == "Info")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
        else if(logger_level == "Warn")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
        else if(logger_level == "Error")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Error);
        else if(logger_level == "Fatal")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal);
        else{
            ROS_WARN("Unknown logger level: %s, use default value: Info", logger_level.c_str());
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
        }
        if (!img_processor_ptr->initialize())
        {
            ROS_ERROR("Cannot initialize Image Processor...");
            return;
        }
        return;
    }

    // 将ImageProcessorNodelet类注册为ROS的插件
    PLUGINLIB_EXPORT_CLASS(msckf_vio::ImageProcessorNodelet, nodelet::Nodelet);

} // end namespace msckf_vio
