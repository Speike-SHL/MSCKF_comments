/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <msckf_vio/msckf_vio_nodelet.h>

namespace msckf_vio
{
    /// @brief 创建后端节点并初始化MsckfVio类
    /// @see MsckfVio::MsckfVio(ros::NodeHandle &pnh)
    /// @see MsckfVio::initialize()
    void MsckfVioNodelet::onInit()
    {
        setlocale(LC_ALL, "");
        msckf_vio_ptr.reset(new MsckfVio(getPrivateNodeHandle()));

        auto nh = getPrivateNodeHandle();
        std::string msckf_vio_config_file;
        nh.param<std::string>("msckf_vio_config_file", msckf_vio_config_file, "config/default.yaml");
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
        std::string logger_level;
        logger_level = config["settings"]["logger_level"] ? config["settings"]["logger_level"].as<std::string>() : "UnknownLevel";
        if (logger_level == "Debug")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
        else if (logger_level == "Info")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
        else if (logger_level == "Warn")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
        else if (logger_level == "Error")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Error);
        else if (logger_level == "Fatal")
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal);
        else
        {
            ROS_WARN("Unknown logger level: %s, use default value: Info", logger_level.c_str());
            ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
        }

        if (!msckf_vio_ptr->initialize())
        {
            ROS_ERROR("Cannot initialize MSCKF VIO...");
            return;
        }
        return;
    }

    PLUGINLIB_EXPORT_CLASS(msckf_vio::MsckfVioNodelet, nodelet::Nodelet);

} // end namespace msckf_vio
