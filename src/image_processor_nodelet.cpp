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
