/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef IMAGE_PROCESSOR_NODELET_H
#define IMAGE_PROCESSOR_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <msckf_vio/image_processor.h>

namespace msckf_vio
{
  /**
   * @brief ImageProcessorNodelet 继承自ros中的Nodelet类
   *        Nodelet类是ROS中的一种插件类，可以在一个节点中运行多个Nodelet插件, 但是实际运行时
   *        看起来就像是多个节点在运行。Nodelet插件通过将多个节点的功能集成到一个进程中，可以
   *        减少节点间的通信开销，提高系统的性能，这对于传输图像和点云特别有用。
   */
  class ImageProcessorNodelet : public nodelet::Nodelet
  {
  public:
    ImageProcessorNodelet() { return; }
    ~ImageProcessorNodelet() { return; }

  private:
    virtual void onInit();
    ImageProcessorPtr img_processor_ptr;
  };
} // end namespace msckf_vio

#endif
