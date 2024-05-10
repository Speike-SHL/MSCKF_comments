/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_FEATURE_H
#define MSCKF_VIO_FEATURE_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "math_utils.hpp"
#include "imu_state.h"
#include "cam_state.h"

namespace msckf_vio
{

    /**
     * @brief Feature Salient part of an image. Please refer
     *    to the Appendix of "A Multi-State Constraint Kalman
     *    Filter for Vision-aided Inertial Navigation" for how
     *    the 3d position of a feature is initialized.
     *    一个特征可以理解为一个三维点，由多帧观测到
     */
    struct Feature
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef long long int FeatureIDType;

        /**
         * @brief 优化求解特征点世界系下三维坐标的配置
         */
        struct OptimizationConfig
        {
            /// 位移是否足够，用于判断点是否能做三角化
            double translation_threshold;
            /// huber参数
            double huber_epsilon;
            /// 迭代更新阈值，优化的每次迭代都会有更新量，这个量如果太小则表示与目标值接近
            double estimation_precision;
            /// LM算法lambda的初始值
            double initial_damping;

            /// 内外轮最大迭代次数
            int outer_loop_max_iteration;
            int inner_loop_max_iteration;

            OptimizationConfig()
                : translation_threshold(0.2),
                  huber_epsilon(0.01),
                  estimation_precision(5e-7),
                  initial_damping(1e-3),
                  outer_loop_max_iteration(10),
                  inner_loop_max_iteration(10)
            {
                return;
            }
        };

        /// 构造函数，点ID，点位置，是否初始化
        Feature() : id(0), position(Eigen::Vector3d::Zero()), is_initialized(false) {}

        Feature(const FeatureIDType &new_id) : id(new_id), position(Eigen::Vector3d::Zero()),
                                            is_initialized(false) {}

        inline void cost(
            const Eigen::Isometry3d &T_c0_ci,
            const Eigen::Vector3d &x, const Eigen::Vector2d &z,
            double &e) const;

        /**
         * @brief jacobian 求一个观测对应的雅可比
         * @param T_c0_ci 相对位姿，Tcic0 第一帧相机在第i帧相机下的位姿
         * @param x 第一帧相机下的估计的三维点坐标 的逆深度形式(x/z, y/z, 1/z)
         * @param z ci下的真实观测归一化坐标
         * @param J 雅可比 归一化坐标误差相对于三维点的
         * @param r 误差
         * @param w 权重，鲁棒核函数
         * @return 是否三角化成功
         */
        inline void jacobian(
            const Eigen::Isometry3d &T_c0_ci,
            const Eigen::Vector3d &x, const Eigen::Vector2d &z,
            Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r,
            double &w) const;

        /**
         * @brief generateInitialGuess Compute the initial guess of
         *    the feature's 3d position using only two views.
         * @param T_c1_c2: A rigid body transformation taking
         *    a vector from c2 frame to c1 frame.
         * @param z1: feature observation in c1 frame.
         * @param z2: feature observation in c2 frame.
         * @return p: Computed feature position in c1 frame.
         */
        inline void generateInitialGuess(
            const Eigen::Isometry3d &T_c1_c2, const Eigen::Vector2d &z1,
            const Eigen::Vector2d &z2, Eigen::Vector3d &p) const;

        /**
         * @brief 检查输入的相机姿态，确保有足够的平移来进行三角化
         * @param cam_states : 所有参与计算的相机位姿
         * @return 如果输入的相机姿态之间的平移足够，则返回True。
         */
        inline bool checkMotion(
            const CamStateServer &cam_states) const;

        /**
         * @brief 根据当前所有可用的观测初始化特征点的位置，进行三角化 + LM优化
         * @param cam_states: A map containing the camera poses with its
         *    ID as the associated key value.
         * @return The computed 3d position is used to set the position
         *    member variable. Note the resulted position is in world
         *    frame.
         * @return 是否三角化成功
         * @note Please refer to the Appendix of "A Multi-State Constraint Kalman
         *    Filter for Vision-aided Inertial Navigation" for how the 3d position of a feature is initialized.
         */
        inline bool initializePosition(
            const CamStateServer &cam_states);

        /// 特征点唯一的ID,long long int
        FeatureIDType id;

        /// 下一个特征点应该使用的ID
        static FeatureIDType next_id;

        /// 观测，储存当前特征点被哪些相机帧观测到，key为相机帧id，value为该特征点在该相机帧下的归一化坐标
        /// 还指定了比较器和分配器用于排序和内存分配，根据id升序排序
        std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
                 Eigen::aligned_allocator<std::pair<const StateIDType, Eigen::Vector4d>>>
            observations;

        /// 特征点的3D位置，世界坐标系下
        Eigen::Vector3d position;

        /// 3d特征点是否初始化, 即是否三角化
        bool is_initialized;

        /// 归一化特征观测的噪声
        static double observation_noise;

        // Optimization configuration for solving the 3d position.
        static OptimizationConfig optimization_config;
    };

    typedef Feature::FeatureIDType FeatureIDType;

    /// 键为特征点id(long long int)，值为特征点(feature), 升序排序，内存分配器为Eigen::aligned_allocator
    typedef std::map<FeatureIDType, Feature, std::less<int>,
                     Eigen::aligned_allocator<std::pair<const FeatureIDType, Feature>>>
        MapServer;

    bool Feature::checkMotion(
        const CamStateServer &cam_states) const
    {
        // 1. 取出单个特征点对应的始末帧id，即单个特征点被观测到的第一帧和最后一帧
        const StateIDType &first_cam_id = observations.begin()->first;
        const StateIDType &last_cam_id = (--observations.end())->first;

        // 2. 分别赋值位姿
        // 左相机第一帧的Twc
        Eigen::Isometry3d first_cam_pose;
        // 左相机第一帧的Rcw -> Rwc
        first_cam_pose.linear() =
            quaternionToRotation(cam_states.find(first_cam_id)->second.orientation).transpose();
        // 左相机第一帧的twc
        first_cam_pose.translation() =
            cam_states.find(first_cam_id)->second.position;
        // 左相机最后一帧的Twc
        Eigen::Isometry3d last_cam_pose;
        // 左相机最后一帧的Rcw -> Rwc
        last_cam_pose.linear() =
            quaternionToRotation(cam_states.find(last_cam_id)->second.orientation).transpose();
        // 左相机最后一帧的twc
        last_cam_pose.translation() =
            cam_states.find(last_cam_id)->second.position;

        // Get the direction of the feature when it is first observed.
        // This direction is represented in the world frame.
        // 3. 求出投影射线在世界坐标系下的方向
        Eigen::Vector3d feature_direction(
            observations.begin()->second(0),
            observations.begin()->second(1), 1.0);                        // 特征在第一帧的左相机上的归一化坐标
        feature_direction = feature_direction / feature_direction.norm(); // 特征在第一帧的左相机上的方向，从相机原点指向归一化坐标的射线，相机系中
        feature_direction = first_cam_pose.linear() * feature_direction;  // 特征在第一帧的世界坐标系下的方向

        // Compute the translation between the first frame
        // and the last frame. We assume the first frame and
        // the last frame will provide the largest motion to
        // speed up the checking process.
        // 4. 求出始末两帧在世界坐标系下的位移（这段判断非常精彩！！！！）
        Eigen::Vector3d translation =
            last_cam_pose.translation() - first_cam_pose.translation();

        // 这里相当于两个向量点乘 这个结果等于两个向量的模乘以cos夹角
        // 也相当于translation 在 feature_direction上的投影
        // 其实就是translation在feature_direction方向上的长度
        double parallel_translation =
            translation.transpose() * feature_direction;

        // 这块直接理解比较抽象，假设十四讲图7-11中，左图为第一帧，右图为最后一帧，
        // O1p1为投影射线在世界系下的方向，因为为单位向量，长度为1
        // O1O2为始末两帧的位移即translation，假设长度为t
        // 那么parallel_translation就是O1O2在O1p1上的投影，即O1O2在O1p1上的长度，假设O1p1与O1O2的夹角为0，等于t * cosθ
        // 使用带入法，分别带入 0° 180° 跟90°
        // 0°，位移方向与射线方向相同，translation方向与feature_direction方向相同， parallel_translation = t, orthogonal_translation = 0
        // 180°，位移方向与射线方向相反，translation方向与feature_direction方向相反， parallel_translation = -t, orthogonal_translation = 0
        // 90°，位移方向与射线方向垂直，translation方向与feature_direction方向垂直， parallel_translation = 0, orthogonal_translation = translation
        // 所以这块的判断即考虑了角度，同时考虑了位移。运动方向要尽可能与投影射线垂直，且位移要足够大
        Eigen::Vector3d orthogonal_translation =
            translation - parallel_translation * feature_direction;

        if (orthogonal_translation.norm() >
            optimization_config.translation_threshold)
            return true;
        else
            return false;
    }

    bool Feature::initializePosition(
        const CamStateServer &cam_states)
    {
        // 存放每个观测以及每个对应相机的pos，注意这块是左右目独立存放，即一帧图像会有两个cam_poses和两个measurements
        std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> cam_poses(0);
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> measurements(0);

        // 1. 准备数据，左右相机的归一化坐标和位姿
        for (auto &m : observations)
        {
            // 1.0 在cam_states中查找观测对应的相机位姿
            auto cam_state_iter = cam_states.find(m.first);
            if (cam_state_iter == cam_states.end())
                continue;

            // 1.1 添加左相机和右相机的归一化坐标
            measurements.push_back(m.second.head<2>());
            measurements.push_back(m.second.tail<2>());

            // 左右相机的 Twc
            Eigen::Isometry3d cam0_pose;
            cam0_pose.linear() =
                quaternionToRotation(cam_state_iter->second.orientation).transpose(); // Rcw -> Rwc
            cam0_pose.translation() = cam_state_iter->second.position;                // twc
            Eigen::Isometry3d cam1_pose;
            cam1_pose = cam0_pose * CAMState::T_cam0_cam1.inverse();

            // 1.2 添加相机位姿
            cam_poses.push_back(cam0_pose);
            cam_poses.push_back(cam1_pose);
        }

        // All camera poses should be modified such that it takes a
        // vector from the first camera frame in the buffer to this
        // camera frame.
        // 2. 转换坐标系，将所有的相机位姿转换为，第一个相机到当前相机的位姿
        Eigen::Isometry3d T_c0_w = cam_poses[0]; // 第一帧位姿
        for (auto &pose : cam_poses)
            pose = pose.inverse() * T_c0_w; // Twci' * Twc0 = Tciw * Twc0 = Tcic0，代表第一个相机到当前相机的位姿

        // 3. 使用首末位姿粗略计算出一个三维点坐标
        Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
        generateInitialGuess( // 传入的是：TcLastc0(第一帧在最后一帧下的位姿), z0(第一帧的观测), zlast(最后一帧的观测), p(第一帧的三维点)
            cam_poses[cam_poses.size() - 1], measurements[0],
            measurements[measurements.size() - 1], initial_position);
        // 弄成逆深度形式，为了避免局部极小值，提高数值稳定性
        Eigen::Vector3d solution(
            initial_position(0) / initial_position(2),
            initial_position(1) / initial_position(2),
            1.0 / initial_position(2));

        // Apply Levenberg-Marquart method to solve for the 3d position.
        double lambda = optimization_config.initial_damping;
        int inner_loop_cntr = 0;
        int outer_loop_cntr = 0;
        bool is_cost_reduced = false;
        double delta_norm = 0;

        // 4. 利用初计算的点计算在各个相机下的误差，作为初始误差
        double total_cost = 0.0;
        for (int i = 0; i < cam_poses.size(); ++i)
        {
            double this_cost = 0.0;
            // 计算投影误差（归一化坐标）
            cost(cam_poses[i], solution, measurements[i], this_cost);
            total_cost += this_cost;
        }

        // Outer loop.
        // 5. LM优化开始， 优化三维点坐标，不优化位姿，比较简单
        do
        {
            // A是  J^t * J  B是 J^t * r
            // 可能有同学疑问自己当初学的时候是 -J^t * r
            // 这个无所谓，因为这里是负的更新就是正的，而这里是正的，所以更新是负的
            // 总之就是有一个是负的，总不能误差越来越大吧
            Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
            Eigen::Vector3d b = Eigen::Vector3d::Zero();

            // 5.1 计算AB矩阵
            for (int i = 0; i < cam_poses.size(); ++i)
            {
                Eigen::Matrix<double, 2, 3> J;
                Eigen::Vector2d r;
                double w;

                // 计算一目相机观测的雅可比与误差
                // J 归一化坐标误差相对于三维点的雅可比
                // r
                // w 权重，同信息矩阵
                jacobian(cam_poses[i], solution, measurements[i], J, r, w);

                // 鲁棒核约束
                if (w == 1)
                {
                    A += J.transpose() * J;
                    b += J.transpose() * r;
                }
                else
                {
                    double w_square = w * w;
                    A += w_square * J.transpose() * J;
                    b += w_square * J.transpose() * r;
                }
            }

            // Inner loop.
            // Solve for the delta that can reduce the total cost.
            // 这层是在同一个雅可比下多次迭代，如果效果不好说明需要调整阻尼因子了，因为线性化不是很好
            // 如果多次一直误差不下降，退出循环重新计算雅可比
            do
            {
                // LM算法中的lambda
                Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
                Eigen::Vector3d delta = (A + damper).ldlt().solve(b);
                // 更新
                Eigen::Vector3d new_solution = solution - delta;
                // 统计本次修改量的大小，如果太小表示接近目标值或者陷入局部极小值，那么就没必要继续了
                delta_norm = delta.norm();

                // 计算更新后的误差
                double new_cost = 0.0;
                for (int i = 0; i < cam_poses.size(); ++i)
                {
                    double this_cost = 0.0;
                    cost(cam_poses[i], new_solution, measurements[i], this_cost);
                    new_cost += this_cost;
                }

                // 如果更新后误差比之前小，说明确实是往好方向发展
                // 我们高斯牛顿的JtJ比较接近真实情况所以减少阻尼，增大步长，delta变大，加快收敛
                if (new_cost < total_cost)
                {
                    is_cost_reduced = true;
                    solution = new_solution;
                    total_cost = new_cost;
                    lambda = lambda / 10 > 1e-10 ? lambda / 10 : 1e-10;
                }
                // 如果不行，那么不要这次迭代的结果
                // 说明高斯牛顿的JtJ不接近二阶的海森矩阵
                // 那么增大阻尼，减小步长，delta变小
                // 并且算法接近一阶的最速下降法
                else
                {
                    is_cost_reduced = false;
                    lambda = lambda * 10 < 1e12 ? lambda * 10 : 1e12;
                }

            } while (inner_loop_cntr++ <
                         optimization_config.inner_loop_max_iteration &&
                     !is_cost_reduced);

            inner_loop_cntr = 0;

            // 直到迭代次数到了或者更新量足够小了
        } while (outer_loop_cntr++ <
                     optimization_config.outer_loop_max_iteration &&
                 delta_norm > optimization_config.estimation_precision);

        // Covert the feature position from inverse depth
        // representation to its 3d coordinate.
        // 取出最后的结果
        Eigen::Vector3d final_position(solution(0) / solution(2),
                                       solution(1) / solution(2), 1.0 / solution(2));

        // Check if the solution is valid. Make sure the feature
        // is in front of every camera frame observing it.
        // 6. 深度验证，把三维点在第一帧相机下的坐标转换回第i帧相机下的坐标，然后判断z是否大于0
        bool is_valid_solution = true;
        for (const auto &pose : cam_poses)
        {
            Eigen::Vector3d position =
                pose.linear() * final_position + pose.translation();
            if (position(2) <= 0)
            {
                is_valid_solution = false;
                std::cout << "失败" << std::endl;
                break;
            }
        }

        // 7. 更新结果
        // Convert the feature position to the world frame.
        position = T_c0_w.linear() * final_position + T_c0_w.translation();

        if (is_valid_solution)
            is_initialized = true;

        return is_valid_solution;
    }

    /**
     * @brief cost 计算重投影误差（归一化坐标）
     * @param T_c0_ci 相对位姿，Tcic0 第一帧相机在第i帧相机下的位姿
     * @param x 第一帧相机下的估计的三维点坐标 的逆深度形式(x/z, y/z, 1/z)
     * @param z ci下的真实观测归一化坐标
     * @param e 误差
     */
    void Feature::cost(
        const Eigen::Isometry3d &T_c0_ci,
        const Eigen::Vector3d &x, const Eigen::Vector2d &z,
        double &e) const
    {
        // Compute hi1, hi2, and hi3 as Equation (37).
        const double &alpha = x(0);
        const double &beta = x(1);
        const double &rho = x(2);

        //             [x/z]
        // h = Rcic0 * [y/z] + 1/z * tcic0 = (Rcic0 * P + tcic0) / z
        //             [ 1 ]
        // =>
        // h1    | R11 R12 R13    alpha / rho       t1    |
        // h2 =  | R21 R22 R23 *  beta  / rho   +   t2    |   *  rho
        // h3    | R31 R32 R33    1 / rho           t3    |
        Eigen::Vector3d h =
            T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) +
            rho * T_c0_ci.translation();
        double &h1 = h(0);
        double &h2 = h(1);
        double &h3 = h(2);

        // 求出在另一个相机ci下的归一化坐标
        Eigen::Vector2d z_hat(h1 / h3, h2 / h3);

        // Compute the residual.
        // 两个归一化坐标算误差
        e = (z_hat - z).squaredNorm();
        return;
    }

    void Feature::jacobian(
        const Eigen::Isometry3d &T_c0_ci,
        const Eigen::Vector3d &x, const Eigen::Vector2d &z,
        Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r,
        double &w) const
    {

        // Compute hi1, hi2, and hi3 as Equation (37).
        const double &alpha = x(0); // x/z
        const double &beta = x(1);  // y/z
        const double &rho = x(2);   // 1/z

        //             [x/z]
        // h = Rcic0 * [y/z] + 1/z * tcic0 = (Rcic0 * P + tcic0) / z
        //             [ 1 ]
        // =>
        // h1    | R11 R12 R13    alpha / rho       t1    |
        // h2 =  | R21 R22 R23 *  beta  / rho   +   t2    |   *  rho
        // h3    | R31 R32 R33    1 / rho           t3    |
        Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) +
                            rho * T_c0_ci.translation();
        double &h1 = h(0);
        double &h2 = h(1);
        double &h3 = h(2);

        // Compute the Jacobian.
        // 首先明确一下误差与三维点的关系
        // 下面的r是误差 r = z_hat - z;  Eigen::Vector2d z_hat(h1 / h3, h2 / h3)
        // 我们要求r对三维点的雅可比，其中z是观测，与三维点坐标无关
        // 因此相当于求z_hat相对于 alpha beta rho的雅可比，首先要求出他们之间的关系
        // 归一化坐标设为x y
        // x = h1/h3 y = h2/h3
        // 因此雅可比矩阵为:
        // | ∂x/∂alpha ∂x/∂beta ∂x/∂rho |
        // | ∂y/∂alpha ∂y/∂beta ∂y/∂rho |
        // 先写出h与alpha beta rho的关系，也就是上面写的
        // h1 = R11 * alpha + R12 * beta + R13 + rho * t1
        // h2 = R21 * alpha + R22 * beta + R23 + rho * t2
        // h3 = R31 * alpha + R32 * beta + R33 + rho * t3
        // 然后求h1 相对于alpha beta rho的导数 再求h2的  再求h3的
        // 链式求导法则
        // ∂x/∂alpha = ∂x/∂h1 * ∂h1/∂alpha + ∂x/∂h3 * ∂h3/∂alpha
        // ∂x/∂h1 = 1/h3       ∂h1/∂alpha = R11
        // ∂x/∂h3 = -h1/h3^2   ∂h3/∂alpha = R31
        // 剩下的就好求了

        //     | R11 R12 t1 |
        // W = | R21 R22 t2 |
        //     | R31 R32 t3 |
        Eigen::Matrix3d W;
        W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
        W.rightCols<1>() = T_c0_ci.translation();

        // h1 / h3 相对于 alpha beta rho的
        J.row(0) = 1 / h3 * W.row(0) - h1 / (h3 * h3) * W.row(2);
        // h1 / h3 相对于 alpha beta rho的
        J.row(1) = 1 / h3 * W.row(1) - h2 / (h3 * h3) * W.row(2);

        // 求取误差
        Eigen::Vector2d z_hat(h1 / h3, h2 / h3);
        r = z_hat - z;

        // Compute the weight based on the residual.
        // 使用鲁棒核函数约束
        double e = r.norm();
        if (e <= optimization_config.huber_epsilon)
            w = 1.0;
        // 如果误差大于huber_epsilon但是没超过他的2倍，2 * huber_epsilon / e ∈ 2 * (1~1/2) 那么会放大权重w>1
        // 如果误差大的离谱，超过他的2倍，缩小他的权重, 2 * huber_epsilon / e ∈ 2 * (1/2 ~ 0) 那么会缩小权重w<1
        else
            w = std::sqrt(2.0 * optimization_config.huber_epsilon / e);

        return;
    }

    /**
     * @brief generateInitialGuess 两帧做一次三角化
     * @param T_c1_c2 两帧间的相对位姿，c1在c2下的位姿
     * @param z1 c1下的观测
     * @param z2 c2下的观测 都是归一化坐标
     * @param p 三维点在c1下的坐标
     */
    void Feature::generateInitialGuess(
        const Eigen::Isometry3d &T_c1_c2, const Eigen::Vector2d &z1,
        const Eigen::Vector2d &z2, Eigen::Vector3d &p) const
    {
        // 列出方程
        // P2 = R21 * P1 + t21  下面省略21， 求P1
        // 两边左乘P2的反对称矩阵
        // P2^ * (R * P1 + t) = 0
        // 其中左右可以除以P2的深度，这样P2就成了z2，且P1可以分成z1（归一化平面） 乘上我们要求的深度d
        // z2^ * (R * z1 * d + t) = 0
        // 令m = R * z1
        // z2^ * (m * d + t) = 0
        // | 0   -1   z2y |    ( | m0 |         )
        // | 1    0  -z2x | *  ( | m1 | * d + t )  =  0
        // |-z2y z2x   0  |    ( | m2 |         )
        // z2^ * m * d = - z2^ * t
        // A * d = b
        // 会发现这三行里面两行是线性相关的，所以只取前两行
        // 构造一个最小二乘问题去求解深度
        Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

        Eigen::Vector2d A(0.0, 0.0);
        A(0) = m(0) - z2(0) * m(2); // 对应第二行
        // 按照上面推导这里应该是负的但是不影响，因为我们下边b(1)也给取负了
        A(1) = m(1) - z2(1) * m(2); // 对应第一行

        Eigen::Vector2d b(0.0, 0.0);
        b(0) = z2(0) * T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
        b(1) = z2(1) * T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

        // Solve for the depth.
        // 解方程得出p1的深度值
        double depth = (A.transpose() * A).inverse() * A.transpose() * b;
        p(0) = z1(0) * depth;
        p(1) = z1(1) * depth;
        p(2) = depth;
        return;
    }

} // namespace msckf_vio

#endif // MSCKF_VIO_FEATURE_H
