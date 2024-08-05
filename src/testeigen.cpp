#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iomanip>
#include <chrono>
#include "msckf_vio/imu_state.h"
#include "msckf_vio/modern_robotics.h"

using namespace std;

int main()
{
     msckf_vio::RobotState robotstate;
     auto X_i = robotstate.getX_i();
     cout << X_i << endl << endl;

     X_i = Eigen::MatrixXd::Random(X_i.rows() - 2, X_i.cols() - 2);
     cout << X_i << endl << endl;
     robotstate.setX_i(X_i);

     X_i = Eigen::MatrixXd::Random(X_i.rows() + 5, X_i.cols() + 5);
     cout << X_i << endl << endl;
     robotstate.setX_i(X_i);
}
