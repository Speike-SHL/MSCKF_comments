#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iomanip>
#include <chrono>

using namespace std;

int main()
{
     Eigen::Matrix<double, 9, 9> tmp = Eigen::Matrix<double, 9, 9>::Identity();
     cout << tmp << endl << endl;
     Eigen::MatrixXd tmp2 = Eigen::MatrixXd::Random(8, 8);
     cout << tmp2 << endl << endl;
     tmp.topLeftCorner(tmp2.rows(), tmp2.cols()) = tmp2;
     cout << tmp << endl << endl;

     cout << endl;

     cout << tmp.array().isNaN().any() << endl;
     cout << tmp.diagonal().transpose() << endl;

     // Eigen::MatrixXd tmp = Eigen::MatrixXd::Random(8, 8);
     // state_server.robot_state.setX_i(tmp);

     // int dimX_i = state_server.robot_state.dimX_i();
     // state_server.robot_state.X_i.conservativeResize(30, 30);
     // state_server.robot_state.X_i.conservativeResize(dimX_i, dimX_i);
}
