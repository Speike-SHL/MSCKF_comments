#include <Eigen/Dense>
#include <iostream>

using namespace std;

int main() {
    Eigen::MatrixXd A;
    A << 1, 2, 3, 4;
    cout << A << endl;
    Eigen::MatrixXd B;
    B << 1, 2, 3, 4;
    cout << B << endl;
    cout << A * B << endl;
}
