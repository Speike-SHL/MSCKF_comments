#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <boost/stacktrace.hpp>
#include <boost/current_function.hpp>
#include <stdexcept>

using namespace std;

#define REDCOLOR(STRING) "\033[1;31m" << STRING << "\033[0m"
#define REDCOUT(STRING) cout << "\033[31m" << STRING << "\033[m"

/* class SafeMatrixXd
{
private:
     Eigen::MatrixXd mat;

     // 帮助函数：检查尺寸是否匹配并输出文件名、行号和调用栈
     void checkSize(const Eigen::MatrixXd &other, const char *functionName, const char *file, int line) const
     {
          if (mat.cols() != other.rows())
          {
               std::cerr << "Matrix dimension mismatch in function " << REDCOLOR(functionName) << "\n";
               std::cerr << "File: " << REDCOLOR(file) << ", Line: " << REDCOLOR(line) << "\n";
               std::cerr << "Stack trace:\n"
                         << boost::stacktrace::stacktrace() << "\n";
               throw std::invalid_argument("Matrix dimension mismatch for multiplication.");
          }
     }

     // 帮助函数：检查块索引是否越界并输出文件名、行号和调用栈
     void checkBlockIndex(int startRow, int startCol, int blockRows, int blockCols, const char *functionName, const char *file, int line) const
     {
          if (startRow + blockRows > mat.rows() || startCol + blockCols > mat.cols())
          {
               std::cerr << "Block index out of bounds in function " << functionName << "\n";
               std::cerr << "File: " << file << ", Line: " << line << "\n";
               std::cerr << "Stack trace:\n"
                         << boost::stacktrace::stacktrace() << "\n";
               throw std::out_of_range("Block index out of matrix bounds.");
          }
     }

public:
     // 构造函数
     SafeMatrixXd(size_t rows, size_t cols) : mat(rows, cols) {}
     SafeMatrixXd(const Eigen::MatrixXd &other) : mat(other) {}

     // 访问底层矩阵
     const Eigen::MatrixXd &matrix() const { return mat; }

     // 重载各种运算符
     SafeMatrixXd operator*(const SafeMatrixXd &other) const
     {
          checkSize(other.mat, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__);
          return SafeMatrixXd(mat * other.mat);
     }

     SafeMatrixXd &operator=(const SafeMatrixXd &other)
     {
          mat = other.mat;
          return *this;
     }

     SafeMatrixXd &operator=(const Eigen::Block<Eigen::MatrixXd> &block)
     {
          mat = block;
          return *this;
     }

     friend std::ostream &operator<<(std::ostream &os, const SafeMatrixXd &m) { return os << m.mat; }

     // 重载各种方法
     Eigen::Block<Eigen::MatrixXd> block(size_t startRow, size_t startCol, size_t blockRows, size_t blockCols)
     {
          checkBlockIndex(startRow, startCol, blockRows, blockCols, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__);
          return mat.block(startRow, startCol, blockRows, blockCols);
     }

     // topLeftCorner
     Eigen::Block<Eigen::MatrixXd> topLeftCorner(size_t blockRows, size_t blockCols)
     {
          checkBlockIndex(0, 0, blockRows, blockCols, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__);
          return mat.topLeftCorner(blockRows, blockCols);
     }

     size_t rows() const { return mat.rows(); }
     size_t cols() const { return mat.cols(); }

     static SafeMatrixXd Random(size_t rows, size_t cols)
     {
          return SafeMatrixXd(Eigen::MatrixXd::Random(rows, cols));
     }
     static SafeMatrixXd Identity(size_t rows, size_t cols)
     {
          return SafeMatrixXd(Eigen::MatrixXd::Identity(rows, cols));
     }
}; */

void checkSize(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const char *functionName, const char *file, int line)
{
     if (A.cols() != B.rows())
     {
          std::cerr << "Matrix dimension mismatch in function " << REDCOLOR(functionName) << "\n";
          std::cerr << "File: " << REDCOLOR(file) << ", Line: " << REDCOLOR(line) << ", ";
          std::cerr << "A * B : " << REDCOLOR(A.cols() << "x" << A.rows() << " * " << B.cols() << "x" << B.rows()) << "\n";
          std::cerr << "Stack trace:\n"
                    << boost::stacktrace::stacktrace() << "\n";
          throw std::invalid_argument("Matrix dimension mismatch for multiplication.");
     }
}

// 帮助函数：检查块索引是否越界并输出文件名、行号和调用栈
void checkBlockIndex(const Eigen::MatrixXd &A, int startRow, int startCol, int blockRows, int blockCols, const char *functionName, const char *file, int line)
{
     if (startRow + blockRows > A.rows() || startCol + blockCols > A.cols())
     {
          std::cerr << "Block index out of bounds in function " << REDCOLOR(functionName) << "\n";
          std::cerr << "File: " << REDCOLOR(file) << ", Line: " << REDCOLOR(line) << ", ";
          std::cerr << "A size : " << REDCOLOR(A.rows() << "x" << A.cols() << " Index: " << startRow << ", " << startCol << ", " << blockRows << ", " << blockCols) << "\n";
          std::cerr
              << "Stack trace:\n"
              << boost::stacktrace::stacktrace() << "\n";
          throw std::out_of_range("Block index out of matrix bounds.");
     }
}

int main()
{
/*      SafeMatrixXd tmp = SafeMatrixXd::Identity(9, 9);
     cout << tmp << endl
          << endl;
     SafeMatrixXd tmp2 = SafeMatrixXd::Random(8, 8);
     cout << tmp2 << endl
          << endl;
     tmp.topLeftCorner(tmp2.rows(), tmp2.cols()) = tmp2.matrix();
     cout << tmp << endl
          << endl;

     cout << endl;

     cout << tmp.matrix().array().isNaN().any() << endl;
     cout << tmp.matrix().diagonal().transpose() << endl; */

     // Eigen::MatrixXd tmp = Eigen::MatrixXd::Random(8, 8);
     // state_server.robot_state.setX_i(tmp);

     // int dimX_i = state_server.robot_state.dimX_i();
     // state_server.robot_state.X_i.conservativeResize(30, 30);
     // state_server.robot_state.X_i.conservativeResize(dimX_i, dimX_i);

     cout << "---------------------------" << endl;
     Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
     cout << A << endl
          << endl;
     Eigen::MatrixXd B = Eigen::MatrixXd::Random(4, 4);
     cout << B << endl
          << endl;
     Eigen::VectorXd C = Eigen::VectorXd::Random(4);
     cout << C << endl
          << endl;

     // checkSize(A, B, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__);
     // checkBlockIndex(A, 0, 0, 4, 3, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__);
     checkBlockIndex(C, 0, 0, 5, C.cols(), BOOST_CURRENT_FUNCTION, __FILE__, __LINE__);
}
