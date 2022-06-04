#include <sstream>

#include "vector.h"
#include "matrixfunctions.h"

namespace cave
{
    void operator+=(Matrix &m, Vector &v)
    {
        if (m.rows() != v.rows())
        {
            std::stringstream ss;
            ss << "Cannot add this matrix and vector: incompatible sizes.\n";
            ss << "Matrix: " << m.str() << "\n";
            ss << "Vector: " << v.str() << "\n";
            throw std::logic_error(ss.str());
        }

        cave::MatrixFunctions::modify(m, [&](int row, double value){
            return value + v.get(row);
        });
    }

}