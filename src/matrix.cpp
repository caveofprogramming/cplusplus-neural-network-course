#include "matrix.h"

namespace cave
{
    std::ostream& operator<<(std::ostream &out, cave::Matrix m)
    {
        for(int row = 0; row < m._rows; ++row)
        {
            for(int col = 0; col < m._cols; ++col)
            {
                out << m._v[row * m._cols + col];
            }

            out << "\n";
        }
        return out;
    }
}