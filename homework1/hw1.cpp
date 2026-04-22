#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
    float a, b, c;
    std::cin >> a >> b >> c;

    std::cout << std::fixed << std::setprecision(6);

    const double epsilon = 1e-9;

    if (std::fabs(a) < epsilon)
    {
        if (std::fabs(b) < epsilon)
        {
            if (std::fabs(c) < epsilon)
            {
                std::cout << "any\n";
            }
            else
            {
                std::cout << "incorrect\n";
            }
        }
        else
        {
            std::cout << -c / b << '\n';
        }
    }
    else
    {
        double d = b * b - 4 * a * c;
        if (d < -epsilon)
        {
            std::cout << "imaginary\n";
        }
        else if (d > epsilon)
        {
            double x1 = (-b + std::sqrt(d)) / (2 * a);
            double x2 = (-b - std::sqrt(d)) / (2 * a);
            std::cout << x1 << ' ' << x2 << '\n';
        }
        else
        {
            std::cout << -b / (2 * a) << '\n';
        }
    }

    return 0;
}
