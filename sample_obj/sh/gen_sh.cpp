#include <complex>
#include <string>
#include <fstream>
#include <limits>
#include <iostream>
#include <cstdint>
#include <array>
#include <vector>
using complex = std::complex<double>;
struct Vector3d {
    Vector3d() : data{0.0, 0.0, 0.0} {}
    Vector3d(double x, double y, double z) : data{x, y, z} {}
    double x() const { return data[0]; };
    double y() const { return data[1]; };
    double z() const { return data[2]; };
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    Vector3d operator+(const Vector3d& other) const {
        return Vector3d(data[0] + other[0], data[1] + other[1],
                        data[2] + other[2]);
    }
    Vector3d operator-(const Vector3d& other) const {
        return Vector3d(data[0] - other[0], data[1] - other[1],
                        data[2] - other[2]);
    }
    Vector3d operator*(double s) const {
        return Vector3d(data[0] * s, data[1] * s, data[2] * s);
    }
    Vector3d operator/(double s) const {
        return Vector3d(data[0] / s, data[1] / s, data[2] / s);
    }
    double data[3];
};

double norm(const Vector3d& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
Vector3d normalized(const Vector3d& v) { return v / norm(v); }

struct Vertex {
    Vector3d p, c;
};

struct Triangle {
    unsigned int v[3];
};

struct Mesh {
    std::vector<Vertex> vert;
    std::vector<Triangle> faces;

    void write_obj(const std::string& path) {
        std::ofstream ofs(path);
        for (size_t i = 0; i < vert.size(); ++i) {
            ofs << "v";
            const auto& v = vert[i];
            for (int j = 0; j < 3; ++j) {
                ofs << " " << v.p[j];
            }
            for (int j = 0; j < 3; ++j) {
                ofs << " " << v.c[j];
            }
            ofs << "\n";
        }
        for (size_t i = 0; i < faces.size(); ++i) {
            ofs << "f";
            const auto& f = faces[i];
            for (int j = 0; j < 3; ++j) {
                ofs << " " << f.v[j] + 1;
            }
            ofs << "\n";
        }
    }
};

namespace {
// assuming that @d is unit. This is not verified for efficiency.
double HardcodedSH00(const Vector3d& d) {
    // 0.5 * sqrt(1/pi)
    return 0.282095;
}

double HardcodedSH1n1(const Vector3d& d) {
    // -sqrt(3/(4pi)) * y
    return -0.488603 * d.y();
}

double HardcodedSH10(const Vector3d& d) {
    // sqrt(3/(4pi)) * z
    return 0.488603 * d.z();
}

double HardcodedSH1p1(const Vector3d& d) {
    // -sqrt(3/(4pi)) * x
    return -0.488603 * d.x();
}

double HardcodedSH2n2(const Vector3d& d) {
    // 0.5 * sqrt(15/pi) * x * y
    return 1.092548 * d.x() * d.y();
}

double HardcodedSH2n1(const Vector3d& d) {
    // -0.5 * sqrt(15/pi) * y * z
    return -1.092548 * d.y() * d.z();
}

double HardcodedSH20(const Vector3d& d) {
    // 0.25 * sqrt(5/pi) * (-x^2-y^2+2z^2)
    return 0.315392 * (-d.x() * d.x() - d.y() * d.y() + 2.0 * d.z() * d.z());
}

double HardcodedSH2p1(const Vector3d& d) {
    // -0.5 * sqrt(15/pi) * x * z
    return -1.092548 * d.x() * d.z();
}

double HardcodedSH2p2(const Vector3d& d) {
    // 0.25 * sqrt(15/pi) * (x^2 - y^2)
    return 0.546274 * (d.x() * d.x() - d.y() * d.y());
}

double HardcodedSH3n3(const Vector3d& d) {
    // -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
    return -0.590044 * d.y() * (3.0 * d.x() * d.x() - d.y() * d.y());
}

double HardcodedSH3n2(const Vector3d& d) {
    // 0.5 * sqrt(105/pi) * x * y * z
    return 2.890611 * d.x() * d.y() * d.z();
}

double HardcodedSH3n1(const Vector3d& d) {
    // -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
    return -0.457046 * d.y() *
           (4.0 * d.z() * d.z() - d.x() * d.x() - d.y() * d.y());
}

double HardcodedSH30(const Vector3d& d) {
    // 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
    return 0.373176 * d.z() *
           (2.0 * d.z() * d.z() - 3.0 * d.x() * d.x() - 3.0 * d.y() * d.y());
}

double HardcodedSH3p1(const Vector3d& d) {
    // -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
    return -0.457046 * d.x() *
           (4.0 * d.z() * d.z() - d.x() * d.x() - d.y() * d.y());
}

double HardcodedSH3p2(const Vector3d& d) {
    // 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
    return 1.445306 * d.z() * (d.x() * d.x() - d.y() * d.y());
}

double HardcodedSH3p3(const Vector3d& d) {
    // -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
    return -0.590044 * d.x() * (d.x() * d.x() - 3.0 * d.y() * d.y());
}

double HardcodedSH4n4(const Vector3d& d) {
    // 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
    return 2.503343 * d.x() * d.y() * (d.x() * d.x() - d.y() * d.y());
}

double HardcodedSH4n3(const Vector3d& d) {
    // -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
    return -1.770131 * d.y() * d.z() * (3.0 * d.x() * d.x() - d.y() * d.y());
}

double HardcodedSH4n2(const Vector3d& d) {
    // 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
    return 0.946175 * d.x() * d.y() * (7.0 * d.z() * d.z() - 1.0);
}

double HardcodedSH4n1(const Vector3d& d) {
    // -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
    return -0.669047 * d.y() * d.z() * (7.0 * d.z() * d.z() - 3.0);
}

double HardcodedSH40(const Vector3d& d) {
    // 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
    double z2 = d.z() * d.z();
    return 0.105786 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0);
}

double HardcodedSH4p1(const Vector3d& d) {
    // -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
    return -0.669047 * d.x() * d.z() * (7.0 * d.z() * d.z() - 3.0);
}

double HardcodedSH4p2(const Vector3d& d) {
    // 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
    return 0.473087 * (d.x() * d.x() - d.y() * d.y()) *
           (7.0 * d.z() * d.z() - 1.0);
}

double HardcodedSH4p3(const Vector3d& d) {
    // -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
    return -1.770131 * d.x() * d.z() * (d.x() * d.x() - 3.0 * d.y() * d.y());
}

double HardcodedSH4p4(const Vector3d& d) {
    // 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
    double x2 = d.x() * d.x();
    double y2 = d.y() * d.y();
    return 0.625836 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2));
}

double eval_sh(int l, int m, const Vector3d& dir) {
    switch (l) {
        case 0:
            return HardcodedSH00(dir);
        case 1:
            switch (m) {
                case -1:
                    return HardcodedSH1n1(dir);
                case 0:
                    return HardcodedSH10(dir);
                case 1:
                    return HardcodedSH1p1(dir);
            }
        case 2:
            switch (m) {
                case -2:
                    return HardcodedSH2n2(dir);
                case -1:
                    return HardcodedSH2n1(dir);
                case 0:
                    return HardcodedSH20(dir);
                case 1:
                    return HardcodedSH2p1(dir);
                case 2:
                    return HardcodedSH2p2(dir);
            }
        case 3:
            switch (m) {
                case -3:
                    return HardcodedSH3n3(dir);
                case -2:
                    return HardcodedSH3n2(dir);
                case -1:
                    return HardcodedSH3n1(dir);
                case 0:
                    return HardcodedSH30(dir);
                case 1:
                    return HardcodedSH3p1(dir);
                case 2:
                    return HardcodedSH3p2(dir);
                case 3:
                    return HardcodedSH3p3(dir);
            }
        case 4:
            switch (m) {
                case -4:
                    return HardcodedSH4n4(dir);
                case -3:
                    return HardcodedSH4n3(dir);
                case -2:
                    return HardcodedSH4n2(dir);
                case -1:
                    return HardcodedSH4n1(dir);
                case 0:
                    return HardcodedSH40(dir);
                case 1:
                    return HardcodedSH4p1(dir);
                case 2:
                    return HardcodedSH4p2(dir);
                case 3:
                    return HardcodedSH4p3(dir);
                case 4:
                    return HardcodedSH4p4(dir);
            }
    }
    return 0.0;
}

Mesh gen_sh_mesh(int sh_l, int sh_m, int rings = 100, int sectors = 200,
                 const Vector3d& color_p = Vector3d{0.2, 0.2, 1.0},
                 const Vector3d& color_n = Vector3d{1.0, 1.0, 0.0}) {
    Mesh m;
    m.vert.resize(rings * sectors);
    m.faces.resize((rings - 1) * sectors * 2);
    const double R = M_PI / (double)(rings - 1);
    const double S = 2 * M_PI / (double)sectors;
    Vertex* vptr = m.vert.data();
    const double EPS = 1e-6;
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < sectors; s++) {
            const double z = sin(-0.5f * M_PI + r * R);
            const double x = cos(s * S) * sin(r * R);
            const double y = sin(s * S) * sin(r * R);
            // SH Eval
            Vector3d dir(x, y, z);
            double t = eval_sh(sh_l, sh_m, dir);
            // double tpx =
            //     eval_sh(sh_l, sh_m, normalized(dir + Vector3d(EPS, 0.0,
            //     0.0)));
            // double tpy =
            //     eval_sh(sh_l, sh_m, normalized(dir + Vector3d(0.0, EPS,
            //     0.0)));
            // double tpz =
            //     eval_sh(sh_l, sh_m, normalized(dir + Vector3d(0.0, 0.0,
            //     EPS)));
            // double tnx =
            //     eval_sh(sh_l, sh_m, normalized(dir - Vector3d(EPS, 0.0,
            //     0.0)));
            // double tny =
            //     eval_sh(sh_l, sh_m, normalized(dir - Vector3d(0.0, EPS,
            //     0.0)));
            // double tnz =
            //     eval_sh(sh_l, sh_m, normalized(dir - Vector3d(0.0, 0.0,
            //     EPS)));

            vptr->p = dir * -std::fabs(t);
            vptr->c = t >= 0.0 ? color_p : color_n;
            // Advance
            ++vptr;
        }
    }
    Triangle* ptr = m.faces.data();
    for (int r = 0; r < rings - 1; r++) {
        const int nx_r = r + 1;
        for (int s = 0; s < sectors; s++) {
            const int nx_s = (s + 1) % sectors;
            ptr->v[0] = r * sectors + nx_s;
            ptr->v[1] = r * sectors + s;
            ptr->v[2] = nx_r * sectors + s;
            ++ptr;
            ptr->v[0] = nx_r * sectors + s;
            ptr->v[1] = nx_r * sectors + nx_s;
            ptr->v[2] = r * sectors + nx_s;
            ++ptr;
        }
    }
    return m;
}
}  // namespace

int32_t main(int32_t argc, char** argv) {
    if (argc < 2) {
        std::cout << "Input: max degree l <= 4\n";
        return 0;
    }
    int max_l = std::atoi(argv[1]);
    if (max_l > 4) {
        std::cout << "max_l set to 4 since that's the max supported\n";
        max_l = 4;
    }
    for (int l = 0; l <= max_l; ++l) {
        for (int m = -l; m <= l; ++m) {
            std::string name = "sh_" + std::to_string(l) + "_";
            if (m < 0)
                name.push_back('n');
            else
                name.push_back('p');
            name.append(std::to_string(std::abs(m)));
            name.append(".obj");
            gen_sh_mesh(l, m).write_obj(name);
            name.append(".offs");

            std::ofstream ofs(name);
            ofs << m * 2.5 << " 0 " << (-l + max_l * 0.5) * 1.8 << "\n";
        }
    }
}
