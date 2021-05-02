#include "volrend/common.hpp"
#include "volrend/internal/data_spec.hpp"

namespace volrend {
namespace internal {
namespace {

template <typename scalar_t>
VOLREND_COMMON_FUNCTION void maybe_precalc_basis(
    const TreeSpec& VOLREND_RESTRICT tree, const scalar_t* VOLREND_RESTRICT dir,
    scalar_t* VOLREND_RESTRICT out) {
    const int basis_dim = tree.data_format.basis_dim;
    switch (tree.data_format.format) {
        case DataFormat::ASG: {
            // UNTESTED ASG
            const scalar_t* ptr = tree.extra;
            for (int i = 0; i < basis_dim; ++i) {
                const scalar_t *ptr_mu_x = ptr + 2, *ptr_mu_y = ptr + 5,
                               *ptr_mu_z = ptr + 8;
                scalar_t S = _dot3(dir, ptr_mu_z);
                scalar_t dot_x = _dot3(dir, ptr_mu_x);
                scalar_t dot_y = _dot3(dir, ptr_mu_y);
                out[i] =
                    S * expf(-ptr[0] * dot_x * dot_x - ptr[1] * dot_y * dot_y) /
                    basis_dim;
                ptr += 11;
            }
        }  // ASG
        break;
        case DataFormat::SG: {
            const scalar_t* ptr = tree.extra;
            for (int i = 0; i < basis_dim; ++i) {
                out[i] = expf(ptr[0] * (_dot3(dir, ptr + 1) - 1.f)) / basis_dim;
                ptr += 4;
            }
        }  // SG
        break;
        case DataFormat::SH: {
            // SH Coefficients from
            // https://github.com/google/spherical-harmonics
            out[0] = 0.28209479177387814;
            const scalar_t x = dir[0], y = dir[1], z = dir[2];
            const scalar_t xx = x * x, yy = y * y, zz = z * z;
            const scalar_t xy = x * y, yz = y * z, xz = x * z;
            switch (basis_dim) {
                case 25:
                    out[16] = 2.5033429417967046 * xy * (xx - yy);
                    out[17] = -1.7701307697799304 * yz * (3 * xx - yy);
                    out[18] = 0.9461746957575601 * xy * (7 * zz - 1.f);
                    out[19] = -0.6690465435572892 * yz * (7 * zz - 3.f);
                    out[20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3);
                    out[21] = -0.6690465435572892 * xz * (7 * zz - 3);
                    out[22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1.f);
                    out[23] = -1.7701307697799304 * xz * (xx - 3 * yy);
                    out[24] = 0.6258357354491761 *
                              (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
                    [[fallthrough]];
                case 16:
                    out[9] = -0.5900435899266435 * y * (3 * xx - yy);
                    out[10] = 2.890611442640554 * xy * z;
                    out[11] = -0.4570457994644658 * y * (4 * zz - xx - yy);
                    out[12] =
                        0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy);
                    out[13] = -0.4570457994644658 * x * (4 * zz - xx - yy);
                    out[14] = 1.445305721320277 * z * (xx - yy);
                    out[15] = -0.5900435899266435 * x * (xx - 3 * yy);
                    [[fallthrough]];
                case 9:
                    out[4] = 1.0925484305920792 * xy;
                    out[5] = -1.0925484305920792 * yz;
                    out[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
                    out[7] = -1.0925484305920792 * xz;
                    out[8] = 0.5462742152960396 * (xx - yy);
                    [[fallthrough]];
                case 4:
                    out[1] = -0.4886025119029199 * y;
                    out[2] = 0.4886025119029199 * z;
                    out[3] = -0.4886025119029199 * x;
            }
        }  // SH
        break;

        default:
            // Do nothing
            break;
    }  // switch
}

}  // namespace
}  // namespace internal
}  // namespace volrend
