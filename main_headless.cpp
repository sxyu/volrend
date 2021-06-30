#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

#include "volrend/internal/auto_filesystem.hpp"

#include "volrend/common.hpp"
#include "volrend/n3tree.hpp"

#include "volrend/internal/opts.hpp"

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/renderer_kernel.hpp"
#include "volrend/internal/imwrite.hpp"

namespace {
std::string path_basename(const std::string &str) {
    for (size_t i = str.size() - 1; ~i; --i) {
        const char c = str[i];
        if (c == '/' || c == '\\') {
            return str.substr(i);
        }
    }
    return str;
}

std::string remove_ext(const std::string &str) {
    for (size_t i = str.size() - 1; ~i; --i) {
        const char c = str[i];
        if (c == '.') {
            return str.substr(0, i);
        }
    }
    return str;
}

int read_transform_matrices(const std::string &path,
                            std::vector<glm::mat4x3> &out) {
    std::ifstream ifs(path);
    int cnt = 0;
    if (!ifs) {
        fprintf(stderr, "ERROR: '%s' does not exist\n", path.c_str());
        std::exit(1);
    }
    while (ifs) {
        glm::mat4x3 tmp;
        float garb;
        // Recall GL is column major
        ifs >> tmp[0][0] >> tmp[1][0] >> tmp[2][0] >> tmp[3][0];
        if (!ifs) break;
        ifs >> tmp[0][1] >> tmp[1][1] >> tmp[2][1] >> tmp[3][1];
        ifs >> tmp[0][2] >> tmp[1][2] >> tmp[2][2] >> tmp[3][2];
        if (ifs) {
            ifs >> garb >> garb >> garb >> garb;
        }
        ++cnt;
        out.push_back(std::move(tmp));
    }
    return cnt;
}

void read_intrins(const std::string &path, float &fx, float &fy) {
    std::ifstream ifs(path);
    if (!ifs) {
        fprintf(stderr, "ERROR: intrin '%s' does not exist\n", path.c_str());
        std::exit(1);
    }
    float _;  // garbage
    ifs >> fx >> _ >> _ >> _;
    ifs >> _ >> fy;
}
}  // namespace

int main(int argc, char *argv[]) {
    using namespace volrend;
    cxxopts::Options cxxoptions(
        "volrend_headless",
        "Headless PlenOctree volume rendering (c) PlenOctree authors 2021");
    internal::add_common_opts(cxxoptions);

    // clang-format off
    cxxoptions.add_options()
        ("o,write_images", "output directory of images; "
         "if empty, DOES NOT save (for timing only)",
                cxxopts::value<std::string>()->default_value(""))
        ("i,intrin", "intrinsics matrix 4x4; if set, overrides the fx/fy",
                cxxopts::value<std::string>()->default_value(""))
        ("r,reverse_yz", "use OpenCV camera space convention instead of NeRF",
                cxxopts::value<bool>())
        ("scale", "scaling to apply to image",
                cxxopts::value<float>()->default_value("1.0"))
        ("max_imgs", "max images to render, default no limit",
                cxxopts::value<int>()->default_value("0"))
        ;
    // clang-format on

    cxxoptions.allow_unrecognised_options();

    // Pass a list of camera pose *.txt files after npz file
    // each file should have 4x4 c2w pose matrix
    cxxoptions.positional_help("npz_file [c2w_txt_4x4...]");

    cxxopts::ParseResult args = internal::parse_options(cxxoptions, argc, argv);

    const int device_id = args["gpu"].as<int>();
    if (~device_id) {
        cuda(SetDevice(device_id));
    }

    // Load all transform matrices
    std::vector<glm::mat4x3> trans;
    std::vector<std::string> basenames;
    for (auto path : args.unmatched()) {
        int cnt = read_transform_matrices(path, trans);
        std::string fname = remove_ext(path_basename(path));
        if (cnt == 1) {
            basenames.push_back(fname);
        } else {
            for (int i = 0; i < cnt; ++i) {
                std::string tmp = std::to_string(i);
                while (tmp.size() < 6) tmp = "0" + tmp;
                basenames.push_back(fname + "_" + tmp);
            }
        }
    }

    if (args["reverse_yz"].as<bool>()) {
        puts("INFO: Use OpenCV camera convention\n");
        // clang-format off
        glm::mat4x4 cam_trans(1, 0, 0, 0,
                              0, -1, 0, 0,
                              0, 0, -1, 0,
                              0, 0, 0, 1);
        // clang-format on
        for (auto &transform : trans) {
            transform = transform * cam_trans;
        }
    } else {
        puts("INFO: Use NeRF camera convention\n");
    }

    if (trans.size() == 0) {
        fputs("WARNING: No camera poses specified, quitting\n", stderr);
        return 1;
    }
    std::string out_dir = args["write_images"].as<std::string>();

    N3Tree tree(args["file"].as<std::string>());

    int width = args["width"].as<int>(), height = args["height"].as<int>();
    float fx = args["fx"].as<float>();
    if (fx < 0) fx = 1111.11f;
    float fy = args["fy"].as<float>();
    if (fy < 0) fy = fx;

    {
        // Load intrin matrix
        std::string intrin_path = args["intrin"].as<std::string>();
        if (intrin_path.size()) {
            read_intrins(intrin_path, fx, fy);
        }
    }

    {
        float scale = args["scale"].as<float>();
        if (scale != 1.f) {
            int owidth = width, oheight = height;
            width *= scale;
            height *= scale;
            fx *= (float)width / owidth;
            fy *= (float)height / oheight;
        }
    }

    {
        int max_imgs = args["max_imgs"].as<int>();
        if (max_imgs > 0 && trans.size() > (size_t)max_imgs) {
            trans.resize(max_imgs);
            basenames.resize(max_imgs);
        }
    }

    Camera camera(width, height, fx, fy);
    cudaArray_t array;
    cudaStream_t stream;

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    std::vector<uint8_t> buf;
    if (out_dir.size()) {
        std::filesystem::create_directories(out_dir);
        buf.resize(4 * width * height);
    }

    cuda(MallocArray(&array, &channelDesc, width, height));
    cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));
    cudaArray_t depth_arr = nullptr;  // Not using depth buffer

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i = 0; i < trans.size(); ++i) {
        camera.transform = trans[i];
        camera._update(false);

        RenderOptions options = internal::render_options_from_args(args);

        launch_renderer(tree, camera, options, array, depth_arr, stream, true);

        if (out_dir.size()) {
            cuda(Memcpy2DFromArrayAsync(buf.data(), 4 * width, array, 0, 0,
                                        4 * width, height,
                                        cudaMemcpyDeviceToHost, stream));
            std::string fpath = out_dir + "/" + basenames[i] + ".png";
            internal::write_png_file(fpath, buf.data(), width, height);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds = milliseconds / trans.size();

    printf("%.10f ms per frame\n", milliseconds);
    printf("%.10f fps\n", 1000.f / milliseconds);

    cuda(FreeArray(array));
    cuda(StreamDestroy(stream));
}
