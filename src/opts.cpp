#include "volrend/internal/opts.hpp"
#include <cstdio>

namespace volrend {
namespace internal {

void add_common_opts(cxxopts::Options& options) {
    // clang-format off
    options.add_options()
        ("file", "npz file storing octree data", cxxopts::value<std::string>())
        ("draw", "npz drawlist file",
         cxxopts::value<std::string>()->default_value(""))
        ("gpu", "CUDA device id (only if using cuda; defaults to first one)",
             cxxopts::value<int>()->default_value("-1"))
        ("w,width", "image width", cxxopts::value<int>()->default_value("800"))
        ("h,height", "image height", cxxopts::value<int>()->default_value("800"))
        ("fx", "focal length in x direction; -1 = 1111 or default for NDC",
            cxxopts::value<float>()->default_value("-1.0"))
        ("fy", "focal length in y direction; -1 = use fx",
            cxxopts::value<float>()->default_value("-1.0"))
        ("bg", "background brightness 0-1", cxxopts::value<float>()->default_value("1.0"))
        ("s,step_size", "step size epsilon added to computed cube size",
             cxxopts::value<float>()->default_value("1e-4"))
        ("e,stop_thresh", "early stopping threshold (on remaining intensity)",
             cxxopts::value<float>()->default_value("1e-2"))
        ("a,sigma_thresh", "sigma threshold (skip cells with < sigma)",
             cxxopts::value<float>()->default_value("1e-2"))
        ("help", "Print this help message")
        ;
    // clang-format on
}

cxxopts::ParseResult parse_options(cxxopts::Options& options, int argc,
                                   char* argv[]) {
    options.parse_positional({"file"});
    cxxopts::ParseResult args = options.parse(argc, argv);
    if (args.count("help")) {
        printf("%s\n", options.help().c_str());
        std::exit(0);
    }
    return args;
}

RenderOptions render_options_from_args(cxxopts::ParseResult& args) {
    RenderOptions options;
    options.background_brightness = args["bg"].as<float>();
#ifdef VOLREND_CUDA
    if (args.count("grid")) {
        options.show_grid = true;
        options.grid_max_depth = args["grid"].as<int>();
    }
    if (args.count("probe")) {
        options.enable_probe = true;
        auto probe = args["probe"].as<std::vector<float>>();
        if (probe.size() < 3) {
            fprintf(stderr, "ERROR: --probe must be of format 'x,y,z'\n");
        } else {
            for (int i = 0; i < 3; ++i) options.probe[i] = probe[i];
        }
    }
#endif
    options.step_size = args["step_size"].as<float>();
    options.stop_thresh = args["stop_thresh"].as<float>();
    options.sigma_thresh = args["sigma_thresh"].as<float>();
    return options;
}

}  // namespace internal
}  // namespace volrend
