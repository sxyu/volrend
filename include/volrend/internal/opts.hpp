#pragma once
#include <cxxopts.hpp>
#include "volrend/render_options.hpp"

namespace volrend {
namespace internal {
namespace {

void add_common_opts(cxxopts::Options& options) {
    // clang-format off
    options.add_options()
        ("file", "npz file storing octree data", cxxopts::value<std::string>())
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
        ("show_grid", "show grid", cxxopts::value<bool>())
        ("help", "Print this help message")
        ;
    // clang-format on
}

cxxopts::ParseResult parse_options(cxxopts::Options& options, int argc,
                                   char* argv[]) {
    options.parse_positional({"file"});
    cxxopts::ParseResult args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }
    return args;
}

RenderOptions render_options_from_args(cxxopts::ParseResult& args) {
    RenderOptions options;
    options.background_brightness = args["bg"].as<float>();
    options.show_grid = args["show_grid"].as<bool>();
    options.step_size = args["step_size"].as<float>();
    options.stop_thresh = args["stop_thresh"].as<float>();
    options.sigma_thresh = args["sigma_thresh"].as<float>();
    return options;
}

}  // namespace
}  // namespace internal
}  // namespace volrend
