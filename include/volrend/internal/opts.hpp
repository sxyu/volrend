#pragma once
// Command line option parsing

#include <cxxopts.hpp>
#include "volrend/render_options.hpp"

namespace volrend {
namespace internal {

void add_common_opts(cxxopts::Options& options);

cxxopts::ParseResult parse_options(cxxopts::Options& options, int argc,
                                   char* argv[]);

RenderOptions render_options_from_args(cxxopts::ParseResult& args);

}  // namespace internal
}  // namespace volrend
