/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // IWYU pragma: keep. For automatic conversion of std::string to Python string.

#include <memory>
#include <string>

#include "xla/python/profiler/profile_data.h"
#include "tsl/platform/protobuf.h"

namespace {

NB_MODULE(fake_profile_data, m) {
  m.def("from_text_proto", [](const std::string& text_proto) {
    auto xspace = std::make_shared<tensorflow::profiler::XSpace>();
    tsl::protobuf::TextFormat::ParseFromString(text_proto, xspace.get());
    return tensorflow::profiler::python::ProfileData(xspace);
  });
  m.def("text_proto_to_serialized_xspace", [](const std::string& text_proto) {
    tensorflow::profiler::XSpace xspace;
    tsl::protobuf::TextFormat::ParseFromString(text_proto, &xspace);
    const auto serialized = xspace.SerializeAsString();
    return nanobind::bytes(serialized.data(), serialized.size());
  });
}

}  // namespace
