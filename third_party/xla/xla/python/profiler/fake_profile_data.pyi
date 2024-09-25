# Copyright 2024 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for generating fake program execution data for testing."""

from xla.python.profiler import profile_data


def from_text_proto(text_proto: str) -> profile_data.ProfileData:
  """Creates a ProfileData from a text proto."""
  ...


def text_proto_to_serialized_xspace(text_proto: str) -> bytes:
  """Converts a text proto to a serialized XSpace."""
  ...
