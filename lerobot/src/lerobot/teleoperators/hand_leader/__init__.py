#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

# Import configs (these don't depend on MediaPipe)
from .config_hand_leader import HandLeaderConfig, HandLeaderIPCConfig

# Import IPC version (no MediaPipe dependency in main process)
from .hand_leader_ipc import HandLeaderIPC

# Note: HandLeader (old version) not imported to avoid MediaPipe dependency in main process
# It runs MediaPipe directly, while HandLeaderIPC uses two-process architecture

__all__ = ["HandLeaderConfig", "HandLeaderIPCConfig", "HandLeaderIPC"]