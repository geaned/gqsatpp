#!/bin/bash

# Copyright 2019-2020 Nvidia Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage: ./evaluate.sh -v VAL_PROBLEMS_PATHS -d MODEL_DIR -c MODEL_CHECKPOINT

while getopts v:d:c: flag
do
  case "${flag}" in
    v) val=${OPTARG};;
    d) dir=${OPTARG};;
    c) chkp=${OPTARG};;
  esac
done

python3 evaluate.py \
  --env-name sat-v0 \
  --core-steps -1 \
  --eps-final 0.0 \
  --eval-time-limit 100000000000000 \
  --no_restarts \
  --test_time_max_decisions_allowed 500 \
  --eval-problems-paths $val \
  --model-dir $dir \
  --model-checkpoint $chkp
