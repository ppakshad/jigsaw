#!/usr/bin/env bash
python3 pipeline.py -D \
  -R FEDZ-25 \
  --confidence 25 \
  --donor-depth 5 \
  --organ-depth 200 \
  --harvest \
  --preload
