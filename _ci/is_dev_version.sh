#!/bin/bash

version=$(python -c "from setuptools_scm import get_version; print(get_version())" 2>/dev/null || cat version.txt 2>/dev/null || echo "")

if [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+$ ]]; then
  echo "Valid dev version."
  exit 0
else
  echo "Invalid dev version! Version must have .dev[xx] suffix."
  exit 1
fi
