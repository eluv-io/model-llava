#!/bin/bash

set -e

buildscripts/build_container.bash -t "llava:${IMAGE_TAG:-latest}" -f Containerfile .
