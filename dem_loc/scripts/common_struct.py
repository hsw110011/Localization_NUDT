#!/usr/bin/env python3
"""Compatibility import for loc_bev.common_struct."""

import os
import sys

_PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from loc_bev.common_struct import *  # noqa: F401,F403
