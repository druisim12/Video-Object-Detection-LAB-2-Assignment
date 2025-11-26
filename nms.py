# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from mega_core import _C

# Try importing Apex AMP (optional)
try:
    from apex import amp
    # Only valid with fp32 inputs - give AMP the hint
    nms = amp.float_function(_C.nms)
except ImportError:
    # Fallback: no Apex, use plain NMS
    nms = _C.nms

# Add docstring safely (ignore if object doesn't allow it)
try:
    nms.__doc__ = """
    This function performs Non-Maximum Suppression (NMS).
    If Apex AMP is available, it uses amp.float_function for mixed precision.
    Otherwise, it defaults to the standard implementation.
    """
except (AttributeError, TypeError):
    pass

