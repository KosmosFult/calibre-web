# -*- coding: utf-8 -*-
"""AI integration package.

The Flask blueprint is exported here so the existing application bootstrap can
keep using ``from .ai import ai``.  The implementation itself lives in small,
testable modules under this package.
"""

from .routes import ai

__all__ = ["ai"]
