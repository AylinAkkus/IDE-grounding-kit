"""
DOM Optimized Module

This module provides optimized DOM processing using Chrome DevTools Protocol (CDP).
"""

from .service import DOMService
from .views import DOMElementNode, DOMTextNode, DOMTree, WorkbenchLayout, DOMState
from .utils.enrichment import DOMEnricher, enrich_dom_tree

__all__ = [
    'DOMService',
    'DOMElementNode', 
    'DOMTextNode',
    'DOMTree',
    'DOMEnricher',
    'enrich_dom_tree',
    'WorkbenchLayout'
]