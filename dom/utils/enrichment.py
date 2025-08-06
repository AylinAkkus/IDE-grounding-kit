"""
DOM Enrichment Utility

This module provides functionality to enrich DOM elements with computed styles,
bounding boxes, and derived properties like visibility and interactivity.
"""

import logging
from typing import Any

from playwright.async_api import CDPSession, Page

from .interactive import is_interactive_element
from .visible import is_visible_element
from ..views import DOMElementNode, DOMTree

logger = logging.getLogger(__name__)


class DOMEnricher:
	"""
	Enriches DOM elements with computed styles, layout data, and derived properties.
	"""

	# Computed styles that are useful for determining element visibility and interactivity
	USEFUL_COMPUTED_STYLES = [
		'display',
		'visibility',
		'opacity',
		'position',
		'z-index',
		'pointer-events',
		'cursor',
		'overflow',
		'overflow-x',
		'overflow-y',
		'width',
		'height',
		'top',
		'left',
		'right',
		'bottom',
		'transform',
		'clip',
		'clip-path',
		'user-select',
		'background-color',
		'color',
		'border',
		'margin',
		'padding',
	]

	def __init__(self):
		pass

	async def enrich_dom_tree(
		self,
		dom_tree: DOMTree,
		sessions: dict[str, CDPSession],
		session_frame_urls: dict[CDPSession, str],
		main_session: CDPSession,
		page: Page = None,
	):
		"""
		Enrich the entire DOM tree with computed styles and layout data.

		Args:
		    dom_tree: The DOM tree to enrich
		    backend_node_map: Mapping of backend node IDs to elements
		    sessions: All CDP sessions (for iframes)
		    session_frame_urls: Mapping of sessions to frame URLs
		    main_session: Main CDP session
		    page: The Playwright page object (needed for dynamic iframe session creation)
		"""
		try:
			logger.info('Starting DOM tree enrichment...')

			# Get all elements from the DOM tree to enrich
			all_elements = dom_tree.get_all_elements()

			# Create a mapping by frame_url and backend_node_id for efficient lookup
			elements_by_frame_and_backend = {}
			for element in all_elements:
				frame_key = element.frame_url
				if frame_key not in elements_by_frame_and_backend:
					elements_by_frame_and_backend[frame_key] = {}
				elements_by_frame_and_backend[frame_key][element.backend_node_id] = element

			# Enrich main frame
			main_frame_url = session_frame_urls.get(main_session, 'unknown')
			if main_frame_url in elements_by_frame_and_backend:
				await self._enrich_session_nodes(main_session, main_frame_url, elements_by_frame_and_backend[main_frame_url])

			# Enrich iframe sessions that we already have
			for session in sessions.values():
				frame_url = session_frame_urls.get(session, 'unknown')
				if frame_url in elements_by_frame_and_backend:
					await self._enrich_session_nodes(session, frame_url, elements_by_frame_and_backend[frame_url])

			# Check for unenriched iframe elements and create sessions dynamically
			if page:
				# Get all frame URLs that haven't been enriched yet
				enriched_urls = {main_frame_url}
				enriched_urls.update(session_frame_urls.values())

				unenriched_frame_urls = []
				for frame_url in elements_by_frame_and_backend.keys():
					if frame_url not in enriched_urls and frame_url != 'unknown':
						unenriched_frame_urls.append(frame_url)

				# Process each unenriched frame URL
				for frame_url in unenriched_frame_urls:
					logger.debug(f'Found unenriched frame URL: {frame_url}')

					# Find an element from this frame to use for session creation
					# We need any element that belongs to this frame
					frame_element = None
					for element in elements_by_frame_and_backend[frame_url].values():
						frame_element = element
						break

					if frame_element:
						try:
							# Use the element's get_session_id method to create a CDP session
							logger.debug(f'Creating CDP session for frame: {frame_url}')
							iframe_session = await frame_element.get_session(page)

							if iframe_session:
								# Store the session for future use
								sessions[frame_url] = iframe_session
								session_frame_urls[iframe_session] = frame_url

								# Enrich all elements from this frame
								await self._enrich_session_nodes(
									iframe_session, frame_url, elements_by_frame_and_backend[frame_url]
								)

						except Exception as e:
							logger.warning(f'Failed to create session for frame {frame_url}: {e}')

			logger.info('DOM tree enrichment completed')

		except Exception as e:
			logger.warning(f'Error during DOM tree enrichment: {e}')

	async def _enrich_session_nodes(self, session: CDPSession, frame_url: str, backend_node_map: dict[int, DOMElementNode]):
		"""
		Enrich nodes for a specific CDP session using DOMSnapshot.captureSnapshot.
		"""
		try:
			logger.debug(f'Enriching nodes for frame: {frame_url}')

			# Capture snapshot with all the data we need
			snapshot_result = await session.send(
				'DOMSnapshot.captureSnapshot',
				{
					'computedStyles': self.USEFUL_COMPUTED_STYLES,
					'includePaintOrder': True,
					'includeDOMRects': True,
					'includeBlendedBackgroundColors': False,
					'includeTextColorOpacities': False,
				},
			)

			# Process the snapshot data
			await self._process_snapshot_data(snapshot_result, frame_url, backend_node_map)

		except Exception as e:
			logger.warning(f'Error enriching session {frame_url}: {e}')

	async def _process_snapshot_data(
		self, snapshot_result: dict[str, Any], frame_url: str, backend_node_map: dict[int, DOMElementNode]
	):
		"""
		Process the DOMSnapshot data and enrich elements.
		"""
		try:
			documents = snapshot_result.get('documents', [])
			strings = snapshot_result.get('strings', [])

			if not documents:
				logger.debug(f'No documents in snapshot for frame {frame_url}')
				return

			enriched_count = 0

			for document in documents:
				# Extract node data
				nodes = document.get('nodes', {})
				layout = document.get('layout', {})

				# Get arrays from nodes
				backend_node_ids = nodes.get('backendNodeId', [])
				node_types = nodes.get('nodeType', [])

				# Get layout data (only for rendered elements)
				layout_node_indices = layout.get('nodeIndex', [])
				layout_bounds = layout.get('bounds', [])
				paint_orders = layout.get('paintOrders', [])

				# Create mapping from node index to layout index
				node_to_layout = {}
				for layout_idx, node_idx in enumerate(layout_node_indices):
					node_to_layout[node_idx] = layout_idx

				# Process each node
				for node_idx, backend_node_id in enumerate(backend_node_ids):
					# Skip if not an element node (type 1)
					if node_idx >= len(node_types) or node_types[node_idx] != 1:
						continue

					# Find corresponding element in our backend_node_map
					element = None
					for mapped_backend_id, mapped_element in backend_node_map.items():
						if mapped_element.frame_url == frame_url and mapped_backend_id == backend_node_id:
							element = mapped_element
							break

					if not element:
						continue

					# Get computed styles for this element (available for all elements)
					self._extract_computed_styles(element, node_idx, document, strings)

					# Get layout data if available (only for rendered elements)
					if node_idx in node_to_layout:
						layout_idx = node_to_layout[node_idx]
						self._extract_layout_data(element, layout_idx, layout_bounds, paint_orders)

					# Add derived properties
					self._add_visibility_properties(element)
					enriched_count += 1

			logger.debug(f'Enriched {enriched_count} elements in frame {frame_url}')

		except Exception as e:
			logger.warning(f'Error processing snapshot data for frame {frame_url}: {e}')

	def _extract_computed_styles(self, element: DOMElementNode, node_idx: int, document: dict[str, Any], strings: list[str]):
		"""
		Extract computed styles for an element from the snapshot.
		"""
		try:
			# Look for computed styles in the document
			# The format varies, so we need to check multiple possible locations

			# Check if there's a computedStyles section in the document
			computed_styles_data = document.get('computedStyles', {})
			if computed_styles_data:
				indices = computed_styles_data.get('index', [])
				values = computed_styles_data.get('value', [])

				# Find the styles for this node
				for i, idx in enumerate(indices):
					if idx == node_idx and i < len(values):
						style_array = values[i]
						self._parse_style_array(element, style_array, strings)
						return

			# Alternative: check if styles are in layout section
			layout = document.get('layout', {})
			layout_node_indices = layout.get('nodeIndex', [])
			layout_styles = layout.get('styles', [])

			# Find layout index for this node
			for layout_idx, layout_node_idx in enumerate(layout_node_indices):
				if layout_node_idx == node_idx and layout_idx < len(layout_styles):
					style_array = layout_styles[layout_idx]
					self._parse_style_array(element, style_array, strings)
					return

		except Exception as e:
			logger.debug(f'Error extracting computed styles for element: {e}')

	def _parse_style_array(self, element: DOMElementNode, style_array: list[int], strings: list[str]):
		"""
		Parse a style array from the snapshot into computed styles.
		"""
		try:
			styles_dict = {}

			# The style array contains values in the same order as the USEFUL_COMPUTED_STYLES array
			# Each value is an index into the strings array
			for i, value_idx in enumerate(style_array):
				if i < len(self.USEFUL_COMPUTED_STYLES) and value_idx < len(strings):
					style_name = self.USEFUL_COMPUTED_STYLES[i]
					style_value = strings[value_idx]
					styles_dict[style_name] = style_value

			if styles_dict:
				element.computed_styles.update(styles_dict)

		except Exception as e:
			logger.debug(f'Error parsing style array: {e}')

	def _extract_layout_data(
		self, element: DOMElementNode, layout_idx: int, layout_bounds: list[list[float]], paint_orders: list[int]
	):
		"""
		Extract layout data (bounding box, paint order) for an element.
		"""
		try:
			# Extract bounding box
			if layout_idx < len(layout_bounds):
				bounds = layout_bounds[layout_idx]
				if len(bounds) >= 4:  # [x, y, width, height]
					bounding_box = {'x': bounds[0], 'y': bounds[1], 'width': bounds[2], 'height': bounds[3]}
					element.bounding_box = bounding_box

					# Also populate computed_properties.offsetRects for visibility utility
					element.computed_properties['offsetRects'] = [{'width': bounds[2], 'height': bounds[3]}]

			# Extract paint order
			if layout_idx < len(paint_orders):
				element.paint_order = paint_orders[layout_idx]

		except Exception as e:
			logger.debug(f'Error extracting layout data: {e}')

	def _add_visibility_properties(self, element: DOMElementNode):
		"""
		Add convenience properties for checking element visibility and interactivity.
		"""
		styles = getattr(element, 'computed_styles', {})

		# Use utility functions for visibility and interactivity
		element.is_visible = is_visible_element(element)
		element.is_interactive = is_interactive_element(element)


# Convenience function for easy usage
async def enrich_dom_tree(
	dom_tree: DOMTree,
	sessions: dict[str, CDPSession],
	session_frame_urls: dict[CDPSession, str],
	main_session: CDPSession,
	page: Page = None,
) -> None:
	"""
	Convenience function to enrich a DOM tree.

	Args:
	    dom_tree: The DOM tree to enrich
	    sessions: All CDP sessions (for iframes)
	    session_frame_urls: Mapping of sessions to frame URLs
	    main_session: Main CDP session
	    page: The Playwright page object (needed for dynamic iframe session creation)
	"""
	enricher = DOMEnricher()
	await enricher.enrich_dom_tree(dom_tree, sessions, session_frame_urls, main_session, page)
