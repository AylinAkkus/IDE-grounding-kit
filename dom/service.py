import asyncio
import logging
from enum import IntEnum
from typing import Any

from playwright.async_api import BrowserContext, CDPSession, Page

from .utils.enrichment import enrich_dom_tree
from .views import DOMElementNode, DOMTextNode, DOMTree, WorkbenchLayout, DOMState

# @file purpose: Defines DOMService for building DOM trees using raw CDP calls

logger = logging.getLogger(__name__)


class NodeType(IntEnum):
	"""CDP DOM Node Types"""

	ELEMENT = 1
	TEXT = 3
	COMMENT = 8
	DOCUMENT = 9
	DOCTYPE = 10
	DOCUMENT_FRAGMENT = 11  # Shadow root


class DOMService:
	"""˘
	Service for building DOM trees using raw Chrome DevTools Protocol (CDP) calls.
	This implementation mirrors the Go version but uses Python and Playwright's CDP interface.
	For iframe content extraction, it uses PyCDP when available to overcome Playwright's CDP session limitations.
	"""

	def __init__(self, page: Page, context: BrowserContext, cdp_session: CDPSession):
		self.page: Page = page
		self.context: BrowserContext = context
		self.cdp_session: CDPSession = cdp_session
		self.dom_tree: DOMTree | None = None
		# State tracking for frame processing
		self.processed_frames: set[str] = set()
		self.processed_iframe_nodes: set[int] = set()
		self.sessions: dict[str, Any] = {}
		self.session_frame_urls: dict[CDPSession, str] = {}
		# Add depth tracking for iframe recursion
		self.iframe_depth = 0
		self.max_iframe_depth = 3  # Prevent infinite recursion

	async def set_page(self, page: Page):
		"""Set the page to work with"""
		self.page = page
		self.cdp_session = await page.context.new_cdp_session(page)


	async def get_dom_state(self) -> DOMState:
		"""Get the cached DOM tree or build a new one"""
		if self.dom_tree is None:
			self.dom_tree = await asyncio.wait_for(
				self.build_dom_tree(),
				timeout=30.0,  # 30 second overall timeout
			)

		interactive_elements_list = self.dom_tree.get_interactive_elements()
		selector_map = {element.backend_node_id: element for element in interactive_elements_list}

		return DOMState(self.dom_tree, selector_map)

	async def get_dom_tree(self) -> DOMTree | None:
		"""Get the cached DOM tree or build a new one"""
		if self.dom_tree is None:
			return await asyncio.wait_for(
				self.build_dom_tree(),
				timeout=30.0,  # 30 second overall timeout
			)
		return self.dom_tree

	async def build_dom_tree(self) -> DOMTree:
		"""
		Build a comprehensive DOM tree including main document, iframes, and shadow roots.
		Returns a unified DOMTree with all content merged into a single tree structure.
		The tree is fully enriched with computed styles, positioning, and visibility data.
		"""
		try:
			logger.info('Starting DOM tree construction...')

			# Reset state
			self.processed_frames.clear()
			self.processed_iframe_nodes.clear()
			self.sessions.clear()
			self.session_frame_urls.clear()

			# get targets and print all info
			targets_result = await self.cdp_session.send('Target.getTargets')
			target_infos = targets_result.get('targetInfos', [])
			for target in target_infos:
				logger.info(f'Target: {target}')

			# Get the main document and start recursive traversal
			doc_result = await self.cdp_session.send('DOM.getDocument', {'depth': -1, 'pierce': True})

			root_node = doc_result.get('root')
			if not root_node:
				raise ValueError('CDP did not provide root node - cannot build DOM tree without CDP data')
			else:
				# Get the document URL from the root node
				document_url = root_node.get('documentURL', self.page.url)
				# Store the main session URL
				self.session_frame_urls[self.cdp_session] = document_url
				# Start recursive traversal from root
				root_element = await self._traverse_node_recursive(root_node, document_url)

			if root_element is None:
				raise ValueError('Failed to build DOM tree from CDP data - root element is None')

			# Create the DOM tree
			self.dom_tree = DOMTree(root_element)

			# Enrich the tree with computed styles and positioning data
			await enrich_dom_tree(self.dom_tree, self.sessions, self.session_frame_urls, self.cdp_session, self.page)

			logger.info('DOM tree construction and enrichment completed successfully')
			return self.dom_tree

		except Exception as e:
			logger.error(f'Error building DOM tree: {e}')
			# Re-raise the exception instead of creating fallback
			raise RuntimeError(f'Failed to build DOM tree from CDP data: {e}') from e
		
	async def get_workbench_layout(self, dom_tree: DOMTree) -> WorkbenchLayout:
		"""Get the workbench layout from the DOM tree"""
		all_elements = dom_tree.get_all_elements()
		workbench_nodes = dom_tree.root.find_all(lambda x: "monaco-workbench" in x.attributes.get("class", ""))
		if not workbench_nodes:
			raise RuntimeError("Could not find .monaco-workbench")
		workbench = workbench_nodes[0]
		grid_node = next(
			(child for child in workbench.children if "monaco-grid-view" in child.attributes.get("class", "")),
			None
		)
		if not grid_node:
			raise RuntimeError("Could not find monaco-grid-view among workbench children")
		grid_branch = next(
			(c for c in grid_node.children if "monaco-grid-branch-node" in c.attributes.get("class", "")),
			None
		)
		if not grid_branch:
			raise RuntimeError("No monaco-grid-branch-node inside monaco-grid-view")

		layout_parts = grid_branch.find_all(lambda x: "part" in x.attributes.get("class", ""))
		
		return WorkbenchLayout(layout_parts)

	async def _traverse_node_recursive(self, cdp_node: dict[str, Any], frame_url: str) -> DOMElementNode | None:
		"""
		Recursively traverse a CDP node and its children, switching to iframe sessions when needed.
		"""
		try:
			node_id = cdp_node.get('nodeId')
			# Validate that CDP provides node_id
			if node_id is None:
				raise ValueError(f'CDP did not provide node_id for node: {cdp_node}')

			# Extract all required CDP fields
			backend_node_id = cdp_node.get('backendNodeId')
			if backend_node_id is None:
				raise ValueError(f'CDP did not provide backendNodeId for node: {cdp_node}')

			node_type = NodeType(cdp_node.get('nodeType', 0))
			node_name = cdp_node.get('nodeName', '')

			# Handle different node types
			if node_type == NodeType.DOCUMENT:
				# For document nodes, process children directly
				children = cdp_node.get('children', [])
				for child in children:
					result = await self._traverse_node_recursive(child, frame_url)
					if result:
						return result
				return None

			elif node_type == NodeType.DOCTYPE:
				# Skip DOCTYPE nodes
				return None

			elif node_type == NodeType.ELEMENT:
				# Extract attributes
				attributes = {}
				attrs_list = cdp_node.get('attributes', [])
				for i in range(0, len(attrs_list), 2):
					if i + 1 < len(attrs_list):
						attributes[attrs_list[i]] = attrs_list[i + 1]

				# Create DOM element
				element = DOMElementNode(
					node_id=node_id,
					tag=node_name.lower(),
					backend_node_id=backend_node_id,
					frame_url=frame_url,
					attributes=attributes,
				)

				# Process all children (elements, text nodes, shadow roots)
				children = cdp_node.get('children', [])
				for child in children:
					child_element = await self._traverse_node_recursive(child, frame_url)
					if child_element:
						element.append_child(child_element)
						# If it's a text node, aggregate its text to parent's text_content
						if isinstance(child_element, DOMTextNode):
							element.text_content += ' ' + child_element.text if element.text_content else child_element.text

				# Process shadow roots - they can appear as a separate property
				# Shadow roots are DOCUMENT_FRAGMENT nodes, we process their children directly
				shadow_roots = cdp_node.get('shadowRoots', [])
				for shadow_root in shadow_roots:
					if shadow_root.get('nodeType') == NodeType.DOCUMENT_FRAGMENT:
						# Process shadow root children directly into this element
						shadow_children = shadow_root.get('children', [])
						for shadow_child in shadow_children:
							child_element = await self._traverse_node_recursive(shadow_child, frame_url)
							if child_element:
								element.append_child(child_element)
					else:
						# If it's not a document fragment, process normally
						shadow_element = await self._traverse_node_recursive(shadow_root, frame_url)
						if shadow_element:
							element.append_child(shadow_element)

				# Special handling for iframe elements
				if element.tag == 'iframe':
					logger.debug(
						f'Found iframe element: src={element.attributes.get("src", "")}, backend_node_id={backend_node_id}'
					)
					# Check if this iframe node has already been processed
					if backend_node_id in self.processed_iframe_nodes:
						logger.debug(f'Skipping already processed iframe node: {backend_node_id}')
					else:
						self.processed_iframe_nodes.add(backend_node_id)

						# Check if CDP already provided content for this iframe
						iframe_content_node = cdp_node.get('contentDocument')
						if iframe_content_node:
							logger.debug('Found contentDocument in CDP data for iframe')
							# Process the content document directly
							iframe_content = await self._traverse_node_recursive(iframe_content_node, frame_url)
							if iframe_content:
								element.append_child(iframe_content)
						else:
							# Try to get iframe content through frame API
							iframe_document_url = element.attributes.get('src', '')
							element.frame_url = iframe_document_url
							child_node = await self._process_iframe_content(element)
							if child_node:
								element.append_child(child_node)

				return element

			elif node_type == NodeType.DOCUMENT_FRAGMENT:
				# Document fragments only appear as shadow roots in the shadowRoots array
				# They should never be traversed directly through normal recursion
				logger.error(f'Unexpected DOCUMENT_FRAGMENT in normal traversal: node_id={node_id}')
				return None

			elif node_type == NodeType.TEXT:
				# Handle text nodes directly
				text_value = cdp_node.get('nodeValue', '')
				if text_value and text_value.strip():
					return DOMTextNode(
						node_id=node_id, text=text_value.strip(), backend_node_id=backend_node_id, frame_url=frame_url
					)
				return None

			elif node_type == NodeType.COMMENT:
				# Skip comment nodes
				return None

			else:
				logger.debug(f'Unhandled node type {node_type}: {node_name}')
				return None

		except Exception as e:
			logger.error(f'Error traversing node: {e}')
			return None

	async def _process_iframe_content(self, iframe_element: DOMElementNode):
		"""
		Process iframe content using Playwright CDP session with proper target management.
		Returns the root HTML element of the iframe content.
		"""
		# Check depth limit
		if self.iframe_depth >= self.max_iframe_depth:
			logger.warning(f'Maximum iframe depth ({self.max_iframe_depth}) reached, skipping frame: {iframe_element.frame_url}')
			return None

		if not iframe_element.frame_url or iframe_element.frame_url in self.processed_frames:
			logger.debug(f'Skipping already processed frame: {iframe_element.frame_url}')
			return

		self.processed_frames.add(iframe_element.frame_url)
		self.iframe_depth += 1  # Increment depth

		try:
			# Add timeout to prevent infinite waiting
			result = await asyncio.wait_for(
				self._process_iframe_with_playwright_cdp(iframe_element),
				timeout=10.0,  # 10 second timeout per iframe
			)
			return result

		except TimeoutError:
			logger.warning(f'Timeout processing iframe content for frame {iframe_element.frame_url}')
			return None
		except Exception as e:
			logger.warning(f'Error processing iframe content for frame {iframe_element.frame_url}: {e}')
			return None
		finally:
			self.iframe_depth -= 1  # Decrement depth when done

	async def _process_iframe_with_playwright_cdp(self, iframe_element: DOMElementNode):
		"""Process iframe content using Playwright's frame API and CDP"""
		try:
			# Find the iframe frame using Playwright's frame API
			frame_cdp_session = await iframe_element.get_session(self.page)

			if not frame_cdp_session:
				logger.warning(f'Could not create CDP session for iframe: {iframe_element.frame_url}')
				return None

			logger.debug(f'Getting DOM document for frame: {iframe_element.frame_url}')
			frame_doc_result = await asyncio.wait_for(
				frame_cdp_session.send('DOM.getDocument', {'depth': -1, 'pierce': True}),
				timeout=5.0,  # 5 second timeout for CDP call
			)

			root = frame_doc_result.get('root')
			if root:
				# Get the actual document URL from the iframe document
				iframe_document_url = root.get('documentURL', iframe_element.frame_url)
				logger.debug(f'Got iframe document: {iframe_document_url}')
				# Store the iframe session URL
				self.session_frame_urls[frame_cdp_session] = iframe_document_url
				# Store the session
				self.sessions[iframe_document_url] = frame_cdp_session
				# Use the iframe document URL for the iframe content
				return await self._traverse_node_recursive(root, iframe_document_url)
			else:
				logger.warning('No root node in iframe document')
				return None

		except Exception as e:
			logger.error(f'Error processing iframe with Playwright: {e}')
			import traceback

			logger.debug(traceback.format_exc())
			return None
