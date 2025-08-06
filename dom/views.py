import asyncio
import json
from collections.abc import Callable
from typing import (
	Any,
	Optional,
)
from urllib.parse import urlparse

from playwright.async_api import CDPSession, Page


class DOMNode:
	"""
	Base node for any node in the DOM tree.
	"""

	def __init__(self, node_id: int, backend_node_id: int, frame_url: str):
		# CDP node identifier
		self.node_id: int = node_id
		# CDP backend node identifier
		self.backend_node_id: int = backend_node_id
		# Frame URL instead of frame ID
		self.frame_url: str = frame_url
		# Pointer to parent node (None for root)
		self.parent: Optional['DOMElementNode'] = None

	def is_element(self) -> bool:
		return isinstance(self, DOMElementNode)

	def is_text(self) -> bool:
		return isinstance(self, DOMTextNode)


class DOMElementNode(DOMNode):
	"""
	Node that represents an HTML element, with tag, attributes,
	computed styles and list of children.
	"""

	def __init__(
		self,
		node_id: int,
		backend_node_id: int,
		frame_url: str,
		tag: str,
		attributes: dict[str, str] | None = None,
		text_content: str | None = None,
	):
		super().__init__(node_id, backend_node_id, frame_url)
		# Tag name, e.g. "div", "span", "a", etc.
		self.tag: str = tag.lower()
		# Attributes as they come from HTML: id, class, href, onclick…
		self.attributes: dict[str, str] = attributes or {}
		# Direct text content (only for elements that contain text)
		self.text_content: str = text_content or ''
		# Children in order
		self.children: list[DOMNode] = []
		# Computed styles or data (e.g. display, visibility, boundingBox…)
		self.computed_styles: dict[str, Any] = {}
		# Other useful properties (e.g. scrollHeight, clientHeight…)
		self.computed_properties: dict[str, Any] = {}
		# Bounding box from layout data
		self.bounding_box: dict[str, float] | None = None
		# Paint order (z-index equivalent)
		self.paint_order: int | None = None
		# Convenience visibility and interactivity flags (set during enrichment)
		self.is_visible: bool | None = None
		self.is_interactive: bool | None = None

	def append_child(self, node: DOMNode) -> None:
		node.parent = self
		self.children.append(node)

	def get_ancestors(self) -> list['DOMElementNode']:
		anc = []
		p = self.parent
		while p:
			anc.append(p)
			p = p.parent
		return anc

	async def get_session(self, page: Page) -> CDPSession:
		"""
		Get the CDP session to interact with the element by matching frame URL.
		"""

		iframe_frame = None
		for i, frame in enumerate(page.frames):
			frame_url = frame.url

			if self.frame_url == frame_url:
				iframe_frame = frame
				break

		if not iframe_frame:
			return None

		# Create a CDP session for this specific frame
		return await asyncio.wait_for(page.context.new_cdp_session(iframe_frame), timeout=3.0)

	# --------- Internal searches by predicate -------------

	def find_all(self, predicate: Callable[['DOMElementNode'], bool]) -> list['DOMElementNode']:
		"""
		Recursively traverses the subtree, returns all
		DOMElementNode instances for which predicate(e) is True.
		"""
		results: list['DOMElementNode'] = []
		if isinstance(self, DOMElementNode) and predicate(self):
			results.append(self)
		for child in self.children:
			if isinstance(child, DOMElementNode):
				results.extend(child.find_all(predicate))
		return results

	def find_by_id(self, element_id: str) -> Optional['DOMElementNode']:
		matches = self.find_all(lambda e: e.attributes.get('id') == element_id)
		return matches[0] if matches else None


class DOMTextNode(DOMNode):
	"""
	Pure text node.
	"""

	def __init__(self, node_id: int, backend_node_id: int, frame_url: str, text: str):
		super().__init__(node_id, backend_node_id, frame_url)
		self.text: str = text


class DOMTree:
	"""
	Encapsulates a complete DOM tree (with root) and offers
	convenience search methods.
	"""

	def __init__(self, root: DOMElementNode):
		self.root = root

	# --------- Search methods -------------

	def get_all_elements(self) -> list[DOMElementNode]:
		return self.root.find_all(lambda e: True)

	def get_visible_elements(self) -> list[DOMElementNode]:
		return self.root.find_all(lambda e: e.is_visible)

	def get_interactive_elements(self) -> list[DOMElementNode]:
		"""Get elements that are interactive (enriched data when available)"""
		return self.root.find_all(lambda e: e.is_interactive)

	def get_elements_with_bounding_box(self) -> list[DOMElementNode]:
		"""Get elements that have bounding box data from layout"""
		return self.root.find_all(lambda e: e.bounding_box is not None)

	def get_elements_by_paint_order(self, min_paint_order: int = 0) -> list[DOMElementNode]:
		"""Get elements with paint order >= min_paint_order (higher values are on top)"""
		return self.root.find_all(lambda e: e.paint_order is not None and e.paint_order >= min_paint_order)

	def get_element_by_id(self, node_id: int, backend_node_id: int) -> DOMElementNode | None:
		"""Get element by node_id or backend_node_id"""
		return self.root.find_all(lambda e: e.node_id == node_id or e.backend_node_id == backend_node_id)

	def get_element_by_condition(self, condition: Callable[['DOMElementNode'], bool]) -> DOMElementNode | None:
		"""Get element by condition"""
		return self.root.find_all(condition)

	# --------- LLM translation -------------

	def translate_all_to_llm(self) -> str:
		"""Convert all elements to LLM-friendly HTML format"""
		return self._to_llm_html()

	def translate_interactive_to_llm(self) -> str:
		"""Convert interactive elements to LLM-friendly HTML format"""
		return self._to_llm_html(interactive_only=True)

	def translate_visible_to_llm(self) -> str:
		"""Convert visible elements to LLM-friendly HTML format"""
		return self._to_llm_html(visible_only=True)

	# --------- LLM translation utils -------------
	def _to_llm_html(self, interactive_only: bool = False, visible_only: bool = False) -> str:
		"""
		Convert the DOM tree to LLM-friendly HTML format with numbered interactive elements.
		Interactive elements are numbered sequentially starting from 0, and a mapping is created
		to match the selector_map used elsewhere in the system.
		
		Args:
			interactive_only: If True, only include interactive elements
			visible_only: If True, only include visible elements
		"""
		formatted_lines = []
		interactive_counter = 0
		# Create mapping from LLM index to backend_node_id (same as selector_map keys)
		self._llm_index_to_backend_node_id = {}
		
		def is_interactive_element(node: DOMElementNode) -> bool:
			"""Check if an element is interactive"""
			if node.is_interactive is not None:
				return node.is_interactive
			
			# Common interactive tags
			interactive_tags = {'button', 'a', 'input', 'select', 'textarea', 'form'}
			if node.tag in interactive_tags:
				return True
				
			# Elements with click handlers or specific roles
			if 'onclick' in node.attributes or 'href' in node.attributes:
				return True
				
			role = node.attributes.get('role', '').lower()
			interactive_roles = {'button', 'link', 'menuitem', 'tab', 'checkbox', 'radio'}
			if role in interactive_roles:
				return True
				
			return False
		
		def should_include_element(node: DOMElementNode) -> bool:
			"""Determine if element should be included based on filters"""
			if interactive_only and not is_interactive_element(node):
				return False
			if visible_only and node.is_visible is False:
				return False
			return True
		
		def get_element_text(node: DOMElementNode) -> str:
			"""Extract meaningful text from element and its children"""
			texts = []
			
			# Add direct text content
			if node.text_content and node.text_content.strip():
				texts.append(node.text_content.strip())
			
			# Add text from child text nodes
			for child in node.children:
				if isinstance(child, DOMTextNode) and child.text.strip():
					texts.append(child.text.strip())
			
			return ' '.join(texts).strip()
		
		def format_attributes(node: DOMElementNode) -> str:
			"""Format relevant attributes for LLM"""
			attrs = []
			
			# Important attributes to include
			important_attrs = ['id', 'class', 'type', 'name', 'value', 'placeholder', 'title', 'aria-label', 'role', 'href', 'src', 'alt']
			
			for attr in important_attrs:
				if attr in node.attributes and node.attributes[attr].strip():
					value = node.attributes[attr].strip()
					# Truncate long values
					if len(value) > 50:
						value = value[:47] + "..."
					attrs.append(f'{attr}="{value}"')
			
			return ' '.join(attrs)
		
		def process_node(node: DOMNode, depth: int = 0) -> None:
			nonlocal interactive_counter
			
			if isinstance(node, DOMElementNode):
				if should_include_element(node):
					indent = "  " * depth
					
					# Check if this is an interactive element
					is_interactive = is_interactive_element(node)
					
					# Build the element representation
					element_parts = []
					
					# Add interactive number if applicable
					if is_interactive:
						element_parts.append(f"[{interactive_counter}]")
						# Store mapping for selector_map compatibility
						self._llm_index_to_backend_node_id[interactive_counter] = node.backend_node_id
						interactive_counter += 1
					
					# Add tag
					element_parts.append(f"<{node.tag}")
					
					# Add attributes
					attrs = format_attributes(node)
					if attrs:
						element_parts.append(attrs)
					
					# Add text content
					text = get_element_text(node)
					if text:
						element_parts.append(f">{text}")
						element_parts.append(f"</{node.tag}>")
					else:
						element_parts.append("/>")
					
					# Join and add to output
					line = indent + " ".join(element_parts)
					formatted_lines.append(line)
					
					# Process children with increased depth
					for child in node.children:
						process_node(child, depth + 1)
				else:
					# Even if we don't include this element, process its children
					for child in node.children:
						process_node(child, depth)
			
			elif isinstance(node, DOMTextNode):
				# Include standalone text nodes if they have meaningful content
				if node.text.strip() and (not node.parent or should_include_element(node.parent)):
					indent = "  " * depth
					formatted_lines.append(f"{indent}{node.text.strip()}")
		
		# Start processing from root
		process_node(self.root)
		
		# Add summary comment at the top
		summary_lines = [
			"<!-- DOM Tree for LLM -->",
			f"<!-- Interactive elements are numbered [0], [1], etc. -->",
			f"<!-- Total interactive elements: {interactive_counter} -->",
			""
		]
		
		return "\n".join(summary_lines + formatted_lines)
	
	def get_llm_index_to_backend_node_id_mapping(self) -> dict[int, int]:
		"""
		Get the mapping from LLM indices [0], [1], [2] to actual backend_node_ids used in selector_map.
		This should be called after _to_llm_html() to get the current mapping.
		
		Returns:
			dict mapping LLM index -> backend_node_id
		"""
		return getattr(self, '_llm_index_to_backend_node_id', {})

class DOMState:
	def __init__(self, dom_tree: DOMTree, selector_map: dict[int, DOMElementNode]):
		self.dom_tree = dom_tree
		self.selector_map = selector_map
	
	def get_element_by_llm_index(self, llm_index: int) -> DOMElementNode | None:
		"""
		Get a DOM element by its LLM index (e.g., the number in [0], [1], [2]).
		This uses the mapping created during LLM HTML generation.
		
		Args:
			llm_index: The index shown to the LLM (0, 1, 2, etc.)
			
		Returns:
			The corresponding DOMElementNode or None if not found
		"""
		# Get the backend_node_id for this LLM index
		mapping = self.dom_tree.get_llm_index_to_backend_node_id_mapping()
		backend_node_id = mapping.get(llm_index)
		
		if backend_node_id is None:
			return None
			
		# Return the element from the selector_map
		return self.selector_map.get(backend_node_id)

class WorkbenchLayout:
    def __init__(self, layout_parts: list[DOMElementNode]):
        self.parts = {}
        for part in layout_parts:
            class_str = part.attributes.get("class", "")
            for key in ["titlebar", "banner", "sidebar", "editor", "panel", "auxiliarybar", "statusbar"]:
                if key in class_str and key not in self.parts:
                    self.parts[key] = DOMTree(part)

    def get(self, name: str) -> Optional[DOMTree]:
        return self.parts.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.parts.items()}
    
    def tablists(self) -> list[DOMElementNode]:
        editor = self.get("editor")
        if not editor:
            return []
        return editor.root.find_all(lambda e: e.attributes.get("role") == "tablist")

    def all_open_files(self) -> list[DOMElementNode]:
        tablists = self.tablists()
        open_files = []
        for tablist in tablists:
            open_files.extend(tablist.find_all(lambda e: e.attributes.get("role") == "tab"))
        return open_files
