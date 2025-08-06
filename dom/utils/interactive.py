from enum import Enum

from ..views import DOMElementNode


class CursorType(Enum):
	"""Interactive cursor types that indicate clickable elements"""

	POINTER = 'pointer'
	MOVE = 'move'
	TEXT = 'text'
	GRAB = 'grab'
	GRABBING = 'grabbing'
	CELL = 'cell'
	COPY = 'copy'
	ALIAS = 'alias'
	ALL_SCROLL = 'all-scroll'
	COL_RESIZE = 'col-resize'
	CONTEXT_MENU = 'context-menu'
	CROSSHAIR = 'crosshair'
	E_RESIZE = 'e-resize'
	EW_RESIZE = 'ew-resize'
	HELP = 'help'
	N_RESIZE = 'n-resize'
	NE_RESIZE = 'ne-resize'
	NESW_RESIZE = 'nesw-resize'
	NS_RESIZE = 'ns-resize'
	NW_RESIZE = 'nw-resize'
	NWSE_RESIZE = 'nwse-resize'
	ROW_RESIZE = 'row-resize'
	S_RESIZE = 's-resize'
	SE_RESIZE = 'se-resize'
	SW_RESIZE = 'sw-resize'
	VERTICAL_TEXT = 'vertical-text'
	W_RESIZE = 'w-resize'
	ZOOM_IN = 'zoom-in'
	ZOOM_OUT = 'zoom-out'


class NonInteractiveCursor(Enum):
	"""Cursor types that indicate non-interactive elements"""

	NOT_ALLOWED = 'not-allowed'
	NO_DROP = 'no-drop'
	WAIT = 'wait'
	PROGRESS = 'progress'
	INITIAL = 'initial'
	INHERIT = 'inherit'


class InteractiveTag(Enum):
	"""HTML tags that are inherently interactive"""

	A = 'a'
	BUTTON = 'button'
	INPUT = 'input'
	SELECT = 'select'
	TEXTAREA = 'textarea'
	DETAILS = 'details'
	SUMMARY = 'summary'
	LABEL = 'label'
	OPTION = 'option'
	OPTGROUP = 'optgroup'
	FIELDSET = 'fieldset'
	LEGEND = 'legend'


class InteractiveRole(Enum):
	"""ARIA roles that indicate interactive elements"""

	BUTTON = 'button'
	LINK = 'link'
	MENUITEM = 'menuitem'
	MENUITEMRADIO = 'menuitemradio'
	MENUITEMCHECKBOX = 'menuitemcheckbox'
	RADIO = 'radio'
	CHECKBOX = 'checkbox'
	TAB = 'tab'
	SWITCH = 'switch'
	SLIDER = 'slider'
	SPINBUTTON = 'spinbutton'
	COMBOBOX = 'combobox'
	SEARCHBOX = 'searchbox'
	TEXTBOX = 'textbox'
	LISTBOX = 'listbox'
	OPTION = 'option'
	SCROLLBAR = 'scrollbar'
	TREEITEM = 'treeitem'
	GRIDCELL = 'gridcell'
	COLUMNHEADER = 'columnheader'
	ROWHEADER = 'rowheader'


class DisableAttribute(Enum):
	"""Attributes that disable interactivity"""

	DISABLED = 'disabled'
	READONLY = 'readonly'


# Pre-computed sets for performance
INTERACTIVE_CURSORS: set[str] = {cursor.value for cursor in CursorType}
NON_INTERACTIVE_CURSORS: set[str] = {cursor.value for cursor in NonInteractiveCursor}
INTERACTIVE_TAGS: set[str] = {tag.value for tag in InteractiveTag}
INTERACTIVE_ROLES: set[str] = {role.value for role in InteractiveRole}
DISABLE_ATTRIBUTES: set[str] = {attr.value for attr in DisableAttribute}
MOUSE_EVENT_ATTRIBUTES: set[str] = {'onclick', 'onmousedown', 'onmouseup', 'ondblclick'}
INTERACTIVE_CLASS_INDICATORS: set[str] = {'button', 'dropdown-toggle'}
INTERACTIVE_DATA_ATTRIBUTES: set[str] = {'data-index', 'data-toggle', 'data-action', 'data-onclick', 'data-click'}
SEMANTIC_INTERACTIVE_ATTRIBUTES: set[str] = {'aria-label', 'aria-labelledby', 'aria-describedby'}
EMPTY_HREF_VALUES: set[str] = {'', '#', 'javascript:void(0)', 'javascript:;'}


def _has_interactive_cursor(node: DOMElementNode) -> bool:
	"""Check if element has an interactive cursor style"""
	if node.tag == 'html':
		return False

	cursor = node.computed_styles.get('cursor', '')
	return cursor in INTERACTIVE_CURSORS


def _is_explicitly_disabled(node: DOMElementNode) -> bool:
	"""Check if element is explicitly disabled"""
	# Check disable attributes
	for attr in DISABLE_ATTRIBUTES:
		if attr in node.attributes and node.attributes[attr] in ('true', ''):
			return True

	# Note: In the DOM representation, we don't have access to JS properties
	# like element.disabled, element.readOnly, element.inert
	# These would need to be captured during DOM extraction
	return False


def _has_interactive_classes_or_data(node: DOMElementNode) -> bool:
	"""Check for interactive CSS classes or data attributes"""
	# Check CSS classes
	class_attr = node.attributes.get('class', '')
	if any(cls in class_attr for cls in INTERACTIVE_CLASS_INDICATORS):
		return True

	# Check data attributes
	for attr in INTERACTIVE_DATA_ATTRIBUTES:
		if attr in node.attributes:
			return True

	# Check specific data attribute values
	if node.attributes.get('data-toggle') == 'dropdown':
		return True

	if node.attributes.get('aria-haspopup') == 'true':
		return True

	# Check for any data-* attribute that suggests interactivity
	for attr_name in node.attributes:
		if attr_name.startswith('data-') and any(
			keyword in attr_name.lower() for keyword in ['click', 'action', 'toggle', 'trigger', 'handler']
		):
			return True

	return False


def _has_mouse_event_handlers(node: DOMElementNode) -> bool:
	"""Check for mouse event handler attributes"""
	return any(attr in node.attributes for attr in MOUSE_EVENT_ATTRIBUTES)


def _has_valid_tabindex(node: DOMElementNode) -> bool:
	"""Check if element has a valid tabindex that makes it interactive"""
	if 'tabindex' in node.attributes:
		try:
			tabindex = int(node.attributes['tabindex'])
			return tabindex >= 0
		except ValueError:
			pass
	return False


def _is_valid_link(node: DOMElementNode) -> bool:
	"""Check if element is a valid interactive link"""
	if node.tag != 'a':
		return False

	href = node.attributes.get('href', '')
	return href and href not in EMPTY_HREF_VALUES


def _has_pointer_events_disabled(node: DOMElementNode) -> bool:
	"""Check if pointer events are disabled"""
	return node.computed_styles.get('pointer-events') == 'none'


def _has_semantic_interactive_attributes(node: DOMElementNode) -> bool:
	"""Check for semantic attributes that suggest interactivity"""
	return any(attr in node.attributes for attr in SEMANTIC_INTERACTIVE_ATTRIBUTES)


def _is_content_editable_enhanced(node: DOMElementNode) -> bool:
	"""Enhanced check for content editability"""
	# Basic contenteditable check
	contenteditable = node.attributes.get('contenteditable', '')
	if contenteditable == 'true':
		# Additional check: ensure user-select is not disabled
		user_select = node.computed_styles.get('user-select', '')
		return user_select != 'none'
	return False


def is_interactive_element(node: DOMElementNode) -> bool:
	"""
	Determines if a DOM element is interactive based on multiple criteria.

	This function checks for:
	- Interactive cursor styles (pointer, grab, etc.)
	- Interactive HTML tags (button, input, etc.)
	- Interactive ARIA roles
	- Event handler attributes
	- CSS classes and data attributes indicating interactivity
	- Contenteditable elements
	- Tabindex values
	- Valid links with meaningful hrefs
	- Pointer events status
	- Semantic interactive attributes

	Args:
	    node: DOMElementNode to check for interactivity

	Returns:
	    bool: True if element is interactive, False otherwise
	"""
	if not node or not isinstance(node, DOMElementNode):
		return False

	# Early exit: check if pointer events are disabled
	if _has_pointer_events_disabled(node):
		return False

	# Primary check: interactive cursor style (most reliable indicator)
	if _has_interactive_cursor(node):
		return True

	# Special case: valid links with meaningful hrefs
	if _is_valid_link(node):
		return True

	# Check for valid tabindex (makes any element focusable/interactive)
	if _has_valid_tabindex(node):
		return True

	# Check for inherently interactive HTML tags
	if node.tag in INTERACTIVE_TAGS:
		# Verify not disabled by cursor or attributes
		cursor = node.computed_styles.get('cursor', '')
		if cursor in NON_INTERACTIVE_CURSORS:
			return False

		if _is_explicitly_disabled(node):
			return False

		return True

	# Check for interactive ARIA roles
	role = node.attributes.get('role', '')
	aria_role = node.attributes.get('aria-role', '')

	if role in INTERACTIVE_ROLES or aria_role in INTERACTIVE_ROLES:
		return True

	# Enhanced contenteditable check
	if _is_content_editable_enhanced(node):
		return True

	# Check for interactive CSS classes and data attributes
	if _has_interactive_classes_or_data(node):
		return True

	# Check for mouse event handlers
	if _has_mouse_event_handlers(node):
		return True

	# Check for semantic interactive attributes (aria-label, etc.)
	if _has_semantic_interactive_attributes(node):
		return True

	return False
