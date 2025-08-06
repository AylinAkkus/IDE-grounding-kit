from ..views import DOMElementNode


def is_visible_element(node: DOMElementNode) -> bool:
	"""
	Determines if a DOM element is visible based on computed styles and layout rects.

	This function checks for:
	- Element has positive width and height from offsetRects (equivalent to offsetWidth > 0 && offsetHeight > 0)
	- Visibility is not 'hidden'
	- Display is not 'none'

	Args:
	    node: DOMElementNode to check for visibility

	Returns:
	    bool: True if element is visible, False otherwise
	"""
	if not node or not isinstance(node, DOMElementNode):
		return False

	# Check computed styles
	visibility = node.computed_styles.get('visibility', '')
	display = node.computed_styles.get('display', '')

	# Check if display is none or visibility is hidden
	if display == 'none' or visibility == 'hidden':
		return False

	# Check dimensions from layout rects (equivalent to offsetWidth/offsetHeight)
	# offsetRects is a list of rectangles with 'width' and 'height' properties
	offset_rects = node.computed_properties.get('offsetRects', [])

	# Element is visible if any rect has positive dimensions
	return any(rect.get('width', 0) > 0 and rect.get('height', 0) > 0 for rect in offset_rects)
