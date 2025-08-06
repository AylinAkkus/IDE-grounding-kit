import asyncio
import logging

from playwright.async_api import Page

from ..views import DOMElementNode

logger = logging.getLogger(__name__)


async def highlight_element(element: DOMElementNode, page: Page):
	"""Highlight an element in the browser by setting its style attribute"""

	# Define colors array like in the Go example
	colors = [
		'#FF0000',  # Red
		'#00FF00',  # Green
		'#0000FF',  # Blue
		'#FFA500',  # Orange
		'#800080',  # Purple
		'#008080',  # Teal
		'#FF69B4',  # Hot Pink
		'#4B0082',  # Indigo
		'#FF4500',  # Orange Red
		'#2E8B57',  # Sea Green
		'#DC143C',  # Crimson
		'#4682B4',  # Steel Blue
	]

	# Get a color (using element's node_id as index for consistency)
	color_index = element.node_id % len(colors)
	base_color = colors[color_index]

	# Create style with border and background color (with opacity)
	style = f'border: 2px solid {base_color} !important; background-color: {base_color}1A !important; box-sizing: border-box !important;'

	session = await element.get_session(page)
	try:
		await session.send('DOM.getDocument', {'depth': -1, 'pierce': True})
		await session.send('DOM.setAttributeValue', {'nodeId': element.node_id, 'name': 'style', 'value': style})
	except Exception:
		return
