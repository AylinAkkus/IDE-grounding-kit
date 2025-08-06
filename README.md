# IDE Grounding Kit

**IDE Grounding Kit** is a modular pipeline for generating grounded UI datasets from IDEs like **Cursor** or **VS Code**, using CDP calls to Electron Browser with Browser Use Extraction layer.
It extracts icon elements from the DOM, annotates their functions using an LLM, and formats the output into datasets suitable for training with share-gpt format. It can very easily be expanded to extract other custom elements from the GUI (for example function calls from the editor panel) by writing a simple filter function.

## Run Cursor or VS Code with Remote Debugging

```cmd
"C:\Users\<YOUR_USERNAME>\AppData\Local\Programs\cursor\Cursor.exe" ^
  --remote-debugging-port=9222 ^
  --allow-remote-origins="*"
```
## Set Your API Key
Create a .env file in the project root directory and add your API key:
```sh
VLM_API_KEY=your_api_key_here
```
## Get the WebSocket URL
Open your browser and visit:
http://localhost:9222/json
Copy the websocket url that looks like:
ws://127.0.0.1:9222/devtools/browser/<session_id>

## Run the Pipeline
```sh
python main.py \
  --mode grounding \
  --theme-name dark_theme \
  --web-socket-url ws://127.0.0.1:9222/devtools/browser/<session_id>
```
## Output Structure
After running the script, the output folder will look like:

```sh
data/
└── dark_theme/
    ├── images/                          # Icon bounding box images
    ├── icons_data.json                  # Raw icon metadata
    ├── annotated_icons.jsonl            # Annotated instructions
    ├── dark_theme.png                   # Full-page screenshot
    └── dark_theme_grounding_dataset.json  # Final LLaMAFactory dataset
```
