# Vector Render
Vector Render is a Blender add-on for creating vector graphics from the objects seen through the camera view.
This is useful for example in publications to illustrate three dimensional objects.
The advantage over normal renders using raster graphics is that there is no resolution problem when printing and that the file sizes can be quite small.

## Features
The add-on has the following features:

- Output formats: SVG (Scalable Vector Graphics) and Metapost
- Support for perspective and orthographic projection
- Rendering of mesh objects (with modifiers applied as an option)
- Drawing of text labels with positions in 3D
- Drawing of mesh edges/wireframe where edges obscured by geometry are . . .
  - hidden
  - drawn with a dash pattern
  - shown normally
- Hiding of edges within planes (using an angle limit)
- Drawing of mesh faces (coloured by material as an option)

## Installation
1. Download the latest [release](https://github.com/mrossini-ethz/vector-render/releases) or clone the repository into a directory of your convenience.
2. If you downloaded the zip file, extract it.
3. Open Blender.
4. Go to File -> User Preferences -> Addons.
5. At the bottom of the window, chose *Install From File*.
6. Select the file `vector-render.py` from the directory into which you cloned/extracted the repository.
7. Activate the checkbox for the plugin that you will now find in the list.
