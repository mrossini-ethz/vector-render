# Blender Add-on: Vector Render
# Copyright (C) 2017  Marco Rossini
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 2 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# This Blender plugin is based on the research paper "Recovery of Intrinsic
# and Extrinsic Camera Parameters Using Perspective Views of Rectangles" by
# T. N. Tan, G. D. Sullivan and K. D. Baker, Department of Computer Science,
# The University of Reading, Berkshire RG6 6AY, UK, Email: T.Tan@reading.ac.uk,
# from the Proceedings of the British Machine Vision Conference, published by
# the BMVA Press.

bl_info = {
    "name": "Vector Render",
    "author": "Marco Rossini",
    "version": (0, 0, 2),
     "warning": "This is an unreleased development version.",
    "blender": (2, 80, 0),
    "location": "Properties > Render > Vector Render",
    "description": "Renders the camera view to a vector graphic such as SVG.",
    "wiki_url": "https://github.com/mrossini-ethz/vector-render",
    "tracker_url": "https://github.com/mrossini-ethz/vector-render/issues",
    "support": "COMMUNITY",
    "category": "Render"
}

if "bpy" in locals():
    import importlib as imp
    imp.reload(main)
else:
    from . import main

import bpy

def register():
    bpy.utils.register_class(main.VectorRender)
    bpy.utils.register_class(main.VectorRenderPanel)


def unregister():
    bpy.utils.unregister_class(main.VectorRender)
    bpy.utils.unregister_class(main.VectorRenderPanel)

if __name__ == "__main__":
    register()

