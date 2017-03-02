import bpy
import mathutils
from math import cos, degrees, radians, sqrt, acos, pi
from random import choice

# GENERIC FUNCTIONS -------------------------------------------------------------------------------------------------------------------------------------------

def gamma_correction(linear_colour):
    if linear_colour < 0.00313:
        return linear_colour * 12.92
    else:
        a = 0.055
        return (linear_colour ** (1 / 2.4)) * (1 + a) - a

# BINARY TREE -------------------------------------------------------------------------------------------------------------------------------------------------

class binary_tree_iter:
    def __init__(self, tree):
        self.tree = tree
        self.finished = False
    def __next__(self):
        if self.finished:
            raise StopIteration()
        if self.tree.prev:
            self.tree = self.tree.prev
            return self.tree.obj
        elif self.tree.next:
            self.tree = self.tree.next
            return self.tree.obj
        elif self.tree.parent:
            return self.upnext()
        else:
            self.finished = True
            return self.tree.obj()
    def upnext(self):
            if self.finished:
                raise StopIteration()
            tmp = self.tree
            self.tree = self.tree.parent
            if self.tree.next and self.tree.next is not tmp:
                self.tree = self.tree.next
                return self.tree.obj
            elif self.tree.parent:
                return self.upnext()
            else:
                self.finished = True
                return self.tree.obj

class binary_tree:
    def __init__(self, obj, parent = None):
        self.obj = obj
        self.parent = parent
        self.prev = None
        self.next = None

    def __iter__(self):
        return binary_tree_iter(self)

    def add(self, obj):
        if obj < self.obj:
            if self.prev:
                return self.prev.add(obj)
            else:
                self.prev = binary_tree(obj, self)
                return True
        elif obj > self.obj:
            if self.next:
                return self.next.add(obj)
            else:
                self.next = binary_tree(obj, self)
                return True
            return False

    def get_identical(self, obj):
        if obj > self.obj:
            return self.next.get_identical(obj)
        elif obj < self.obj:
            return self.prev.get_identical(obj)
        elif obj == self.obj:
            return self.obj
        else:
            return None

# METAPOST CLASS ----------------------------------------------------------------------------------------------------------------------------------------------

class metapost:
    def __init__(self, filename):
        self.scale = 1.0

        self.f = open(filename, "w")
        self.f.write("beginfig(-1)\n")
        self.f.write("% Scale unit\n")
        if bpy.data.scenes["Scene"].vector_render_size_unit == "CM":
            unit = "cm"
        elif bpy.data.scenes["Scene"].vector_render_size_unit == "MM":
            unit = "mm"
        elif bpy.data.scenes["Scene"].vector_render_size_unit == "PT":
            unit = "pt"
        else:
            unit = "cm"
        self.f.write("u := %i %s;\n" % (bpy.data.scenes["Scene"].vector_render_size, unit))
        self.f.write("% Dash length\n")
        self.f.write("dl := 0.5;\n")
        self.f.write("% Hidden transparency\n")
        self.f.write("tr := 0.8;\n")
        self.f.write("% Face colour\n")
        self.f.write("color fc;\n")
        self.f.write("fc := 0.8 white;\n")

        # Debugging grid
        if False:
            self.f.write("for x = -0.5 step 0.01 until 0.5: draw (x * u, -u / 2)--(x * u, u / 2) withcolor 0.95 white; endfor\n")
            self.f.write("for x = -0.5 step 0.10 until 0.5: draw (x * u, -u / 2)--(x * u, u / 2) withcolor 0.90 white; endfor\n")
            self.f.write("for y = -0.5 step 0.01 until 0.5: draw (-u / 2, y * u)--(u / 2, y * u) withcolor 0.95 white; endfor\n")
            self.f.write("for y = -0.5 step 0.10 until 0.5: draw (-u / 2, y * u)--(u / 2, y * u) withcolor 0.90 white; endfor\n")
            for x in range(-4, 5, 1):
                self.f.write("label(btex %.1f etex, (+0.45 u, %f u));" % (x / 10, x / 10));
                self.f.write("label(btex %.1f etex, (-0.45 u, %f u));" % (x / 10, x / 10));
                self.f.write("label(btex %.1f etex, (%f u, +0.45 u));" % (x / 10, x / 10));
                self.f.write("label(btex %.1f etex, (%f u, -0.45 u));" % (x / 10, x / 10));

    def set_canvas_size(self, xmin, xmax, ymin, ymax):
        self.f.write("pickup pencircle scaled 0 pt;\n");
        self.f.write("draw (%f u, %f u)--(%f u, %f u)--(%f u, %f u)--(%f u, %f u)--cycle withcolor white;\n" % (xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax))
        self.f.write("pickup pencircle scaled 0.5 pt;\n");
    def dotdraw(self, coord):
            self.f.write("pickup pencircle scaled 2 pt;\n")
            self.f.write("drawdot (%f u, %f u);\n" % (coord[0], coord[1]))
            self.f.write("pickup pencircle scaled 0.5 pt;\n")
    def polydraw(self, segs, colour = None, hidden = False):
        if len(segs) < 2:
            return

        self.f.write("draw (%f u, %f u)" % (segs[0][0] * self.scale, segs[0][1] * self.scale))

        if colour and colour[0] == 0.0 and colour[1] == 0.0 and colour[2] == 0.0 and not hidden:
            colour = None

        dashed_str = ""
        if hidden:
            dashed_str = " dashed evenly scaled dl"

        for i in range(1, len(segs)):
            self.f.write("--(%f u, %f u)" % (segs[i][0] * self.scale, segs[i][1] * self.scale))
        if colour:
            #colour = choice(["black", "red", "green", "blue", "red + green", "green + blue", "red + blue"])
            colour = "(%f, %f, %f)" % (colour[0], colour[1], colour[2])
            if not hidden:
                self.f.write(dashed_str + " withcolor %s;\n" % (colour))
            else:
                self.f.write(dashed_str + " withcolor ((1 - tr) * %s + tr * white);\n" % (colour))
        else:
            self.f.write(dashed_str + ";\n" )
    def polyfill(self, segs, colour = None):
        if len(segs) < 2:
            return
        self.f.write("fill (%f u, %f u)" % (segs[0][0], segs[0][1]))
        for i in range(1, len(segs)):
            self.f.write("--(%f u, %f u)" % (segs[i][0], segs[i][1]))
        if not colour:
            colour = "fc"
        else:
            colour = "(%f, %f, %f)" % (colour[0], colour[1], colour[2])
        self.f.write("--cycle withcolor %s;\n" % (colour))
    def label(self, text, pos, ax, ay, rotation):
        suffix = ""
        # Note: Align is reversed in metapost
        if ay == "BOTTOM" or ay == "TOP_BASELINE":
            if ax == "RIGHT":
                suffix = ".ulft"
            elif ax == "CENTER" or "JUSTIFY" or "FLUSH":
                suffix = ".top"
            elif ax == "LEFT":
                suffix = ".urt"
        elif ay == "CENTER":
            if ax == "RIGHT":
                suffix = ".lft"
            elif ax == "CENTER" or "JUSTIFY" or "FLUSH":
                suffix = ""
            elif ax == "LEFT":
                suffix = ".rt"
        elif ay == "TOP":
            if ax == "RIGHT":
                suffix = ".llft"
            elif ax == "CENTER" or "JUSTIFY" or "FLUSH":
                suffix = ".bot"
            elif ax == "LEFT":
                suffix = ".lrt"
        self.f.write("label%s(%s, (0, 0)) rotated %f shifted (%f u, %f u);\n" % (suffix, text, rotation, pos[1] * self.scale, pos[2] * self.scale))
    def set_linewidth(self, val):
        self.f.write("pickup pencircle scaled %f pt;\n" % (val))
    def __del__(self):
        self.f.write("endfig;\nend\n")

# BSP TREE ----------------------------------------------------------------------------------------------------------------------------------------------------

class bsp_node:
    def __init__(self, poly):
        self.node_polys = [poly]
        self.front_polys = None
        self.back_polys = None

    def insert_poly(self, poly):
        p = self.node_polys[0]

        # Determine whether 'poly' is in front or back of the current poly 'p'
        front = False
        back = False
        for v in poly.qq:
            dist = p.point_dist(v * 0.999 + poly.centre * 0.001)
            if dist > +1e-5:
                front = True
            elif dist < -1e-5:
                back = True

        if front and back:
            split = poly.split(self.node_polys[0])
            for fp in split[0]:
                if self.front_polys:
                    self.front_polys.insert_poly(fp)
                else:
                    self.front_polys = bsp_node(fp)
            for bp in split[1]:
                if self.back_polys:
                    self.back_polys.insert_poly(bp)
                else:
                    self.back_polys = bsp_node(bp)
        elif not front and not back:
            self.node_polys.append(poly)
        elif front:
            if self.front_polys:
                self.front_polys.insert_poly(poly)
            else:
                self.front_polys = bsp_node(poly)
        elif back:
            if self.back_polys:
                self.back_polys.insert_poly(poly)
            else:
                self.back_polys = bsp_node(poly)
        else:
            print("This should never happen.")

    def draw(self, proj, mp):
        # Camera position
        cam = proj.cp
        # Main polygon
        p = self.node_polys[0]

        dist = p.point_dist(cam)
        if dist > 0:
            if self.back_polys:
                self.back_polys.draw(proj, mp)
            self.draw_polys(proj, mp)
            if self.front_polys:
                self.front_polys.draw(proj, mp)
        else:
            if self.front_polys:
                self.front_polys.draw(proj, mp)
            self.draw_polys(proj, mp)
            if self.back_polys:
                self.back_polys.draw(proj, mp)

    def draw_polys(self, proj, mp):
        for p in self.node_polys:
            p.project(proj)
            p.draw(mp)

# BOUNDING BOX ------------------------------------------------------------------------------------------------------------------------------------------------

class bounding_box:
    def __init__(self, points):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        for p in points:
            if self.xmin == None or p[1] < self.xmin:
                self.xmin = p[1]
            if self.xmax == None or p[1] > self.xmax:
                self.xmax = p[1]
            if self.ymin == None or p[2] < self.ymin:
                self.ymin = p[2]
            if self.ymax == None or p[2] > self.ymax:
                self.ymax = p[2]
        # Epsilon
        #self.xmin -= 0.001
        #self.xmax += 0.001
        #self.ymin -= 0.001
        #self.ymax += 0.001
    def __str__(self):
        return "%f <= x <= %f; %f <= y <= %f" % (self.xmin, self.xmax, self.ymin, self.ymax)
    def inside(self, point):
        return point[1] > self.xmin and point[1] < self.xmax and point[2] > self.ymin and point[2] < self.ymax
    def overlap(self, other):
        # Determines whether the two bounding boxes overlap
        return not (other.xmin > self.xmax or other.xmax < self.xmin or other.ymin > self.ymax or other.ymax < self.ymin)

# PROJECTOR ---------------------------------------------------------------------------------------------------------------------------------------------------

class projector:
    def __init__(self, camera):
        # Camera mode (perspective / orthographic)
        self.mode = camera.data.type

        # Location
        self.cp = camera.location

        # Direction, left and up-vector
        euler = camera.rotation_euler
        self.rot = euler.to_matrix()
        self.roti = self.rot.inverted()

        # Calculate the scale factor
        if self.mode == "PERSP":
            self.scale = camera.data.lens / camera.data.sensor_width
        else:
            self.scale = 1.0 / camera.data.ortho_scale

    def transform_to_camera_coords(self, vec):
        return self.roti * vec - self.roti * self.cp

    def project(self, v):
        # Convert the vector from global coordinates to camera coordinates
        res = self.transform_to_camera_coords(v)
        # Perform the projection (z, x, -y)
        if self.mode == "PERSP":
            # Perspective mode
            d = res[2] * self.scale
            x = res[0] * self.scale / abs(d)
            y = res[1] * self.scale / abs(d)
            return mathutils.Vector((res.length, x, y))
        elif self.mode == "ORTHO":
            # Orthographic mode
            d = res[2] * self.scale
            x = res[0] * self.scale
            y = res[1] * self.scale
            return mathutils.Vector((res.length, x, y))
        else:
            return res

    def get_canvas_size(self):
        if bpy.context.scene.render.resolution_x > bpy.context.scene.render.resolution_y:
            w = 1.0
            h = w / bpy.context.scene.render.resolution_x * bpy.context.scene.render.resolution_y
        else:
            h = 1.0
            w = h / bpy.context.scene.render.resolution_x * bpy.context.scene.render.resolution_y
        return (-w / 2, w / 2, -h / 2, h / 2)

# EDGE --------------------------------------------------------------------------------------------------------------------------------------------------------

class edge:
    def __init__(self, vertex1, vertex2):
        self.endpoints = None
        for i in range(3):
            if vertex1[i] < vertex2[i]:
                self.endpoints = [vertex1, vertex2]
                break
            elif vertex1[i] > vertex2[i]:
                self.endpoints = [vertex2, vertex1]
                break
        if not self.endpoints:
            self.endpoints = [vertex1, vertex2]
        self.direction = self.endpoints[1] - self.endpoints[0]
        self.endpoints_proj = [None, None]
        self.direction_proj = None
        self.intersections = [0.0, 1.0]
        self.polymember = []
        self.visibility = [1]
        self.colour = [0.0, 0.0, 0.0]
        self.local_edge_angle_limit_cos = None
        self.bbox = None

    def __eq__(self, edge):
        res = False
        res = res or self.endpoints[0] == edge.endpoints[0] and self.endpoints[1] == edge.endpoints[1]
        res = res or self.endpoints[0] == edge.endpoints[1] and self.endpoints[1] == edge.endpoints[0]
        return res

    def __lt__(self, edge):
        i = 0
        while i < 2:
            j = 0
            while j < 3:
                if self.endpoints[i][j] < edge.endpoints[i][j]:
                    return True
                elif  self.endpoints[i][j] > edge.endpoints[i][j]:
                    return False
                j += 1
            i += 1
        return False

    def project(self, p):
        self.endpoints_proj[0] = p.project(self.endpoints[0])
        self.endpoints_proj[1] = p.project(self.endpoints[1])
        self.direction_proj = self.endpoints_proj[1] - self.endpoints_proj[0]
        self.bbox = bounding_box(self.endpoints_proj)

    def add_polygon(self, poly):
        self.polymember.append(poly)

    def intersect(self, edgetree, skip):
        ux = self.direction_proj[1]
        uy = self.direction_proj[2]
        for i in range(skip, len(edgetree)):
            e = edgetree[i]
            # ignore lines with the same direction because they will never cross
            if self.direction_proj == e.direction_proj or self.direction_proj == -e.direction_proj:
                continue
            # If the two bounding boxes do not overlap, do not bother to check for intersection
            if not self.bbox.overlap(e.bbox):
                continue

            vx = e.direction_proj[1]
            vy = e.direction_proj[2]

            dx = e.endpoints_proj[0][1] - self.endpoints_proj[0][1]
            dy = e.endpoints_proj[0][2] - self.endpoints_proj[0][2]

            t1 = (ux * vy - uy * vx)
            if t1 == 0:
                continue
            t1 = (vy * dx - vx * dy) / t1
            #t1 = e.direction_proj[2] * d[0] - e.direction_proj[1] * d[1]
            #t1 /= self.direction_proj[1] * e.direction_proj[2] - self.direction_proj[2] * e.direction_proj[1]

            # test for intersection within this edge
            if t1 <= 0.0 or t1 >= 1.0:
                continue

            #t2 = (t1 * self.direction_proj[2] - d[1]) / e.direction_proj[2]
            if vy == 0:
                continue
            t2 = (t1 * uy - dy) / vy

            # test for intersection within the other edge
            if t2 <= 0.0 or t2 >= 1.0:
                continue

            self.intersections.append(t1)
            e.intersections.append(t2)

    def check_visibility(self, polygon_list, proj, mp):
        # Make an array to indicate the visibility for each segment of the edge
        self.visibility = []
        # Sort the intersections of the edge by position so that segmets are between consecutive indices
        self.intersections.sort()

        # Get the coordinates of the endpoints in camera coordinates
        #a = proj.transform_to_camera_coords(self.endpoints[0])
        #b = proj.transform_to_camera_coords(self.endpoints[1])
        #x1, y1, z1 = a[0], a[1], a[2]
        #x2, y2, z2 = b[0], b[1], b[2]

        i = 0
        N = len(self.intersections) - 1
        # Iterate over the segments
        while i < N:
            # Midpoint of the projected segment
            t = (self.intersections[i] + self.intersections[i + 1]) / 2.0
            midpoint_proj_y = self.endpoints_proj[0][1] + self.direction_proj[1] * t
            midpoint_proj_z = self.endpoints_proj[0][2] + self.direction_proj[2] * t

            # Midpoint of the segment in camera coordinates
            # Note: The midpoint of the 3d segment is usually not the midpoint of the projected segment
            # (The angle bisector in a triangle does not usually bisect the opposing edge in the midpoint.)
            #t2 = (midpoint_proj_y * x1 - y1) / (y2 - y1 - midpoint_proj_y * (x2 - x1))
            # The midpoint of the 3d segment is usually close to the midpoint of the projected segment
            t2 = (self.intersections[i] + self.intersections[i + 1]) / 2.0
            midpoint_x = self.endpoints[0][0] + self.direction[0] * t2
            midpoint_y = self.endpoints[0][1] + self.direction[1] * t2
            midpoint_z = self.endpoints[0][2] + self.direction[2] * t2

            visible = 1
            # Iterate over all polygons
            for p in polygon_list:
                # Omit the polygon that the edge belongs to
                if p in self.polymember:
                    continue
                # Omit the polygons that do not cover the midpoint of the segment
                if not p.inside(midpoint_proj_y, midpoint_proj_z, mp):
                    continue

                # Projected midpoint is inside the (projected) polygon
                #t = (-p.pd - proj.cp.dot(p.nn)) / p.nn.dot(midpoint - proj.cp)
                t3 = p.nn[0] * (midpoint_x - proj.cp[0]) + p.nn[1] * (midpoint_y - proj.cp[1]) + p.nn[2] * (midpoint_z - proj.cp[2])
                t3 = (-p.pd - (proj.cp[0] * p.nn[0] + proj.cp[1] * p.nn[1] + proj.cp[2] * p.nn[2])) / t3
                # Decide whether the midpoint is visible or not
                if t3 < 1.0:
                    visible = 0
                    break
            # Set the visibility of the segment
            self.visibility.append(visible)
            i += 1

    def draw(self, mp, hidden = False):
        #if hidden:
        #    hidden_colour = list(self.colour)
        #    for i in range(3):
        #        hidden_colour[i] = 0.2 * hidden_colour[i] + 0.8 * 1.0

        N = len(self.visibility)
        visibility_state = self.visibility[0]

        i0 = 0
        i = 0
        while i < N:
            i += 1
            # Skip over segments with the same visibility state
            while i < N and self.visibility[i] == visibility_state:
                i += 1

            ta = self.intersections[i0]
            tb = self.intersections[i]
            if visibility_state == True and not hidden:
                mp.polydraw([
                                [self.endpoints_proj[0][1] + ta * self.direction_proj[1], self.endpoints_proj[0][2] + ta * self.direction_proj[2]],
                                [self.endpoints_proj[0][1] + tb * self.direction_proj[1], self.endpoints_proj[0][2] + tb * self.direction_proj[2]]
                            ], self.colour)
            elif visibility_state == False and hidden:
                mp.polydraw([
                                [self.endpoints_proj[0][1] + ta * self.direction_proj[1], self.endpoints_proj[0][2] + ta * self.direction_proj[2]],
                                [self.endpoints_proj[0][1] + tb * self.direction_proj[1], self.endpoints_proj[0][2] + tb * self.direction_proj[2]]
                            ], self.colour, hidden = True)

            # Reset the state
            if i < N:
                visibility_state = self.visibility[i]
                i0 = i

# POLYGON -----------------------------------------------------------------------------------------------------------------------------------------------------

class polygon:
    def __init__(self, vertices, normal, visible_edges = None):
        self.N = len(vertices)
        self.qq = vertices
        self.nn = normal
        self.nn.normalize()
        self.pd = -self.nn.dot(vertices[0])
        self.qp = [None] * self.N
        self.centre = mathutils.Vector([0, 0, 0])
        for v in self.qq:
            self.centre = self.centre + v
        self.centre = self.centre / float(self.N)
        if visible_edges == None:
            self.visible_edges = [1] * self.N
        else:
            self.visible_edges = visible_edges

        # Face shader
        self.shader = None

        # Face colour
        # 0: front, 1: back
        self.colour = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]
        self.front_facing = True
        self.bbox = None

    def __str__(self):
        s = ""
        for i in range(self.N):
            s += str(self.qq[i]) + "\n" + str(self.qp[i]) + "\n\n"
        return s[:-2]

    def project(self, p):
        for i in range(self.N):
            self.qp[i] = p.project(self.qq[i])
        self.centre_proj = p.project(self.centre)
        if self.nn.dot(p.cp - self.centre) < 0:
            self.front_facing = False
        self.bbox = bounding_box(self.qp)

    def point_dist(self, v):
        return self.nn.dot(v) + self.pd

    def inside(self, proj_x, proj_y, mp):
        # Determiness, whether the given point is within the polygon
        # FIXME: this algorithm is not sensitive to special cases

        # Check, whether the point is inside the bounding box
        if not self.bbox.inside(mathutils.Vector((0, proj_x, proj_y))):
            # Not inside the bounding box
            return False

        # Number of intersections
        n = 0
        i = -1
        while i < self.N - 1:
            i += 1
            ii = (i + 1) % self.N

            #ux = self.qp[ii][1] - self.qp[i][1]
            #uy = self.qp[ii][2] - self.qp[i][2]
            #dx = v[1] - self.qp[i][1]
            #dy = v[2] - self.qp[i][2]

            #t1 = (vy * dx - vx * dy) / (ux * vy - uy * vx)
            # vx = 0, vy = 1
            t1 = self.qp[ii][1] - self.qp[i][1]
            if t1 == 0.0:
                continue
            t1 = (proj_x - self.qp[i][1]) / t1

            # test for intersection within this edge
            if t1 >= 0.0 and t1 < 1.0:
                #t2 = (t1 * uy - dy) / vy
                t2 = t1 * (self.qp[ii][2] - self.qp[i][2]) - (proj_y - self.qp[i][2])
                if t2 > 0:
                    n += 1

        return n % 2 == 1

    def split(self, poly):
        # Split this polygon to the plane in which 'poly' lies.
        # Returns a lists of one fragment in front and in the back to the plane (assumes convex polygon)

        # Array of position of the polygon point w.r.t. the splitting plane
        # -1: behind, 0: on plane, +1 in front
        pos = []

        # Number of vertices in front of splitting plane (a), behind (b), and on the plane (c)
        a = 0
        b = 0
        c = 0

        # Determine pos, a, b, c
        for v in self.qq:
            # Calculate distance of vertex from the splitting plane
            dist = poly.point_dist(v)
            if dist > 0:
                # In front
                pos.append(+1)
                a += 1
            elif dist < 0:
                # Behind
                pos.append(-1)
                b += 1
            else:
                # On the plane
                pos.append(0)
                c += 1

        # Trivial cases
        if b == 0:
            # All vertices in front
            return [[self], []]
        if a == 0:
            # All vertices in back
            return [[], [self]]

        # Problematic case: Multiple vertices lie on the plane.
        # This shouldn't happen if the polygon is convex.
        # For now, do not solve this problem, just remove the entire polygon.
        if c > 1:
            print("problem", c, pos, self, poly)
            return [[], []]

        # Find transitions of the polygon edges through the splitting plane
        # Assume convex polygon.
        # s1 and c > 0: on plane vertex index, c == 0: vertex before first splitting plane transition
        # s2: vertex before second splitting plane transition
        s1 = None
        s2 = None
        if c > 0:
            # Find the on-plane vertex
            for i in range(self.N):
                if pos[i] == 0:
                    s1 = i
                    break

            # Find the front-back transition
            s2 = None
            for i in range(s1 + 1, self.N + s1 - 1):
                if pos[i % self.N] != pos[(i + 1) % self.N]:
                    s2 = i
                    break
        else:
            # Find the first transition
            for i in range(self.N - 1):
                if pos[i] != pos[i + 1]:
                    s1 = i
                    break

            # Find the second transition
            for i in range(s1 + 1, self.N):
                if pos[i % self.N] != pos[(i + 1) % self.N]:
                    s2 = i
                    break

        if c > 0:
            v1a = self.qq[s1 % self.N] * 1.0
            v1b = self.qq[s1 % self.N] * 1.0
        else:
            vt1 = self.qq[(s1 + 1) % self.N] - self.qq[s1 % self.N]
            t1 = (-poly.pd - self.qq[s1 % self.N].dot(poly.nn)) / vt1.dot(poly.nn)
            v1a = self.qq[s1 % self.N] + vt1 * t1
            v1b = self.qq[s1 % self.N] + vt1 * t1

        vt2 = self.qq[(s2 + 1) % self.N] - self.qq[s2 % self.N]
        t2 = (-poly.pd - self.qq[s2 % self.N].dot(poly.nn)) / vt2.dot(poly.nn)
        v2a = self.qq[s2 % self.N] + vt2 * t2
        v2b = self.qq[s2 % self.N] + vt2 * t2

        vert_a = []
        vert_b = []
        segs_a = []
        segs_b = []
        if c > 0:
            for i in range(s1, self.N + s1):
                if i == s1:
                    vert_a.append(v1a)
                    vert_b.append(v1b)
                    if pos[(i + 1) % self.N] > 0:
                        segs_a.append(self.visible_edges[i % self.N])
                        segs_b.append(0)
                    else:
                        segs_a.append(0)
                        segs_b.append(self.visible_edges[i % self.N])
                elif i == s2:
                    if pos[i % self.N] > 0:
                        vert_a.append(self.qq[i % self.N] * 1.0)
                        segs_a.append(self.visible_edges[i % self.N])
                        vert_a.append(v2a)
                        segs_a.append(0)
                        vert_b.append(v2b)
                        segs_b.append(self.visible_edges[i % self.N])
                    else:
                        vert_b.append(self.qq[i % self.N] * 1.0)
                        segs_b.append(self.visible_edges[i % self.N])
                        vert_b.append(v2b)
                        segs_b.append(0)
                        vert_a.append(v2a)
                        segs_a.append(self.visible_edges[i % self.N])
                else:
                    if pos[i % self.N] > 0:
                        vert_a.append(self.qq[i % self.N] * 1.0)
                        segs_a.append(self.visible_edges[i % self.N])
                    else:
                        vert_b.append(self.qq[i % self.N] * 1.0)
                        segs_b.append(self.visible_edges[i % self.N])
        else:
            for i in range(s1 + 1, self.N + s1 + 1):
                if i == s1 + 1:
                    vert_a.append(v1a)
                    vert_b.append(v1b)
                    if pos[i % self.N] > 0:
                        segs_a.append(self.visible_edges[(i - 1) % self.N])
                        segs_b.append(0)
                    else:
                        segs_a.append(0)
                        segs_b.append(self.visible_edges[(i - 1) % self.N])
                if pos[i % self.N] > 0:
                    vert_a.append(self.qq[i % self.N] * 1.0)
                    segs_a.append(self.visible_edges[i % self.N])
                else:
                    vert_b.append(self.qq[i % self.N] * 1.0)
                    segs_b.append(self.visible_edges[i % self.N])
                if i == s2:
                    vert_a.append(v2a)
                    vert_b.append(v2b)
                    if pos[i % self.N] > 0:
                        segs_a.append(0)
                        segs_b.append(self.visible_edges[i % self.N])
                    else:
                        segs_a.append(self.visible_edges[i % self.N])
                        segs_b.append(0)

        res = [[polygon(vert_a, self.nn, segs_a)], [polygon(vert_b, self.nn, segs_b)]]
        # Make a copy of the colours
        res[0][0].colour[0] = list(self.colour[0])
        res[0][0].colour[1] = list(self.colour[1])
        res[1][0].colour[0] = list(self.colour[0])
        res[1][0].colour[1] = list(self.colour[1])

        return res

    def set_shader(self, shader):
        if not shader:
            return

        self.shader = shader
        for c in range(3):
            self.colour[0][c] = gamma_correction(shader[2][c])
            self.colour[1][c] = gamma_correction(shader[2][c])

    def set_colour(self, colour):
        # Front face
        self.colour[0][0] = gamma_correction(colour[0])
        self.colour[0][1] = gamma_correction(colour[1])
        self.colour[0][2] = gamma_correction(colour[2])
        # Back face
        self.colour[1][0] = gamma_correction(colour[0])
        self.colour[1][1] = gamma_correction(colour[1])
        self.colour[1][2] = gamma_correction(colour[2])

    def shade(self, lights, camera):
        if self.shader:
            k_ambient = self.shader[1]
            k_diffuse = self.shader[2]
            k_specular = self.shader[3]
            # Divide value by 5 to match it with blender's behaviour
            alpha = self.shader[4] / 5.0
        else:
            k_ambient = [0.0] * 3
            k_diffuse = [0.8] * 3
            k_specular = [0.0] * 3
            alpha = 10.0

        V = camera - self.centre
        V.normalize()

        I = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # Iterate over lights
        for l in lights:
            # Iterate over RGB colors
            for c in range(3):
                # Ambient light, front and back face
                I[0][c] += k_ambient[c] * l.colour[c]
                I[1][c] += k_ambient[c] * l.colour[c]

                # Phong shading model
                # diffuse
                diffuse = -k_diffuse[c] * l.direction.dot(self.nn)
                if diffuse > 0:
                    I[0][c] += +diffuse * l.colour[c]
                elif diffuse < 0:
                    I[1][c] += -diffuse * l.colour[c]

                # specular
                R = self.nn * -2.0 * l.direction.dot(self.nn) + l.direction
                s = R.dot(V)
                if s > 0:
                    specular = k_specular[c] * (s ** alpha) * l.colour[c]
                    if specular > 0:
                        I[0][c] += specular
                else:
                    specular = k_specular[c] * ((-s) ** alpha) * l.colour[c]
                    if specular > 0:
                        I[1][c] += specular

        # Front and back
        for i in range(2):
            # Colour
            for c in range(3):
                # Set the colour
                self.colour[i][c] = I[i][c]

                # Clip the values
                if self.colour[i][c] > 1.0:
                    self.colour[i][c] = 1.0

                # Apply a gamma correction
                self.colour[i][c] = gamma_correction(self.colour[i][c])

    def draw(self, m):
        segs = []
        for i in range(self.N):
            segs.append([self.qp[i][1], self.qp[i][2]])
        segs_closed = []
        for i in range(self.N + 1):
            segs_closed.append([self.qp[i % self.N][1], self.qp[i % self.N][2]])
        if self.front_facing:
            m.polydraw(segs_closed, self.colour[0])
            m.polyfill(segs, self.colour[0])
        else:
            m.polydraw(segs_closed, self.colour[1])
            m.polyfill(segs, self.colour[1])

# LABEL -------------------------------------------------------------------------------------------------------------------------------------------------------

class label():
    def __init__(self, text, position, rotation, align_x, align_y):
        self.text = text
        self.position = position
        self.position_proj = None
        self.align_x = align_x
        self.align_y = align_y
        self.rotation = rotation
    def project(self, p):
        self.position_proj = p.project(self.position)
    def draw(self, m):
        m.label(self.text, self.position_proj, self.align_x, self.align_y, self.rotation)

# OBJECT TRANSFORM --------------------------------------------------------------------------------------------------------------------------------------------

def object_transform(vertex, position, rotation, scale):
    result = mathutils.Vector((scale[0] * vertex[0], scale[1] * vertex[1], scale[2] * vertex[2]))
    result.rotate(rotation)
    result += position
    return result

def object_normal_transform(normal, rotation):
    result = normal.copy()
    result.rotate(rotation)
    return result

# OPERATOR ----------------------------------------------------------------------------------------------------------------------------------------------------

class VectorRender(bpy.types.Operator):
    """Renders the current scene as a vector graphic"""
    bl_idname = "render.vector_render"
    bl_label = "Vector Render"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        scene = context.scene
        # Operator options
        remove_plane_edges = (scene.vector_render_plane_edges == "HIDE")
        edge_angle_limit = degrees(scene.vector_render_plane_edges_angle)
        hidden_line_removal = True
        draw_hidden_lines = False
        if scene.vector_render_hidden_lines == "HIDE":
            hidden_line_removal = True
        elif scene.vector_render_hidden_lines == "SHOW":
            hidden_line_removal = False
        elif scene.vector_render_hidden_lines == "DASH":
            hidden_line_removal = True
            draw_hidden_lines = True
        draw_wireframe = scene.vector_render_draw_edges
        fill_polygons = scene.vector_render_draw_faces
        draw_labels = True
        set_canvas_size = scene.vector_render_canvas_size

        # Get the scene
        # FIXME: use correct scene
        scene = bpy.data.scenes["Scene"]
        # Get the camera
        camera = scene.camera

        # Prepare lists and trees for polygons and edges
        polylist = []
        polytree = None
        edgelist = []
        edgetree = None
        labellist = []

        # Get the mesh data from all visible objects in the scene
        print("Reading geometry ...")
        #boundgeoms = list(mesh.scene.objects('geometry'))
        for object in scene.objects:
            # Filter non-mesh objects
            if not object.type == "MESH":
                continue
            # Filter hidden objects
            if object.hide_render or not object.is_visible(scene):
                continue
            print("- Reading", object.name, "...")
            object_pos = object.location
            object_rot = object.rotation_euler
            object_scl = object.scale
            # Iterate over polygons
            for poly in object.data.polygons:
                # Get the vertex indices
                vertex_indices = poly.vertices
                # Get the coordinates for each vertex
                vertices = []
                for vi in vertex_indices:
                    vertices.append(object_transform(object.data.vertices[vi].co, object_pos, object_rot, object_scl))
                # Polygon normal
                normal = mathutils.Vector(object_normal_transform(poly.normal, object_rot))
                # Create polygon object
                poly_obj = polygon(vertices, normal)
                polylist.append(poly_obj)
                if fill_polygons:
                    # If the polygon has a material, assign the diffuse colour to it. Otherwise the default color is used.
                    if len(object.material_slots) > 0:
                        poly_obj.set_colour(object.material_slots[poly.material_index].material.diffuse_color)
                    #try:
                    #    poly_obj.set_shader(effects[primitive.material.effect.id])
                    #except:
                    #    pass
                    #if use_lights and len(lights) > 0:
                    #    poly_obj.shade(lights, cp)
                    if polytree:
                        polytree.insert_poly(poly_obj)
                    else:
                        polytree = bsp_node(poly_obj)

                # Iterate over the edges of the polygon
                for eg in poly.edge_keys:
                    # Create an edge object
                    e = edge(object_transform(object.data.vertices[eg[0]].co, object_pos, object_rot, object_scl), object_transform(object.data.vertices[eg[1]].co, object_pos, object_rot, object_scl))
                    #object.data.vertices[eg[0]].co, object.data.vertices[eg[1]].co)
                    # Add the edge to the global edge list
                    added = True
                    if edgetree:
                        added = edgetree.add(e)
                    else:
                        edgetree = binary_tree(e)
                    # Add the polygon to the edge object
                    if added:
                        e.add_polygon(poly_obj)
                    else:
                        edgetree.get_identical(e).add_polygon(poly_obj)

        # Get the text data from all visible text objects in the scene
        print("Reading labels ...")
        for object in scene.objects:
            # Filter non-curve objects
            if not object.type == "FONT":
                continue
            # Filter hidden objects
            if object.hide_render or not object.is_visible(scene):
                continue
            # Get the text of the object
            print("- Reading", object.name, "...")
            lb = label(object.data.body, object.location, degrees(object.rotation_euler.x), object.data.align_x, object.data.align_y)
            labellist.append(lb)

        # Test for geometry
        if not edgetree:
            self.report({'ERROR'}, "No geometry to render.")
            return {'CANCELLED'}

        # Make a list of edges
        edgelist_len = 0
        for e in edgetree:
            edgelist.append(e)
            edgelist_len += 1

        p = projector(camera)

        print("Projecting edges ...")
        for e in edgelist:
            e.project(p)

        print("Projecting polygons ...")
        for poly in polylist:
            poly.project(p)

        print("Projecting labels ...")
        for lb in labellist:
            lb.project(p)

        # Remove of optional edges
        if remove_plane_edges and edge_angle_limit == 0.0:
            edge_angle_limit = 0.05
        edge_angle_limit_cos = cos(radians(edge_angle_limit))
        if remove_plane_edges:
            indices_for_removal = []
            i = edgelist_len - 1
            while i >= 0:
                eg = edgelist[i]
                normal = None
                N = len(eg.polymember)
                if N == 2:
                    angle_cos = eg.polymember[0].nn.dot(eg.polymember[1].nn)
                    if eg.local_edge_angle_limit_cos != None:
                        limit = eg.local_edge_angle_limit_cos
                    else:
                        limit = edge_angle_limit_cos
                    if angle_cos > limit:
                        # Ensure that both faces are facing away or toward the camera before removing the edge
                        view_dir = eg.endpoints[0] - p.cp
                        face_dir_a = view_dir.dot(eg.polymember[0].nn)
                        face_dir_b = view_dir.dot(eg.polymember[1].nn)
                        if face_dir_a > 0 and face_dir_b > 0 or face_dir_a < 0 and face_dir_b < 0:
                            indices_for_removal.append(i)
                i -= 1
            for i in indices_for_removal:
                edgelist.pop(i)
                edgelist_len -= 1

        m = metapost(scene.vector_render_file)

        if hidden_line_removal:
            # Determine the edge intersection points
            print("Intersecting lines ...")
            skip = 1
            for e in edgelist:
                e.intersect(edgelist, skip)
                skip += 1

        if hidden_line_removal:
            print("Removing hidden lines ...")
            for e in edgelist:
                e.check_visibility(polylist, p, m)


        print("Drawing ...")

        # Set canvas size
        if set_canvas_size:
            m.set_canvas_size(*p.get_canvas_size())

        # Fill polygons
        if fill_polygons:
            m.set_linewidth(0.05);
            polytree.draw(p, m)
            m.set_linewidth(0.5);

        # Draw hidden lines
        if draw_wireframe and draw_hidden_lines:
            for e in edgelist:
                e.draw(m, hidden = True)

        # Draw visible lines
        if draw_wireframe:
            for e in edgelist:
                e.draw(m, hidden = False)

        # Draw labels
        if draw_labels:
            for lb in labellist:
                lb.draw(m)

        return {'FINISHED'}

show_edge_options = True

def callback_show_edge_options(self, context):
    global show_edge_options
    show_edge_options = self.vector_render_draw_edges

show_face_options = False

def callback_show_face_options(self, context):
    global show_face_options
    show_face_options = self.vector_render_draw_faces

show_plane_edge_options = True

def callback_show_plane_edge_options(self, context):
    global show_plane_edge_options
    show_plane_edge_options = (self.vector_render_plane_edges == "HIDE")

class VectorRenderPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Vector Render"
    bl_idname = "OBJECT_PT_vector_render"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"

    # Output (file) options
    bpy.types.Scene.vector_render_file = bpy.props.StringProperty(name = "", subtype="FILE_PATH")
    bpy.types.Scene.vector_render_size = bpy.props.FloatProperty(name = "Size", default = 10, soft_min = 0, min = 0.001)
    bpy.types.Scene.vector_render_size_unit = bpy.props.EnumProperty(items = [("CM", "cm", "Unit of the render size (centimeters)"),
                                                                              ("MM", "mm", "Unit of the render size (millimeters)"),
                                                                              ("PT", "pt", "Unit of the render size (PostScript points)")], name = "Unit")
    bpy.types.Scene.vector_render_canvas_size = bpy.props.BoolProperty(name = "Force dimensions", default = False)

    # Drawing options

    # Edges
    bpy.types.Scene.vector_render_hidden_lines = bpy.props.EnumProperty(items = [("HIDE", "Hide", "hide"),
                                                                               ("SHOW", "Show", "show"),
                                                                               ("DASH", "Dash", "dash")], default = "HIDE")
    bpy.types.Scene.vector_render_plane_edges = bpy.props.EnumProperty(items = [("SHOW", "Show", "show"), ("HIDE", "Hide", "hide")],
                                                                                default = "HIDE", update = callback_show_plane_edge_options)
    bpy.types.Scene.vector_render_plane_edges_angle = bpy.props.FloatProperty(name = "Angle limit", default = 0.0, soft_min = 0, min = 0, max = pi,
                                                                              soft_max = pi, subtype = "ANGLE", precision = 1, step = 100)
    bpy.types.Scene.vector_render_draw_edges = bpy.props.BoolProperty(name = "Draw edges", default = True, update = callback_show_edge_options)

    # Faces
    bpy.types.Scene.vector_render_draw_faces = bpy.props.BoolProperty(name = "Draw faces", default = False, update = callback_show_face_options)

    def draw(self, context):
        layout = self.layout

        layout.label("Output:")
        layout.prop(context.scene, "vector_render_file")
        row = layout.row(align = True)
        row.prop(context.scene, "vector_render_size")
        row.prop(context.scene, "vector_render_size_unit")
        layout.prop(context.scene, "vector_render_canvas_size")
        layout.separator()

        layout.prop(context.scene, "vector_render_draw_edges")
        if show_edge_options:
            layout.label("Plane edges:")
            buttonrow = layout.row(align = True)
            buttonrow.prop_enum(context.scene, "vector_render_plane_edges", "HIDE")
            buttonrow.prop_enum(context.scene, "vector_render_plane_edges", "SHOW")
            if show_plane_edge_options:
                layout.prop(context.scene, "vector_render_plane_edges_angle")
            layout.label("Obscured edges:")
            buttonrow = layout.row(align = True)
            buttonrow.prop_enum(context.scene, "vector_render_hidden_lines", "HIDE")
            buttonrow.prop_enum(context.scene, "vector_render_hidden_lines", "DASH")
            buttonrow.prop_enum(context.scene, "vector_render_hidden_lines", "SHOW")
            layout.separator()

        layout.prop(context.scene, "vector_render_draw_faces")
        layout.separator()

        layout.operator("render.vector_render")

def register():
    bpy.utils.register_class(VectorRender)
    bpy.utils.register_class(VectorRenderPanel)


def unregister():
    bpy.utils.unregister_class(VectorRender)
    bpy.utils.unregister_class(VectorRenderPanel)


if __name__ == "__main__":
    register()
