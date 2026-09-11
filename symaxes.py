# symaxes.py
#
# ChimeraX command:
#
#     runsymaxes #1
#
# or:
#
#     runsymaxes #1 C5
#
# The first form uses symmetry operations already associated with a map.
# For volume maps, the script will run "measure symmetry" when needed.
#
# For atomic structures, an optional Cn/Dn argument can be supplied.  The
# symmetry axes are then estimated from the inertia tensor:
#   Cn/Dn, n != 2 : unique principal axis
#   D2           : all three principal axes
#
# The important difference from the previous script is that map symmetry
# operations are used to obtain BOTH the axis direction AND a point on the
# actual axis.  The axis is therefore drawn through the real symmetry center,
# rather than from an arbitrary origin.
#
# The script creates a single graphics model named "Symmetry axes".
# Re-running the command replaces the previous one.

import math
import numpy as np

from chimerax.core.commands import CmdDesc, register, ModelArg, StringArg, RestOfLine
from chimerax.core.models import Model
from chimerax.map import Volume
from chimerax.atomic import AtomicStructure
from chimerax.geometry import normalize_vector, Place, rotation, translation, vector_rotation


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def draw_cylinder(session, p1, p2, radius, color, name):
    """Draw a cylinder using ChimeraX's public 'shape cylinder' command.

    ChimeraX 1.10 does not expose a shape_cylinder Python function from
    chimerax.shape, so use the documented command interface instead.
    """
    from chimerax.core.commands import run

    p1s = ",".join(f"{float(x):.6f}" for x in p1)
    p2s = ",".join(f"{float(x):.6f}" for x in p2)

    run(
        session,
        f'shape cylinder fromPoint {p1s} toPoint {p2s} '
        f'radius {float(radius):.6f} caps true '
        f'color {color} name "{name}"'
    )

    # Find the actual surface model created by the command.
    for m in reversed(session.models.list()):
        if getattr(m, "name", None) == name:
            return m

    raise RuntimeError(f"Could not find created cylinder model {name!r}")


def group_symmetry_graphics(session, source_model, graphics, group_name,
                            symmetry=None, axes_scene=None, center_scene=None):
    """Put one source model and all symmetry graphics under a new group.

    Existing SYMAXES behavior is preserved, but the group also records the
    symmetry geometry in scene coordinates.  The saved geometry lets the
    ``symfit`` command rigidly move the complete assembly (map/model +
    cylinder(s) + labels) without having to modify the original map/model
    coordinates.

    Child ordering is:
        .1 source model
        .2 axis 1
        .3 axis 2
        ...
        .5 labels (D2)
    """
    graphics = [m for m in graphics if m is not None]

    group = Model(group_name, session)
    session.models.add([group])
    group.add([source_model] + graphics)

    # Store symmetry metadata in scene coordinates.  These are intentionally
    # private Python attributes on the group; they do not alter the source
    # model or its displayed geometry.
    group._symaxes_symmetry = symmetry
    group._symaxes_axes_scene = (
        [np.asarray(a, dtype=float).copy() for a in axes_scene]
        if axes_scene is not None else None
    )
    group._symaxes_center_scene = (
        np.asarray(center_scene, dtype=float).copy()
        if center_scene is not None else None
    )

    session.logger.info(
        f"Created symmetry group {group.name} (#{group.id_string}); "
        f"children: " + ", ".join(
            f"{m.name} #{m.id_string}"
            for m in [source_model] + graphics
        )
    )
    return group


def _axis_from_cylinder(cylinder):
    """Recover a cylinder's scene-space axis by PCA of its vertices.

    Used only as a backwards-compatible fallback for groups created by an
    earlier symaxes.py version that did not store symmetry metadata.
    """
    vertices = getattr(cylinder, "vertices", None)
    if vertices is None or len(vertices) < 3:
        b = cylinder.bounds()
        if b is None:
            raise ValueError(f"Cannot recover axis from {cylinder.name}")
        d = np.asarray(b.xyz_max) - np.asarray(b.xyz_min)
        axis = np.zeros(3)
        axis[int(np.argmax(np.abs(d)))] = 1.0
        return unique_direction(axis)

    xyz = np.asarray(cylinder.scene_position.transform_points(vertices), dtype=float)
    xyz -= xyz.mean(axis=0)

    cov = xyz.T @ xyz
    vals, vecs = np.linalg.eigh(cov)
    return unique_direction(vecs[:, np.argmax(vals)])


def _get_group_symmetry_data(session, group):
    """Return live (symmetry, axes_scene, center_scene) for a SYMAXES group.

    We intentionally recover the axis directions from the CURRENT cylinder
    geometry every time.  This is important if the user manually rotates or
    translates an assembly between symfit calls: cached axis metadata would
    otherwise describe the old orientation.
    """
    import re

    symmetry = getattr(group, "_symaxes_symmetry", None)

    if symmetry is None:
        match = re.search(r'\b(C\d+|D\d+)\b', group.name)
        if not match:
            raise ValueError(
                f"{group.name} is not a SYMAXES group with recognizable Cn/Dn symmetry."
            )
        symmetry = match.group(1).upper()

    children = list(group.child_models())
    axis_children = [
        m for m in children
        if "axis" in getattr(m, "name", "").lower()
        and not getattr(m, "name", "").lower().endswith("labels")
    ]

    def axis_number(m):
        mm = re.search(r'axis\s+(\d+)', m.name, re.IGNORECASE)
        return int(mm.group(1)) if mm else 999

    axis_children.sort(key=axis_number)

    if not axis_children:
        raise ValueError(f"No symmetry-axis cylinders found in {group.name}.")

    # These directions come from the cylinders as they EXIST NOW, so manual
    # rotations of the complete group are automatically respected.
    axes = [_axis_from_cylinder(m) for m in axis_children]

    if symmetry == "D2" and len(axes) != 3:
        raise ValueError(
            f"{group.name} is D2 but has {len(axes)} axis cylinders; expected 3."
        )

    # Use the current common intersection of the D2 cylinders when possible.
    if symmetry == "D2":
        points = []
        for cyl in axis_children:
            try:
                points.append(np.asarray(cyl.bounds().center(), dtype=float))
            except Exception:
                points.append(None)

        if all(p is not None for p in points):
            I = np.eye(3)
            A = np.zeros((3, 3), dtype=float)
            b = np.zeros(3, dtype=float)

            for u, pnt in zip(axes, points):
                u = normalize_vector(np.asarray(u, dtype=float))
                P = I - np.outer(u, u)
                A += P
                b += P @ pnt

            center, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            if rank < 3:
                center = None
        else:
            center = None
    else:
        center = None

    # For Cn / Dn, n>2, recover a point from the CURRENT principal-axis
    # cylinder itself.  This is essential for exact line colinearity:
    # matching directions alone is not enough, and the map bounding-box
    # center need not lie exactly on the rendered symmetry line.
    if center is None:
        if symmetry != "D2" and axis_children:
            try:
                center = np.asarray(axis_children[0].bounds().center(), dtype=float)
            except Exception:
                center = None

        if center is None:
            source = children[0]
            try:
                center = np.asarray(source.bounds().center(), dtype=float)
            except Exception:
                center = np.asarray(group.bounds().center(), dtype=float)

    # Keep metadata synchronized as a convenience, but never use it instead
    # of the live cylinder geometry on the next call.
    group._symaxes_symmetry = symmetry
    group._symaxes_axes_scene = [a.copy() for a in axes]
    group._symaxes_center_scene = np.asarray(center, dtype=float).copy()

    return symmetry, axes, np.asarray(center, dtype=float)



def _resolve_model(session, spec):
    """Resolve an exact ChimeraX model/group ID such as #8."""
    spec = spec.strip()
    if not spec.startswith("#"):
        raise ValueError(f"Model/group must be specified as a model ID, got {spec!r}.")

    target_id = spec[1:]
    matches = [
        m for m in session.models.list()
        if getattr(m, "id_string", "") == target_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely resolve model/group {spec}.")
    return matches[0]


def _orthogonal_unit(v, against):
    v = np.asarray(v, dtype=float)
    against = normalize_vector(np.asarray(against, dtype=float))
    v = v - against * np.dot(v, against)
    return normalize_vector(v)


def _rotation_matrix_between_vectors(source_axis, target_axis):
    """Return a numerically stable 3x3 rotation mapping source_axis -> target_axis."""
    a = normalize_vector(np.asarray(source_axis, dtype=float))
    b = normalize_vector(np.asarray(target_axis, dtype=float))

    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))

    # Already parallel.
    if s < 1e-12 and c > 0:
        return np.eye(3)

    # Anti-parallel: choose a deterministic axis perpendicular to a and
    # perform an exact 180 degree rotation.
    if s < 1e-12 and c < 0:
        if abs(a[0]) < 0.8:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        u = normalize_vector(np.cross(a, ref))

        # R = 2 uu^T - I for a 180-degree rotation about u.
        return 2.0 * np.outer(u, u) - np.eye(3)

    # Rodrigues rotation formula.
    vx = np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])

    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

    # Remove accumulated floating-point drift so R is as exactly orthogonal
    # as possible and has determinant +1.
    u, _, vh = np.linalg.svd(R)
    R = u @ vh
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vh

    return R


def _align_two_vectors_transform(source_axis, target_axis, source_center, target_center):
    """Rigidly map one principal axis exactly onto another and centers together.

    This replaces the previous vector_rotation()/axis-center construction with
    an explicit Rodrigues rotation matrix.  The resulting transformed source
    axis is numerically colinear with the target axis to machine precision.
    """
    source_axis = normalize_vector(np.asarray(source_axis, dtype=float))
    target_axis = normalize_vector(np.asarray(target_axis, dtype=float))

    src_center = np.asarray(source_center, dtype=float)
    tgt_center = np.asarray(target_center, dtype=float)

    R = _rotation_matrix_between_vectors(source_axis, target_axis)

    # Rotate about the source center, then translate the center to target.
    rotated_center = R @ src_center
    trans = tgt_center - rotated_center

    matrix = np.column_stack([R, trans])
    return Place(matrix=matrix)



def _signed_rotation_about_axis(u, v, axis):
    """Signed angle (degrees) rotating u toward v about axis."""
    axis = normalize_vector(np.asarray(axis, dtype=float))
    u = _orthogonal_unit(u, axis)
    v = _orthogonal_unit(v, axis)

    s = float(np.dot(axis, np.cross(u, v)))
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.atan2(s, c))


def _d2_frame_transform(source_axes, target_axes, source_index, target_index,
                        source_center, target_center):
    """Fit D2 with both the requested axis pair and the correct twist.

    The requested C2 axes are first made colinear.  We then determine the
    rotation about that common axis from the other two C2 axes.  Because D2
    axes are unoriented lines, the two possible assignments of the remaining
    axes, and both signs of their target directions, are tested.  The solution
    requiring the smallest additional twist about the selected axis is chosen.

    This is deliberately based on the *current* source-axis geometry, so a
    manual rotation of the source group before rerunning symfit is respected.
    """
    s = [normalize_vector(np.asarray(a, dtype=float)) for a in source_axes]
    t = [normalize_vector(np.asarray(a, dtype=float)) for a in target_axes]

    si = source_index
    ti = target_index

    # Stage 1: exactly align the selected source axis to the selected target
    # axis and move the assembly center onto the target center.
    r0 = _align_two_vectors_transform(
        s[si], t[ti], source_center, target_center
    )

    s_after = [
        normalize_vector(r0.apply_without_translation(a))
        for a in s
    ]

    src_remaining = [i for i in range(3) if i != si]
    tgt_remaining = [i for i in range(3) if i != ti]

    target_axis = t[ti]

    candidates = []

    # Two possible pairings of the remaining D2 axes.
    pairings = [
        (tgt_remaining[0], tgt_remaining[1]),
        (tgt_remaining[1], tgt_remaining[0]),
    ]

    for ts1, ts2 in pairings:
        u = s_after[src_remaining[0]]

        # Each target D2 axis is a line, so +/-v is equivalent physically.
        for sign in (1.0, -1.0):
            v = sign * t[ts1]
            theta = _signed_rotation_about_axis(u, v, target_axis)

            twist = rotation(target_axis, theta, center=target_center)

            # Verify the first remaining axis after the candidate twist.
            u1 = normalize_vector(twist.apply_without_translation(u))
            err1 = _check_axis_colinear(u1, t[ts1])

            # Also verify the second remaining axis.  This is the important
            # check that rejects the wrong twist/permutation.
            u2 = normalize_vector(
                twist.apply_without_translation(
                    s_after[src_remaining[1]]
                )
            )
            err2 = _check_axis_colinear(u2, t[ts2])

            # Prefer exact geometric fits, then the smallest twist.
            candidates.append((max(err1, err2), abs(theta), theta, twist))

    # Select the candidate that actually puts BOTH remaining axes on their
    # target axes.  In a good D2 assembly the residuals should be ~0.
    candidates.sort(key=lambda x: (x[0], x[1]))
    best_err, twist_amount, theta, twist = candidates[0]

    if best_err > 1e-4:
        raise ValueError(
            f"Could not uniquely fit the D2 frame: best residual "
            f"is {best_err:.9g} degrees."
        )

    total = twist * r0
    return total, math.copysign(twist_amount, theta), candidates



def _update_group_axis_metadata(group, transform):
    """Update scene-space metadata after moving the whole assembly."""
    axes = getattr(group, "_symaxes_axes_scene", None)
    center = getattr(group, "_symaxes_center_scene", None)

    if axes is None or center is None:
        return

    group._symaxes_axes_scene = [
        normalize_vector(transform.apply_without_translation(a))
        for a in axes
    ]
    group._symaxes_center_scene = np.asarray(transform * center, dtype=float)


def _check_axis_colinear(a, b, tolerance=1e-7):
    """Return angular residual in degrees for two unoriented axes."""
    a = normalize_vector(np.asarray(a, dtype=float))
    b = normalize_vector(np.asarray(b, dtype=float))
    c = np.clip(abs(np.dot(a, b)), -1.0, 1.0)
    return math.degrees(math.acos(c))


def symfit(session, specification):
    """Align two complete SYMAXES assemblies.

    Syntax:
        symfit #8 into #7
        symfit #8 ax 1 into #7 ax 3

    The latter form is for D2 and chooses which C2 axes become colinear.
    """
    import re

    text = specification.strip()

    m = re.match(
        r'^(\#\S+)(?:\s+ax\s+([123]))?\s+into\s+(\#\S+)(?:\s+ax\s+([123]))?$',
        text,
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(
            "Usage: symfit #source into #target "
            "or: symfit #source ax 1 into #target ax 3"
        )

    source = _resolve_model(session, m.group(1))
    target = _resolve_model(session, m.group(3))

    if source is target:
        raise ValueError("Source and target assemblies must be different.")

    source_axis_no = int(m.group(2)) if m.group(2) else None
    target_axis_no = int(m.group(4)) if m.group(4) else None

    source_sym, source_axes, source_center = _get_group_symmetry_data(session, source)
    target_sym, target_axes, target_center = _get_group_symmetry_data(session, target)

    if source_sym != target_sym:
        raise ValueError(
            f"Assemblies have different symmetry: {source_sym} vs {target_sym}."
        )

    if source_sym == "D2":
        if source_axis_no is None or target_axis_no is None:
            raise ValueError(
                "D2 requires an explicit axis pair, e.g. "
                "'symfit #8 ax 1 into #7 ax 3'."
            )

        if len(source_axes) != 3 or len(target_axes) != 3:
            raise ValueError("D2 assemblies must have exactly three axes.")

        transform, twist_degrees, d2_candidates = _d2_frame_transform(
            source_axes,
            target_axes,
            source_axis_no - 1,
            target_axis_no - 1,
            source_center,
            target_center,
        )

        mode = (
            f"D2 axis {source_axis_no} -> axis {target_axis_no}; "
            f"twist about fitted axis = {twist_degrees:.6f} deg"
        )

    else:
        if source_axis_no is not None or target_axis_no is not None:
            raise ValueError(
                f"{source_sym} uses its principal axis automatically; "
                "the 'ax N' syntax is only for D2."
            )

        # For Cn and Dn (n > 2), axis 1 is the sole/principal axis.
        if not source_axes or not target_axes:
            raise ValueError("Could not recover principal symmetry axes.")

        transform = _align_two_vectors_transform(
            source_axes[0],
            target_axes[0],
            source_center,
            target_center,
        )

        mode = f"{source_sym} principal axis"

    # Move the entire source group, not the individual map/model or graphics.
    old_position = source.scene_position
    source.scene_position = transform * old_position

    _update_group_axis_metadata(source, transform)

    # For Cn / Dn (n != 2), enforce coincidence of the INFINITE principal
    # axis lines after the orientation step.  A tiny center discrepancy can
    # leave two perfectly parallel axes laterally displaced, which is visible
    # even when the angular residual is exactly zero.
    if source_sym != "D2":
        _, live_source_axes, live_source_center = _get_group_symmetry_data(
            session, source
        )
        _, live_target_axes, live_target_center = _get_group_symmetry_data(
            session, target
        )

        u = normalize_vector(np.asarray(live_target_axes[0], dtype=float))
        ps = np.asarray(live_source_center, dtype=float)
        pt = np.asarray(live_target_center, dtype=float)

        # Only remove the component perpendicular to the common axis.
        # Translation along the axis is irrelevant to line colinearity.
        delta = pt - ps
        perpendicular_delta = delta - u * np.dot(delta, u)

        if np.linalg.norm(perpendicular_delta) > 0:
            correction = translation(perpendicular_delta)
            source.scene_position = correction * source.scene_position
            _update_group_axis_metadata(source, correction)

    # Quantify the result.
    new_sym, new_axes, new_center = _get_group_symmetry_data(session, source)

    residual = []
    if source_sym == "D2":
        # Verify all three D2 lines, using the user's explicit pairing for the
        # selected axis and the best corresponding pairing for the other two.
        si = source_axis_no - 1
        ti = target_axis_no - 1

        residual.append(
            _check_axis_colinear(new_axes[si], target_axes[ti])
        )

        srem = [i for i in range(3) if i != si]
        trem = [i for i in range(3) if i != ti]

        e_a = max(
            _check_axis_colinear(new_axes[srem[0]], target_axes[trem[0]]),
            _check_axis_colinear(new_axes[srem[1]], target_axes[trem[1]]),
        )
        e_b = max(
            _check_axis_colinear(new_axes[srem[0]], target_axes[trem[1]]),
            _check_axis_colinear(new_axes[srem[1]], target_axes[trem[0]]),
        )
        residual.append(min(e_a, e_b))
    else:
        residual.append(_check_axis_colinear(new_axes[0], target_axes[0]))

    center_error = float(np.linalg.norm(new_center - target_center))

    if source_sym != "D2":
        u = normalize_vector(np.asarray(target_axes[0], dtype=float))
        line_delta = np.asarray(new_center) - np.asarray(target_center)
        line_offset = float(
            np.linalg.norm(line_delta - u * np.dot(line_delta, u))
        )
    else:
        line_offset = 0.0

    session.logger.info(
        f"symfit: moved {source.name} (#{source.id_string}) into "
        f"{target.name} (#{target.id_string}); {mode}"
    )
    if source_sym == "D2":
        session.logger.info(
            f"symfit residual: selected-axis error = {residual[0]:.9g} deg; "
            f"remaining-axis error = {residual[1]:.9g} deg; "
            f"center error = {center_error:.9g} A"
        )
    else:
        session.logger.info(
            f"symfit residual: axis angular error = {residual[0]:.12g} deg; "
            f"principal-line offset = {line_offset:.12g} A; "
            f"center error = {center_error:.12g} A"
        )
        if source_sym != "D2":
            session.logger.info(
                "symfit principal axes: source="
                f"({new_axes[0][0]:.12g},{new_axes[0][1]:.12g},{new_axes[0][2]:.12g}) "
                "target="
                f"({target_axes[0][0]:.12g},{target_axes[0][1]:.12g},{target_axes[0][2]:.12g})"
            )

    if source_sym.startswith("D") and source_sym != "D2":
        session.logger.info(
            "symfit: principal axes are colinear, therefore the "
            "non-principal C2 axes are in the same plane perpendicular "
            "to the principal axis."
        )

    return


def unique_direction(v, tol=1e-5):
    """Normalize an axis and choose a reproducible sign."""
    v = normalize_vector(np.asarray(v, dtype=float))
    # Axis direction is unoriented, so +v and -v are equivalent.
    for x in v:
        if abs(x) > tol:
            if x < 0:
                v = -v
            break
    return v


def angle_distance(a, b):
    """Smallest difference between two angles in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def direction_distance(a, b):
    """Distance between unoriented unit vectors."""
    return min(np.linalg.norm(a - b),
               np.linalg.norm(a + b))


def deduplicate_axes(axes, angle_tol=0.5, direction_tol=1e-4):
    """Merge symmetry operations that have the same rotation axis."""
    unique = []

    for axis, point, angle, shift in axes:
        axis = unique_direction(axis)

        found = False
        for u in unique:
            if direction_distance(axis, u["axis"]) < direction_tol:
                found = True
                break

        if not found:
            unique.append({
                "axis": axis,
                "point": np.asarray(point, dtype=float),
                "angle": angle,
                "shift": shift,
            })

    return unique


def model_extent(model):
    """Return a useful characteristic size in scene coordinates.

    For volumes, use the full 3-D bounding-box diagonal rather than only
    the X extent.  For atomic models, use the same scene-space bounds.
    The symmetry axis is then made only slightly longer than the object.
    """
    try:
        b = model.bounds()
        if b is not None:
            d = np.asarray(b.xyz_max, dtype=float) - np.asarray(b.xyz_min, dtype=float)
            diagonal = float(np.linalg.norm(d))
            if diagonal > 0:
                return diagonal
    except Exception:
        pass

    try:
        xyz = np.asarray(model.atoms.scene_coords, dtype=float)
        if len(xyz):
            d = xyz.max(axis=0) - xyz.min(axis=0)
            diagonal = float(np.linalg.norm(d))
            if diagonal > 0:
                return diagonal
    except Exception:
        pass

    try:
        xyz_min, xyz_max = model.xyz_bounds()
        d = np.asarray(xyz_max, dtype=float) - np.asarray(xyz_min, dtype=float)
        diagonal = float(np.linalg.norm(d))
        if diagonal > 0:
            return diagonal
    except Exception:
        pass

    return 100.0


def axis_length(model):
    """Axis length: 20% longer than the model/map diagonal."""
    return max(10.0, 1.20 * model_extent(model))


def model_center_scene(model):
    """Return the center of the model's displayed scene-space bounds."""
    try:
        b = model.bounds()
        if b is not None:
            return np.asarray(b.center(), dtype=float)
    except Exception:
        pass

    try:
        xyz = np.asarray(model.atoms.scene_coords, dtype=float)
        if len(xyz):
            return 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
    except Exception:
        pass

    raise ValueError("Could not determine model center in scene coordinates.")


def move_point_to_model_center(point, direction, center):
    """
    Keep the symmetry direction exactly as measured, but choose the point on
    that infinite axis that is closest to the model center.

    This is important because axis_center_angle_shift() returns an arbitrary
    point on the rotation axis; it is not guaranteed to be the point through
    which we want to render the finite cylinder.  For a correctly symmetric
    map/model, the true symmetry center lies on this line, so this projection
    removes any harmless offset in the returned axis representative.
    """
    p = np.asarray(point, dtype=float)
    u = normalize_vector(np.asarray(direction, dtype=float))
    c = np.asarray(center, dtype=float)
    return p + u * np.dot(c - p, u)


def scene_point(volume, local_point):
    """Convert a map-local physical coordinate into scene coordinates."""
    return volume.scene_position * np.asarray(local_point, dtype=float)


def scene_vector(volume, local_vector):
    """Convert a map-local vector into a scene vector."""
    return normalize_vector(
        volume.scene_position.apply_without_translation(
            np.asarray(local_vector, dtype=float)
        )
    )


# ---------------------------------------------------------------------------
# Map symmetry
# ---------------------------------------------------------------------------

def ensure_map_symmetry(session, volume):
    """
    Return volume.data.symmetries.

    If no symmetry transformations are assigned, try automatic detection.
    """
    syms = getattr(volume.data, "symmetries", None)
    if syms is not None and len(syms) > 1:
        return syms

    # ChimeraX's automatic map-symmetry search assigns the transformations
    # back to volume.data.symmetries when it succeeds.
    session.logger.info(
        f"No assigned symmetry operations found for {volume.name}; "
        f"running: measure symmetry #{volume.id_string} set true"
    )

    from chimerax.core.commands import run
    run(session, f"measure symmetry #{volume.id_string} set true")

    syms = getattr(volume.data, "symmetries", None)
    if syms is None or len(syms) <= 1:
        raise ValueError(
            f"Could not obtain symmetry operations for map #{volume.id_string}. "
            f"Use 'measure symmetry #{volume.id_string}' first, or assign "
            f"the symmetry with 'volume #{volume.id_string} symmetry ...'."
        )

    return syms


def classify_map_symmetry(session, volume):
    """
    Inspect actual symmetry transforms and identify Cn / Dn.

    Each symmetry operation is a rigid Place transform.  Place provides
    axis_center_angle_shift(), which directly gives:
        axis, point_on_axis, rotation_angle, axial_shift

    For an ordinary Cn/Dn point-group symmetry, axial_shift should be ~0.
    """
    syms = ensure_map_symmetry(session, volume)

    operations = []

    for p in syms:
        try:
            axis, point, angle, shift = p.axis_center_angle_shift()
        except Exception:
            continue

        if p.is_identity():
            continue

        # We only want point-group rotations, not screw/helical operations.
        if abs(shift) > 1e-3:
            continue

        # Ignore numerical zero-angle transforms.
        if angle < 1e-3:
            continue

        operations.append((
            np.asarray(axis, dtype=float),
            np.asarray(point, dtype=float),
            float(angle),  # ChimeraX axis_center_angle_shift() returns degrees
            float(shift),
        ))

    if not operations:
        raise ValueError("No usable rotational symmetry operations found.")

    axes = deduplicate_axes(operations)

    # Useful diagnostic information: actual rotation angles and axis points.
    # This makes coordinate-system issues easy to diagnose from the Log.
    for j, op in enumerate(operations, start=1):
        a, pnt, ang, sh = op
        session.logger.info(
            f"symop {j}: angle={ang:.6f} deg, "
            f"axis=({a[0]:.6f},{a[1]:.6f},{a[2]:.6f}), "
            f"point=({pnt[0]:.6f},{pnt[1]:.6f},{pnt[2]:.6f}), "
            f"shift={sh:.6f}"
        )

    # Determine the largest rotational order present.
    # For a Cn/Dn group, the principal order is the largest order.
    orders = []
    for op in operations:
        angle = abs(op[2])
        order = int(round(360.0 / angle))
        if order >= 2 and abs(360.0 / order - angle) < 1.0:
            orders.append(order)

    if not orders:
        raise ValueError("Could not determine rotational symmetry order.")

    n = max(orders)

    # A Cn group has one unique axis.
    # A Dn group has the principal axis plus n perpendicular 2-fold axes.
    #
    # The D2 special case has exactly three 2-fold axes.
    if len(axes) == 1:
        symmetry = f"C{n}"
    else:
        # If there are multiple axes and n > 2, we expect a Dn group.
        symmetry = f"D{n}"

    return symmetry, operations, axes


def draw_map_axes(session, volume, color="yellow"):
    """
    Draw the requested Cn/Dn axes directly from map symmetry operations.
    """
    symmetry, operations, axes = classify_map_symmetry(session, volume)

    length = axis_length(volume)
    radius = max(0.5, 0.012 * length)

    if symmetry.startswith("C"):
        # All non-identity rotations share the Cn axis.
        axis_info = axes[0]

        axis_local = axis_info["axis"]
        point_local = axis_info["point"]

        # Transform both the point and direction into the scene frame.
        point_scene = scene_point(volume, point_local)
        axis_scene = scene_vector(volume, axis_local)

        # Use the actual map center as the anchor on the infinite symmetry
        # axis.  This prevents an arbitrary representative point returned by
        # axis_center_angle_shift() from making the rendered cylinder appear
        # laterally displaced from the map.
        map_center_scene = model_center_scene(volume)
        point_scene = move_point_to_model_center(
            point_scene, axis_scene, map_center_scene
        )

        p1 = point_scene - 0.5 * length * axis_scene
        p2 = point_scene + 0.5 * length * axis_scene

        s = draw_cylinder(
            session, p1, p2, radius, color,
            f"SYMAXES: #{volume.id_string} {symmetry} symmetry axis"
        )

        offset = np.linalg.norm(point_scene - map_center_scene)
        session.logger.info(
            f"{symmetry}: rendered axis anchored at nearest point to model center; "
            f"axis-center offset = {offset:.6f} A. axis = "
            f"({axis_scene[0]:.6f}, {axis_scene[1]:.6f}, {axis_scene[2]:.6f}), "
            f"point = "
            f"({point_scene[0]:.6f}, {point_scene[1]:.6f}, {point_scene[2]:.6f})"
        )

        group_symmetry_graphics(
            session, volume, [s],
            f"SYMAXES: #{volume.id_string} {symmetry}",
            symmetry=symmetry,
            axes_scene=[axis_scene],
            center_scene=point_scene,
        )

        return [s], symmetry

    # Dn
    if symmetry == "D2":
        # All three actual 2-fold axes are retained.
        axis_infos = axes

        # Sort into a stable order.  The longest/most unique axis ordering is
        # not physically meaningful for D2, so labels are simply assigned
        # reproducibly from the transformed scene vectors.
        axis_infos = sorted(
            axis_infos,
            key=lambda a: tuple(np.round(scene_vector(volume, a["axis"]), 6))
        )

        # Use three visually distinct colors.
        colors = ["yellow", "cyan", "magenta"]

        surfaces = []

        # IMPORTANT: axis_center_angle_shift() returns an arbitrary point
        # lying on each rotation axis.  For D2 those three points are generally
        # different, so averaging them does NOT give the common D2 center.
        #
        # Find the least-squares intersection of the three infinite lines
        #     p_i + t * u_i
        # by minimizing the squared perpendicular distance to all axes:
        #     sum ||(I - u_i u_i^T)(c - p_i)||^2.
        I = np.eye(3)
        A = np.zeros((3, 3), dtype=float)
        b = np.zeros(3, dtype=float)
        for a in axis_infos:
            u = normalize_vector(np.asarray(a["axis"], dtype=float))
            p = np.asarray(a["point"], dtype=float)
            P = I - np.outer(u, u)
            A += P
            b += P @ p

        center_local, residuals, rank, singular_values = np.linalg.lstsq(
            A, b, rcond=None
        )
        if rank < 3:
            raise ValueError(
                "Could not determine the common D2 symmetry center "
                "from the three rotation axes."
            )

        # Report how well the recovered point lies on each axis.
        axis_offsets = []
        for a in axis_infos:
            u = normalize_vector(np.asarray(a["axis"], dtype=float))
            p = np.asarray(a["point"], dtype=float)
            d = np.linalg.norm(np.cross(center_local - p, u))
            axis_offsets.append(float(d))

        session.logger.info(
            "D2 recovered local symmetry center = "
            f"({center_local[0]:.6f}, {center_local[1]:.6f}, "
            f"{center_local[2]:.6f}); "
            "distance to axes = "
            + ", ".join(f"{d:.6g}" for d in axis_offsets)
        )

        center_scene = scene_point(volume, center_local)

        for i, (axis_info, c) in enumerate(zip(axis_infos, colors), start=1):
            axis_scene = scene_vector(volume, axis_info["axis"])

            p1 = center_scene - 0.5 * length * axis_scene
            p2 = center_scene + 0.5 * length * axis_scene

            s = draw_cylinder(
                session, p1, p2, radius, c,
                f"SYMAXES: #{volume.id_string} D2 axis {i}"
            )
            surfaces.append(s)

            session.logger.info(
                f"D2 axis {i}: "
                f"direction = "
                f"({axis_scene[0]:.6f}, {axis_scene[1]:.6f}, {axis_scene[2]:.6f})"
            )

        # Put the labels at the actual positive ends of the axes.
        labels = create_label_markers(
            session,
            center_scene,
            [scene_vector(volume, a["axis"]) for a in axis_infos],
            colors,
            length,
            name=f"SYMAXES: #{volume.id_string} D2 labels",
        )

        group_symmetry_graphics(
            session, volume, surfaces + [labels],
            f"SYMAXES: #{volume.id_string} D2",
            symmetry="D2",
            axes_scene=[
                scene_vector(volume, a["axis"]) for a in axis_infos
            ],
            center_scene=center_scene,
        )

        session.logger.info(
            f"D2: all three 2-fold axes drawn through symmetry center "
            f"({center_scene[0]:.6f}, {center_scene[1]:.6f}, "
            f"{center_scene[2]:.6f})"
        )

        return surfaces, symmetry

    # Dn, n > 2: draw the principal n-fold axis.
    # Select the axis associated with the highest rotation order.
    principal = None
    best_order = -1

    for axis_info in axes:
        order = int(round(360.0 / abs(axis_info["angle"])))
        if order > best_order:
            best_order = order
            principal = axis_info

    point_scene = scene_point(volume, principal["point"])
    axis_scene = scene_vector(volume, principal["axis"])

    map_center_scene = model_center_scene(volume)
    point_scene = move_point_to_model_center(
        point_scene, axis_scene, map_center_scene
    )

    p1 = point_scene - 0.5 * length * axis_scene
    p2 = point_scene + 0.5 * length * axis_scene

    s = draw_cylinder(
        session, p1, p2, radius, color,
        f"SYMAXES: #{volume.id_string} {symmetry} principal symmetry axis"
    )

    offset = np.linalg.norm(point_scene - map_center_scene)
    session.logger.info(
        f"{symmetry}: principal axis anchored at nearest point to model center; "
        f"axis-center offset = {offset:.6f} A. axis = "
        f"({axis_scene[0]:.6f}, {axis_scene[1]:.6f}, {axis_scene[2]:.6f}), "
        f"point = "
        f"({point_scene[0]:.6f}, {point_scene[1]:.6f}, {point_scene[2]:.6f})"
    )

    group_symmetry_graphics(
        session, volume, [s],
        f"SYMAXES: #{volume.id_string} {symmetry}",
        symmetry=symmetry,
        axes_scene=[axis_scene],
        center_scene=point_scene,
    )

    return [s], symmetry


# ---------------------------------------------------------------------------
# Atomic structures
# ---------------------------------------------------------------------------

def inertia_axes(structure):
    """
    Return center, principal axes, and principal values.

    For an exact Cn/Dn object, the two transverse second moments are
    approximately equal.  The remaining eigenvector is therefore the
    principal Cn/Dn axis.
    """
    xyz = np.asarray(structure.atoms.scene_coords, dtype=float)

    if len(xyz) < 3:
        raise ValueError("At least three atoms are needed.")

    try:
        masses = np.asarray(structure.atoms.masses, dtype=float)
        if masses.shape != (len(xyz),):
            raise ValueError
        masses = np.where(masses > 0, masses, 1.0)
    except Exception:
        masses = np.ones(len(xyz), dtype=float)

    center = np.average(xyz, axis=0, weights=masses)
    rel = xyz - center

    cov = np.einsum("i,ij,ik->jk", masses, rel, rel)
    vals, vecs = np.linalg.eigh(cov)

    axes = [unique_direction(vecs[:, i]) for i in range(3)]
    return center, axes, vals


def create_label_markers(session, center, axes, colors, length, name="SYMAXES: labels"):
    """
    Create a marker set and put a 3-D label at the positive end of each
    supplied axis.
    """
    from chimerax.markers import MarkerSet

    ms = MarkerSet(session, name=name)
    session.models.add([ms])

    for i, (axis, color) in enumerate(zip(axes, colors), start=1):
        q = np.asarray(center) + 0.55 * length * np.asarray(axis)

        # MarkerSet colors use RGBA 0-255.
        marker_color = {
            "yellow": (255, 255, 0, 255),
            "cyan": (0, 255, 255, 255),
            "magenta": (255, 0, 255, 255),
        }[color]

        a = ms.create_marker(
            tuple(q),
            marker_color,
            0.01,
            id=i,
        )

        # MarkerSet markers are atoms, so they can receive ordinary 3-D labels.
        # Label text is set via the ChimeraX command interface below.
        mspec = f"#{ms.id_string}:{i}"
        from chimerax.core.commands import run
        run(
            session,
            f'label {mspec} atoms text "{i}" '
            f'height 8.000 color {color}'
        )

    return ms


def draw_atomic_axes(session, structure, symmetry):
    """
    Draw estimated symmetry axes for a PDB/mmCIF atomic model.
    """
    s_upper = symmetry.upper()

    if s_upper[0] not in ("C", "D"):
        raise ValueError("Symmetry must be Cn or Dn.")

    n = int(s_upper[1:])
    if n < 2:
        raise ValueError("n must be >= 2.")

    center, axes, vals = inertia_axes(structure)

    length = axis_length(structure)

    radius = max(0.5, 0.012 * length)

    if s_upper.startswith("C") or (s_upper.startswith("D") and n != 2):
        # Find the pair of nearly equal transverse moments.  The remaining
        # eigenvector is the central Cn/Dn axis.
        pair = min(
            ((abs(vals[0] - vals[1]), 0, 1, 2),
             (abs(vals[0] - vals[2]), 0, 2, 1),
             (abs(vals[1] - vals[2]), 1, 2, 0)),
            key=lambda x: x[0],
        )
        principal = axes[pair[3]]

        p1 = center - 0.5 * length * principal
        p2 = center + 0.5 * length * principal

        s = draw_cylinder(
            session, p1, p2, radius, "yellow",
            f"SYMAXES: #{structure.id_string} {s_upper} estimated symmetry axis"
        )

        group_symmetry_graphics(
            session, structure, [s],
            f"SYMAXES: #{structure.id_string} {s_upper}",
            symmetry=s_upper,
            axes_scene=[principal],
            center_scene=center,
        )

        session.logger.warning(
            f"{s_upper}: axis estimated from atomic coordinates/inertia. "
            f"For an exact assembly axis, use map symmetry operations or "
            f"provide assembly operators."
        )

        return s_upper

    # D2: all three principal axes.
    colors = ["yellow", "cyan", "magenta"]

    surfaces = []
    for i, (axis, color) in enumerate(zip(axes, colors), start=1):
        p1 = center - 0.5 * length * axis
        p2 = center + 0.5 * length * axis

        s = draw_cylinder(
            session, p1, p2, radius, color,
            f"SYMAXES: #{structure.id_string} D2 estimated axis {i}"
        )
        surfaces.append(s)

    labels = create_label_markers(
        session,
        center,
        axes,
        colors,
        length,
        name=f"SYMAXES: #{structure.id_string} D2 labels",
    )

    group_symmetry_graphics(
        session, structure, surfaces + [labels],
        f"SYMAXES: #{structure.id_string} D2",
        symmetry="D2",
        axes_scene=axes,
        center_scene=center,
    )

    session.logger.warning(
        "D2: axes estimated from the atomic coordinates/inertia tensor. "
        "For exact assembly-derived axes, use explicit assembly operations."
    )

    return "D2"


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def runsymaxes(session, model, symmetry=None):
    """
    Usage:

        runsymaxes #1
        runsymaxes #1 C5
        runsymaxes #1 D2
    """
    if isinstance(model, Volume):
        # Ignore an optional symmetry string when a map already has
        # actual symmetry operations.  This ensures we draw what the map
        # says, not a second arbitrary coordinate-system definition.
        surfaces, detected = draw_map_axes(session, model)
        session.logger.info(
            f"Displayed symmetry axes for {model.name}: {detected}"
        )
        return

    if isinstance(model, AtomicStructure):
        if symmetry is None:
            raise ValueError(
                "For an atomic PDB/mmCIF model, specify Cn or Dn, e.g. "
                "'runsymaxes #1 D2'.  Automatic recovery of assembly operators "
                "will be added separately."
            )

        detected = draw_atomic_axes(session, model, symmetry)
        session.logger.info(
            f"Displayed symmetry axes for {model.name}: {detected}"
        )
        return

    raise ValueError(
        "runsymaxes requires a volume map or atomic structure."
    )


def register_command(session):
    desc = CmdDesc(
        required=[("model", ModelArg)],
        optional=[("symmetry", StringArg)],
        synopsis="Draw the actual Cn/Dn symmetry axis/axes",
    )
    register("symaxes", desc, runsymaxes)

symfit_desc = CmdDesc(
    required=[("specification", RestOfLine)],
    synopsis="Rigidly align two complete SYMAXES assemblies",
)
register("symfit", symfit_desc, symfit)


register_command(session)
