import colorsys
import time

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch

from datasets.preprocess.utils import to_numpy


def visualize_4D_point_clouds_open3D(points, colors, timestamps):
    """
    points:   tensor or array of shape (T, ..., 3)
    colors:   tensor or array of shape (T, ..., 3), values in [0,1]
    timestamps: 1D array or list of length T (in seconds)
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("4D Point Cloud Viewer", 1024, 768)

    # Prepare a PointCloud with the first frame
    pcd = o3d.geometry.PointCloud()
    pts = points[0].reshape(-1, 3)
    cols = colors[0].reshape(-1, 3)
    # if these are torch tensors, convert to numpy
    if hasattr(pts, "cpu"):
        pts = pts.cpu().numpy()
        cols = cols.cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    vis.add_geometry(pcd)

    # Render options
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([0, 0, 0])
    render_opt.point_size = 2.0

    # Initial camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat(np.mean(pts, axis=0).tolist())
    ctr.set_up([0, -1, 0])

    # Animate through all frames
    for i in range(1, len(timestamps)):
        pts = points[i].reshape(-1, 3)
        cols = colors[i].reshape(-1, 3)
        if hasattr(pts, "cpu"):
            pts = pts.cpu().numpy()
            cols = cols.cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        # Sleep according to the time difference
        dt = float(timestamps[i] - timestamps[i - 1])
        time.sleep(dt)

    # Keep the window alive (so you can still interact) until you close it
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    vis.destroy_window()


def visualize_4d_point_clouds(point_clouds, colors, timestamps, app_id="4d_pointcloud_demo"):
    # Initialize a new Rerun recording (retains UI state thanks to app_id) and launch the viewer
    rr.init(app_id)  # :contentReference[oaicite:0]{index=0}
    rr.spawn()  # :contentReference[oaicite:1]{index=1}

    # Stream each frame
    for pts, cols, t in zip(point_clouds, colors, timestamps):
        loc_pts = pts.reshape(-1, 3)  # Reshape to (N, 3)
        loc_cols = cols.reshape(-1, 3)  # Reshape to (N, 3)
        # loc_cols = loc_cols * 255  # Convert colors to 0-255 range

        # Set our custom time for this frame
        rr.set_time("frame_time", duration=t)  # :contentReference[oaicite:2]{index=2}

        # Log the colored points into the scene under the path "scene/points"
        rr.log(
            "scene/points",
            rr.Points3D(loc_pts, colors=loc_cols, radii=0.01),
        )  # :contentReference[oaicite:3]{index=3}

    # Create a Spatial3D view to display the points.
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            name="3D Scene",
            # Set the background color to light blue.
            # Configure the line grid.
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True,  # The grid is enabled by default, but you can hide it with this property.
                spacing=0.1,  # Makes the grid more fine-grained.
                # By default, the plane is inferred from view coordinates setup, but you can set arbitrary planes.
                plane=rr.components.Plane3D.XY.with_distance(-5.0),
                stroke_width=2.0,  # Makes the grid lines twice as thick as usual.
                color=[255, 255, 255, 128],  # Colors the grid a half-transparent white.
            ),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)


# # Optionally, block here until the Viewer is closed
# rr.block()
def visualize_4d_point_clouds_with_segmentation(
        point_clouds,
        colors,
        timestamps,
        segmentation_masks=None,
        dynamic_masks=None,
        app_id="4d_pointcloud_segmentation_demo",
        hsv_tuples=None,
        class_colors=None,
):
    if hsv_tuples is None and class_colors is None:
        hsv_tuples = [(i / 37.0, 1.0, 1.0) for i in range(37)]
        class_colors = np.array([
            colorsys.hsv_to_rgb(h, s, v) for (h, s, v) in hsv_tuples
        ], dtype=np.float32)  # shape (37,3), values in [0,1]

    # Initialize a new Rerun recording (retains UI state thanks to app_id) and launch the viewer
    rr.init(app_id)  # :contentReference[oaicite:0]{index=0}
    rr.spawn()  # :contentReference[oaicite:1]{index=1}

    # Stream each frame
    for pts, cols, t in zip(point_clouds, colors, timestamps):
        loc_pts = pts.reshape(-1, 3)  # Reshape to (N, 3)
        loc_cols = cols.reshape(-1, 3).clone()  # Reshape to (N, 3)

        # Change the color of the points based on the segmentation mask such that
        # Those points can be clearly seen in the viewer.
        # For example, if the segmentation mask is 1, overlay a red color on the points.
        if segmentation_masks is not None:
            print(f"segmentation_masks shape: {segmentation_masks.shape}")
            seg_mask = segmentation_masks[t].reshape(-1)  # (N,)
            non_bg = seg_mask != 0
            if non_bg.sum() > 0:
                loc_cols[non_bg] = torch.Tensor(class_colors[seg_mask[non_bg]])

        if dynamic_masks is not None:
            print(f"Dynamic_masks shape: {dynamic_masks.shape}")
            dyn_mask = dynamic_masks[t].reshape(-1)  # (N,)
            # Set the color to white for dynamic masks
            dyn_mask_non_bg = dyn_mask != 0
            loc_cols[dyn_mask_non_bg] = torch.Tensor([1.0, 1.0, 1.0])

        # Set our custom time for this frame
        rr.set_time("frame_time", duration=t)  # :contentReference[oaicite:2]{index=2}

        # Log the colored points into the scene under the path "scene/points"
        rr.log(
            "scene/points",
            rr.Points3D(loc_pts, colors=loc_cols, radii=0.01),
        )  # :contentReference[oaicite:3]{index=3}

    # Create a Spatial3D view to display the points.
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="/",
            name="3D Scene",
            # Set the background color to light blue.
            # Configure the line grid.
            line_grid=rrb.archetypes.LineGrid3D(
                visible=True,  # The grid is enabled by default, but you can hide it with this property.
                spacing=0.1,  # Makes the grid more fine-grained.
                # By default, the plane is inferred from view coordinates setup, but you can set arbitrary planes.
                plane=rr.components.Plane3D.XY.with_distance(-5.0),
                stroke_width=2.0,  # Makes the grid lines twice as thick as usual.
                color=[255, 255, 255, 128],  # Colors the grid a half-transparent white.
            ),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)


def visualize_static_scene_with_segmentation(
        point_clouds,
        colors,
        timestamps,
        segmentation_masks=None,
        dynamic_masks=None,
        app_id="static_scene_segmentation_demo"
):
    rr.init(app_id)  # :contentReference[oaicite:0]{index=0}
    rr.spawn()  # :contentReference[oaicite:1]{index=1}

    bg_pts_list = []
    bg_pt_color_list = []

    for pts, cols, t in zip(point_clouds, colors, timestamps):
        loc_pts = pts.reshape(-1, 3)  # Reshape to (N, 3)
        loc_cols = cols.reshape(-1, 3).clone()  # Reshape to (N, 3)

        # Change the color of the points based on the segmentation mask such that
        # Those points can be clearly seen in the viewer.
        # For example, if the segmentation mask is 1, overlay a red color on the points.
        seg_mask = segmentation_masks[t].reshape(-1)  # (N,)
        bg = torch.Tensor(seg_mask == 0)

        dyn_mask = dynamic_masks[t].reshape(-1)  # (N,)
        dyn_mask_bg = torch.tensor(dyn_mask == 0)

        # add the dynamic mask to the segmentation mask
        mask = torch.logical_or(bg, dyn_mask_bg)

        bg_pts = loc_pts[mask]
        bg_cols = loc_cols[mask]

        bg_pts_list.append(bg_pts)
        bg_pt_color_list.append(bg_cols)

    all_bg_pts = torch.cat(bg_pts_list, dim=0)
    all_bg_cols = torch.cat(bg_pt_color_list, dim=0)

    # Log the colored points into the scene under the path "scene/points"
    rr.log(
        "scene/points",
        rr.Points3D(all_bg_pts, colors=all_bg_cols, radii=0.01),
    )
