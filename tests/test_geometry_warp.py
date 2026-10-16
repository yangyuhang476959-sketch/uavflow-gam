import numpy as np
from experiments.uavflow_predictor_idm.geometry_warp import (
    forward_warp_zdepth, heading_actions_to_camera_transform,
    pixel_support_to_patch_ratio, transport_patch_geometry,
)


def camera(size=28):
    return np.array([[size/2,0,(size-1)/2],[0,size/2,(size-1)/2],[0,0,1]],float)


def test_zero_action_identity_and_patch_alignment():
    depth=np.full((28,28),4.)
    out=forward_warp_zdepth(depth,camera(),heading_actions_to_camera_transform(np.zeros((5,4))))
    assert out['support'].all()
    np.testing.assert_allclose(out['depth'],depth)
    geometry=np.arange(12,dtype=np.float32).reshape(4,3)
    transported,rho=transport_patch_geometry(geometry,out['source_index'])
    np.testing.assert_allclose(transported,geometry)
    np.testing.assert_allclose(rho,1)


def test_support_is_area_ratio_not_binary_resize():
    mask=np.zeros((28,28));mask[:7,:14]=1
    np.testing.assert_allclose(pixel_support_to_patch_ratio(mask),[[.5,0],[0,0]])


def test_positive_heading_yaw_moves_center_feature_left():
    depth=np.full((28,28),4.);mask=np.zeros_like(depth,dtype=bool);mask[14,14]=True
    transform=heading_actions_to_camera_transform([[0,0,0,.15]])
    out=forward_warp_zdepth(depth,camera(),transform,mask)
    y,x=np.where(out['support'])
    assert len(x)==1 and x[0]<14


def test_near_points_have_larger_translation_parallax():
    depth=np.zeros((28,28));depth[10,14]=2;depth[18,14]=10
    out=forward_warp_zdepth(depth,camera(),heading_actions_to_camera_transform([[0,.5,0,0]]))
    locations={int(out['source_index'][y,x]):int(x) for y,x in zip(*np.where(out['support']))}
    assert 14-locations[10*28+14]>14-locations[18*28+14]>0


def test_zbuffer_closest_point_wins():
    depth=np.zeros((28,28));depth[14,14]=2;depth[14,15]=4
    transform=np.eye(4);transform[0,3]=4/14
    out=forward_warp_zdepth(depth,camera(),transform)
    assert out['support'].sum()==1
    assert out['depth'][14,16]==2
    assert out['source_index'][14,16]==14*28+14


def test_forward_motion_expands_near_feature_more():
    depth=np.zeros((28,28));depth[10,18]=2;depth[18,18]=10
    out=forward_warp_zdepth(depth,camera(),heading_actions_to_camera_transform([[1,0,0,0]]))
    locations={int(out['source_index'][y,x]):int(x) for y,x in zip(*np.where(out['support']))}
    assert locations[10*28+18]-18>locations[18*28+18]-18>=0
