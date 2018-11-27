#!/usr/bin/env bash

scenflow_data_path=/home/wpf/data/data_sf

monkaa_frames_cleanpass=$scenflow_data_path"/monkaa_frames_cleanpass"
monkaa_disparity=$scenflow_data_path"/monkaa_disparity"
driving_frames_cleanpass=$scenflow_data_path"/driving_frames_cleanpass"
driving_disparity=$scenflow_data_path"/driving_disparity"
flyingthings3d_frames_cleanpass=$scenflow_data_path"/frames_cleanpass"
flyingthings3d_disparity=$scenflow_data_path"/frames_disparity"

mkdir dataset

ln -s $monkaa_frames_cleanpass dataset/monkaa_frames_cleanpass
ln -s $monkaa_disparity dataset/monkaa_disparity
ln -s $flyingthings3d_frames_cleanpass dataset/frames_cleanpass
ln -s $flyingthings3d_disparity dataset/frames_disparity
ln -s $driving_frames_cleanpass dataset/driving_disparity
ln -s $driving_disparity dataset/driving_frames_cleanpass

