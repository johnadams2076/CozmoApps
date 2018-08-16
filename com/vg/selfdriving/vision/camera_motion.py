#!/usr/bin/env python3

# Copyright (c) 2017 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Demonstrate the manual and auto exposure settings of Cozmo's camera.

This example demonstrates the use of auto exposure and manual exposure for
Cozmo's camera. The current camera settings are overlayed onto the camera
viewer window.
'''
import PIL
import asyncio
import sys
import time

import cv2
from cozmo import util

try:
    from PIL import ImageDraw, ImageFont
    import numpy as np
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')

import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

from  LaneFinder  import pipeline

import matplotlib.pyplot as plt

# A global string value to display in the camera viewer window to make it more
# obvious what the example program is currently doing.
example_mode = ""

# Define an annotator using the annotator decorator
@cozmo.annotate.annotator
def clock(image, scale, annotator=None, world=None, **kw):
    d = ImageDraw.Draw(image)
    bounds = (0, 0, image.width, image.height)
    text = cozmo.annotate.ImageText(time.strftime("%H:%m:%S"),
            position=cozmo.annotate.TOP_RIGHT)
    text.render(d, bounds)

# Define another decorator as a subclass of Annotator
class Battery(cozmo.annotate.Annotator):
    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)
        batt = self.world.robot.battery_voltage
        text = cozmo.annotate.ImageText('BATT %.1fv' % batt, color='green')
        text.render(d, bounds)


# An annotator for live-display of all of the camera info on top of the camera
# viewer window.
@cozmo.annotate.annotator
def camera_info(image, scale, annotator=None, world=None, **kw):
    d = ImageDraw.Draw(image)
    bounds = [3, 0, image.width, image.height]

    camera = world.robot.camera
    text_to_display = "Example Mode: " + example_mode + "\n\n"
    text_to_display += "Fixed Camera Settings (Calibrated for this Robot):\n\n"
    text_to_display += 'focal_length: %s\n' % camera.config.focal_length
    text_to_display += 'center: %s\n' % camera.config.center
    text_to_display += 'fov: <%.3f, %.3f> degrees\n' % (camera.config.fov_x.degrees,
                                                        camera.config.fov_y.degrees)
    text_to_display += "\n"
    text_to_display += "Valid exposure and gain ranges:\n\n"
    text_to_display += 'exposure: %s..%s\n' % (camera.config.min_exposure_time_ms,
                                               camera.config.max_exposure_time_ms)
    text_to_display += 'gain: %.3f..%.3f\n' % (camera.config.min_gain,
                                               camera.config.max_gain)
    text_to_display += "\n"
    text_to_display += "Current settings:\n\n"
    text_to_display += 'Auto Exposure Enabled: %s\n' % camera.is_auto_exposure_enabled
    text_to_display += 'Exposure: %s ms\n' % camera.exposure_ms
    text_to_display += 'Gain: %.3f\n' % camera.gain
    color_mode_str = "Color" if camera.color_image_enabled else "Grayscale"
    text_to_display += 'Color Mode: %s\n' % color_mode_str

    text = cozmo.annotate.ImageText(text_to_display,
                                    position=cozmo.annotate.TOP_LEFT,
                                    line_spacing=2,
                                    color="white",
                                    outline_color="black", full_outline=True)
    text.render(d, bounds)


def demo_camera_exposure(robot: cozmo.robot.Robot):
    global example_mode

    # Ensure camera is in auto exposure mode and demonstrate auto exposure for 5 seconds
    camera = robot.camera
    camera.enable_auto_exposure()
    example_mode = "Auto Exposure"
    #time.sleep(5)

    # Demonstrate manual exposure, linearly increasing the exposure time, while
    # keeping the gain fixed at a medium value.
    #example_mode = "Manual Exposure - Increasing Exposure, Fixed Gain"
    fixed_gain = (camera.config.min_gain + camera.config.max_gain) * 0.5
    #for exposure in range(camera.config.min_exposure_time_ms, camera.config.max_exposure_time_ms+1, 1):
        #camera.set_manual_exposure(exposure, fixed_gain)
        #time.sleep(0.1)
        # Drive forwards for 150 millimeters at 50 millimeters-per-second.
        #robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()

        # Turn 90 degrees to the left.
        # Note: To turn to the right, just use a negative number.
        #robot.turn_in_place(degrees(90)).wait_for_completed()

    # Demonstrate manual exposure, linearly increasing the gain, while keeping
    # the exposure fixed at a relatively low value.
    #example_mode = "Manual Exposure - Increasing Gain, Fixed Exposure"
    #fixed_exposure_ms = 10
    #for gain in np.arange(camera.config.min_gain, camera.config.max_gain, 0.05):
        #camera.set_manual_exposure(fixed_exposure_ms, gain)
       # time.sleep(0.1)
        # Drive forwards for 150 millimeters at 50 millimeters-per-second.
        #robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()

        # Turn 90 degrees to the left.
        # Note: To turn to the right, just use a negative number.
        #robot.turn_in_place(degrees(90)).wait_for_completed()

    # Switch back to auto exposure, demo for a final 5 seconds and then return
    camera.enable_auto_exposure()
    example_mode = "Mode: Auto Exposure"
    time.sleep(5)
def cozmo_program(robot: cozmo.robot.Robot):
    CamMotion(robot)


class CamMotion:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    video = cv2.VideoWriter('video100.avi', fourcc, 20.0, (320, 120))

    def __init__(self, robot: cozmo.robot.Robot):
        self.robot = robot
        #if self.robot.world.robot.battery_voltage < 3.6:
        #    self.drive_to_charger()
        #    self.robot.backup_onto_charger(self, 3)

        ##robot.world.image_annotator.add_annotator('camera_info', camera_info)
        self.robot.world.image_annotator.add_annotator('battery', Battery)
        self.robot.world.image_annotator.add_annotator('clock', clock)
        self.robot.world.image_annotator.annotation_enabled = True
        # Demo with default grayscale camera images
        #robot.camera.color_image_enabled = False
        #demo_camera_exposure(robot)

        # Demo with color camera images
        self.robot.camera.color_image_enabled = True

        self.robot.camera.image_stream_enabled = True

        self.robot.add_event_handler(cozmo.world.EvtNewCameraImage, self.on_new_camera_image)
        self.run_on_track()
        cv2.destroyAllWindows()
        self.video.release()
        #test_run_on_track(robot)
        #demo_camera_exposure(robot)

    def on_new_camera_image(self, evt, **kwargs):
        '''Processes the blobs in Cozmo's view, and determines the correct reaction.'''
        highres_image = self.robot.world.latest_image.raw_image
        opencvimage = np.array(highres_image.resize((320,240), resample = PIL.Image.LANCZOS).convert('RGB'))
        #opencvimage = np.array(highres_image.convert('RGB'))
        #print('highres_image',opencvImage)
        weightedopencvimage = pipeline(opencvimage)
        #print(weightedopencvimage)
        #plt.imshow(weightedopencvimage)
        self.video.write(weightedopencvimage)
        #superimposedimage = PIL.Image.fromarray(weightedopencvimage)


    def drive_to_charger(self):

        '''The core of the drive_to_charger program'''

        # If the robot was on the charger, drive them forward and clear of the charger
        if self.robot.is_on_charger:
            # drive off the charger
            self.robot.drive_off_charger_contacts().wait_for_completed()
            self.robot.drive_straight(distance_mm(100), speed_mmps(50)).wait_for_completed()
            # Start moving the lift down
            self.robot.move_lift(-3)
            # turn around to look at the charger
            self.robot.turn_in_place(degrees(180)).wait_for_completed()
            # Tilt the head to be level
            self.robot.set_head_angle(degrees(0)).wait_for_completed()
            # wait half a second to ensure Cozmo has seen the charger
            time.sleep(0.5)
            # drive backwards away from the charger
            self.robot.drive_straight(distance_mm(-60), speed_mmps(50)).wait_for_completed()

        # try to find the charger
        charger = None

        # see if Cozmo already knows where the charger is
        if self.robot.world.charger:
            if self.robot.world.charger.pose.is_comparable(self.robot.pose):
                print("Cozmo already knows where the charger is!")
                charger = self.robot.world.charger
            else:
                # Cozmo knows about the charger, but the pose is not based on the
                # same origin as the robot (e.g. the robot was moved since seeing
                # the charger) so try to look for the charger first
                pass

        if not charger:
            # Tell Cozmo to look around for the charger
            look_around = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
            try:
                charger = self.robot.world.wait_for_observed_charger(timeout=30)
                print("Found charger: %s" % charger)
            except asyncio.TimeoutError:
                print("Didn't see the charger")
            finally:
                # whether we find it or not, we want to stop the behavior
                look_around.stop()

        if charger:
            # Attempt to drive near to the charger, and then stop.
            action = self.robot.go_to_object(charger, distance_mm(65.0))
            action.wait_for_completed()
            print("Completed action: result = %s" % action)
            print("Done.")

    def run_on_track(self):
        self.robot.set_head_angle(util.degrees(0), duration=10.0).wait_for_completed(30)
        self.robot.set_lift_height(1.0).wait_for_completed(30)

        # T1 - 610
        # T2 - 535
        # T3 - 620
        # T4 - 530
        # Drive forwards for 150 millimeters at 50 millimeters-per-second.
        self.robot.drive_straight(distance_mm(615), speed_mmps(50)).wait_for_completed(30)

        # Turn 90 degrees to the left.
        # Note: To turn to the right, just use a negative number.
        self.robot.turn_in_place(degrees(-90)).wait_for_completed(30)

        self.robot.drive_straight(distance_mm(530), speed_mmps(50)).wait_for_completed(30)
        self.robot.turn_in_place(degrees(-90)).wait_for_completed(30)

        self.robot.drive_straight(distance_mm(610), speed_mmps(50)).wait_for_completed(30)
        self.robot.turn_in_place(degrees(-90)).wait_for_completed(30)

        self.robot.drive_straight(distance_mm(535), speed_mmps(50)).wait_for_completed(30)
        self.robot.turn_in_place(degrees(-90)).wait_for_completed(30)

    def test_run_on_track(robot):
        action1 = robot.set_head_angle(util.degrees(0), duration=10.0)
        action2 = robot.set_lift_height(1.0, in_parallel=True)
        action1.wait_for_completed()
        action2.wait_for_completed()


cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger for this example
cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
