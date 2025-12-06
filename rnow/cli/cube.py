# reinforcenow/cli/cube.py
"""ASCII 3D rotating cube animation for loading states.

Inspired by https://github.com/leonmavr/retrovoxel
Voxel-style cube rendering with depth-based shading.
"""

import contextlib
import math
import os
import sys
import threading
import time

# Display settings
WIDTH = 50
HEIGHT = 18
FRAME_DELAY = 0.04

# ReinforceNow teal: #14B8A6
TEAL_RGB = (20, 184, 166)
TEAL = "\033[38;2;20;184;166m"
RESET = "\033[0m"

# Cube faces with shading characters (front to back)
CHARS = ".,-~:;=!*#$@"


class VoxelCube:
    """Voxel-style 3D cube renderer."""

    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.A = 0  # Rotation angle X
        self.B = 0  # Rotation angle Y
        self.C = 0  # Rotation angle Z
        self.cube_size = 8
        self.distance = 40
        self.k1 = 20  # Projection constant

    def calculate_x(self, i, j, k):
        """Calculate rotated X coordinate."""
        return (
            j * math.sin(self.A) * math.sin(self.B) * math.cos(self.C)
            - k * math.cos(self.A) * math.sin(self.B) * math.cos(self.C)
            + j * math.cos(self.A) * math.sin(self.C)
            + k * math.sin(self.A) * math.sin(self.C)
            + i * math.cos(self.B) * math.cos(self.C)
        )

    def calculate_y(self, i, j, k):
        """Calculate rotated Y coordinate."""
        return (
            j * math.cos(self.A) * math.cos(self.C)
            + k * math.sin(self.A) * math.cos(self.C)
            - j * math.sin(self.A) * math.sin(self.B) * math.sin(self.C)
            + k * math.cos(self.A) * math.sin(self.B) * math.sin(self.C)
            - i * math.cos(self.B) * math.sin(self.C)
        )

    def calculate_z(self, i, j, k):
        """Calculate rotated Z coordinate."""
        return (
            k * math.cos(self.A) * math.cos(self.B)
            - j * math.sin(self.A) * math.cos(self.B)
            + i * math.sin(self.B)
        )

    def render_frame(self):
        """Render a single frame of the rotating cube."""
        output = [[" " for _ in range(self.width)] for _ in range(self.height)]
        z_buffer = [[0 for _ in range(self.width)] for _ in range(self.height)]

        # Render each face of the cube
        cube_size = self.cube_size
        increment = 0.6

        for face in range(6):
            # Generate points for each face
            if face == 0:  # Front
                points = [
                    (i, j, -cube_size)
                    for i in self._frange(-cube_size, cube_size, increment)
                    for j in self._frange(-cube_size, cube_size, increment)
                ]
                char_idx = 0
            elif face == 1:  # Back
                points = [
                    (i, j, cube_size)
                    for i in self._frange(-cube_size, cube_size, increment)
                    for j in self._frange(-cube_size, cube_size, increment)
                ]
                char_idx = 2
            elif face == 2:  # Left
                points = [
                    (-cube_size, j, k)
                    for j in self._frange(-cube_size, cube_size, increment)
                    for k in self._frange(-cube_size, cube_size, increment)
                ]
                char_idx = 4
            elif face == 3:  # Right
                points = [
                    (cube_size, j, k)
                    for j in self._frange(-cube_size, cube_size, increment)
                    for k in self._frange(-cube_size, cube_size, increment)
                ]
                char_idx = 6
            elif face == 4:  # Top
                points = [
                    (i, -cube_size, k)
                    for i in self._frange(-cube_size, cube_size, increment)
                    for k in self._frange(-cube_size, cube_size, increment)
                ]
                char_idx = 8
            else:  # Bottom
                points = [
                    (i, cube_size, k)
                    for i in self._frange(-cube_size, cube_size, increment)
                    for k in self._frange(-cube_size, cube_size, increment)
                ]
                char_idx = 10

            for i, j, k in points:
                x = self.calculate_x(i, j, k)
                y = self.calculate_y(i, j, k)
                z = self.calculate_z(i, j, k) + self.distance

                ooz = 1 / z if z != 0 else 0

                xp = int(self.width / 2 + self.k1 * ooz * x * 2)
                yp = int(self.height / 2 + self.k1 * ooz * y)

                if 0 <= xp < self.width and 0 <= yp < self.height and ooz > z_buffer[yp][xp]:
                    z_buffer[yp][xp] = ooz
                    output[yp][xp] = CHARS[char_idx % len(CHARS)]

        return "\n".join("".join(row) for row in output)

    def _frange(self, start, stop, step):
        """Float range generator."""
        vals = []
        val = start
        while val < stop:
            vals.append(val)
            val += step
        return vals

    def rotate(self, da=0.04, db=0.02, dc=0.01):
        """Update rotation angles."""
        self.A += da
        self.B += db
        self.C += dc


class CubeSpinner:
    """Animated ASCII 3D rotating cube spinner."""

    def __init__(self):
        self.cube = VoxelCube()
        self.running = False
        self.thread = None
        self.message = ""
        self.lines_printed = 0

        # Enable Windows ANSI support
        if os.name == "nt":
            os.system("")

    def _animation_loop(self):
        """Main animation loop."""
        first_frame = True
        while self.running:
            frame = self.cube.render_frame()
            self.cube.rotate()

            # Strip trailing newlines to get accurate line count
            frame = frame.rstrip("\n")
            colored_frame = f"{TEAL}{frame}{RESET}"

            # Count actual lines in this frame
            lines_this_frame = frame.count("\n") + 1

            # Move back to top of previous frame (no clearing - new frame overwrites old)
            if not first_frame and self.lines_printed > 0:
                sys.stdout.write(f"\033[{self.lines_printed}A")
                sys.stdout.write("\r")

            # Print the new frame with newline so cursor ends below
            sys.stdout.write(colored_frame + "\n")
            sys.stdout.flush()

            self.lines_printed = lines_this_frame
            first_frame = False
            time.sleep(FRAME_DELAY)

    def start(self, message=""):
        """Start the animation."""
        self.message = message
        self.running = True
        self.lines_printed = 0
        self.thread = threading.Thread(target=self._animation_loop, daemon=True)
        self.thread.start()

    def update_message(self, message):
        """Update the message."""
        self.message = message

    def stop(self, keep_visible=False):
        """Stop the animation.

        Args:
            keep_visible: If True, leave the cube frozen in place.
                         If False (default), clear the cube from output.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.1)

        if keep_visible:
            # Leave the cube visible, just move cursor below it
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif self.lines_printed > 0:
            # Cursor is currently one line below the cube
            # Move back up to the first line of the cube
            sys.stdout.write(f"\033[{self.lines_printed}A")
            sys.stdout.write("\r")

            # Clear exactly self.lines_printed lines (not the whole screen)
            for i in range(self.lines_printed):
                sys.stdout.write("\033[2K")  # Clear current line only
                if i < self.lines_printed - 1:
                    sys.stdout.write("\n")

            # Move cursor back up to where the first line was
            if self.lines_printed > 1:
                sys.stdout.write(f"\033[{self.lines_printed - 1}A")
            sys.stdout.write("\r")
            sys.stdout.flush()


if __name__ == "__main__":
    spinner = CubeSpinner()
    spinner.start("Uploading...")
    with contextlib.suppress(KeyboardInterrupt):
        time.sleep(3)
    spinner.stop(keep_visible=True)
    print(f"Run started successfully {TEAL}✅{RESET}")
    print("  Project: My Project")
    print("  Model: Qwen/Qwen3-8B")
    print("  Dataset: train.jsonl")
    print()
    print("View your experiment here:")
    print(f"{TEAL}https://reinforcenow.ai/run/abc123{RESET}")
