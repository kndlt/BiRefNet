"""
Purpose:

Input is a image that has just been background matted.
But the matted image contains white seams around the edges.
Or it could be black/red/magenta seams around the deges.
I want to remove those seams.

As an input, we provide:
1. Seam's width in pixels.

Then the code will change the RGBs of those seam pixels to match the nearby non-seam pixels.

"""

