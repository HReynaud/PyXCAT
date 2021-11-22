import argparse
import SimpleITK as sitk
import numpy as np
from PIL import Image

def clip(v, vmin, vmax):
    return min(max(v, vmin), vmax)

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', '--file', type=str, help='Path of the input .nii.gz file.')
    parser.add_argument('-o', '--output', type=str, default='plane.png', help='Output file path and name')
    
    parser.add_argument('-A', '--pointA', type=int, nargs=3, help="Point A relative x, y, z coordinates. Ex: 'x y z'")
    parser.add_argument('-B', '--pointB', type=int, nargs=3, help="Point B relative x, y, z coordinates. Ex: 'x y z'")
    parser.add_argument('-C', '--pointC', type=int, nargs=3, help="Point C relative x, y, z coordinates. Ex: 'x y z'")


    return parser.parse_args()

class Point():
    """
    A class to represent a point in 3D space, defined by a shape.
    """

    def __init__(self, x, y, z, shape=(256, 256, 256)):
        """
        Initialize a point with given x, y, z in relative coordinates space [0-1]

        :param x: x coordinate in relative coordinates space [0-1]
        :param y: y coordinate in relative coordinates space [0-1]
        :param z: z coordinate in relative coordinates space [0-1]

        :param shape: shape of the volume where the points live
        """
        self._x = x
        self._y = y
        self._z = z
        self.max_x = shape[0]
        self.max_y = shape[1]
        self.max_z = shape[2]
    
    def get_relative_coords(self):
        """
        :return: The relative coordinates of the point
        """
        return np.array([self._x, self._y, self._z])
    
    def get_absolute_coords(self):
        """
        :return: The absolute coordinates of the point in their volume space
        """
        return np.array([self.x, self.y, self.z], dtype=np.int)
    
    def __str__(self):
        """
        :return: A string representation of the point
        """
        return "P("+str(np.round(self._x,3))+","+str(np.round(self._y,3))+","+str(np.round(self._z,3))+")"

    @property
    def x(self):
        return self._x*self.max_x
    
    @property
    def y(self):
        return self._y*self.max_y
    
    @property
    def z(self):
        return self._z*self.max_z

class PlaneSampler():
    """
    A class to sample a 2D plane defined by three points from a 3D volume.
    """
    def __init__(self, volume, normalise=True):
        self.volume = volume
        self.normalise = normalise

        # Tracked values, used as optional outputs
        self.coefs = None
        self.plane = None
        self.oob_count = None
        self.coordinates = None
        self.main_ax = None

    def _get_plane_coefs(self, pointA, pointB, pointC):
        """
        Get the coefficients of the plane defined by three points.
        """
        # Get the vector from point A to B
        AB = np.array(pointB.get_relative_coords()) - np.array(pointA.get_relative_coords())
        # Get the vector from point A to C
        AC = np.array(pointC.get_relative_coords()) - np.array(pointA.get_relative_coords())
        # Get the normal vector of the plane
        normal = np.cross(AB, AC)
        # Get the coefficients of the plane
        coefs = np.array([normal[0], normal[1], normal[2], -np.dot(normal, pointA.get_relative_coords())])
        
        if self.normalise:
            # Set the sign of the first coefficient to always be positive
            coefs *= 1 if np.sign(coefs[0])>=0 else -1
            # Normalize the coefficients to have a unit normal vector, using Frobenius norm
            norm = np.linalg.norm(coefs)
            coefs = coefs / (norm if norm != 0 else 1)
        
        self.coefs = coefs
    
    def _get_plane(self, coefs):
        """
        Get the plane defined by a set of coefficients.
        """
        # Get the main axis of the plane, d is excluded 
        main_ax = np.argmax(np.abs(coefs[:-1]))

        # Reorder values depending on the main axis - to avoid code repetition
        sa, sb, sc = np.roll(self.volume.shape, shift=(2-main_ax))
        ca, cb, cc, d = *np.roll(coefs[:-1], shift=(2-main_ax)), coefs[-1]

        # Compute the index maps A, B, C
        A,B = np.meshgrid(np.arange(sa), np.arange(sb), indexing='ij')
        C = np.rint( (d - ca * A - cb * B) / cc ).astype(np.int)

        # Take care of out of bounds values
        P = C.copy()
        S = sc-1
        C[C <= 0]  = 0
        C[C >= sc] = sc-1

        # Reorder ABC into the correct XYZ
        # Use indices to avoid weird behavior of np.roll on list of arrays
        ABC = np.stack((A, B, C))
        idxX, idxY, idxZ = np.roll([0, 1, 2],shift=(main_ax-2))
        X, Y, Z = ABC[idxX], ABC[idxY], ABC[idxZ]

        # Get the indices of the points in the plane
        plane = self.volume[X, Y, Z]
        plane[P < 0] = 0
        plane[P > S] = 0

        # Register output values
        self.plane = plane
        self.oob_count = plane[P < 0].shape[0] + plane[P > S].shape[0]
        self.coordinates = X,Y,Z
        self.main_ax = main_ax
    
    def __call__(self, pointA, pointB, pointC):
        """
        Call the sampling function. For additional return values, see the class attributes.

        :param pointA: Point A of the plane
        :param pointB: Point B of the plane
        :param pointC: Point C of the plane

        :return: The sampled plane
        """
        self._get_plane_coefs(pointA, pointB, pointC)
        self._get_plane(self.coefs)
        return self.plane


if __name__ == '__main__':

    # Parse arguments
    args = get_args()
    print(args)

    # Load the volume
    image = sitk.ReadImage(args.file)
    # image = sitk.ReadImage('/vol/biomedic3/hjr119/XCAT_clean/volumes/TEST_1_CT.nii.gz')
    
    data = sitk.GetArrayFromImage(image)
    data = data/data.max() * 255

    PointA = Point(*args.pointA, data.shape)
    PointB = Point(*args.pointB, data.shape)
    PointC = Point(*args.pointC, data.shape)

    sampler = PlaneSampler(data)

    plane = sampler(PointA, PointB, PointC)

    Image.fromarray(plane.astype(np.uint8)).save(args.output)

    # print(plane.shape, plane.min(), plane.max())




