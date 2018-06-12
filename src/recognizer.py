"""An Optical Characters Recognizer
should be able to recognize characters on vehicle license plates.
"""
import abc

class Recognizer(object):
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def clear(self):
        """Clear anything everything.
        """
        return


    @abc.abstractmethod
    def recognize(self, plate_image):
        """Recognize based on plate image.
        Args:
            plate_image: plate image to be recognized
        Returns:
            should return recognized characters
        """
        return


    @abc.abstractmethod
    def recognize_batch(self, plate_images):
        """Recognize based on plate images.
        Args:
            plate_images: an array of plate_images to be recognized
        Returns:
            should return an array of recognized characters
        """
        return


    # NOTE: temporarily commented out
    # def recognize_in_subplot(self, image, xmin, ymin, xmax, ymax):
    #     """Recognize based on organic image and square boundary box info.
    #     Args:
    #         image: organic image
    #         xmin: x position of plate left top point
    #         ymin: y position of plate left top point
    #         xmax: x position of plate right bottom point
    #         ymax: y position of plate right bottom point
    #     Returns:
    #         should return recognized characters
    #     """
    #     plate_image = image[ymin: ymax, xmin: xmax, :]
    #     return self.recognize(plate_image)
