"""An detector should be able to localize an object,
either a vehicle or a vehicle license plate.
"""
import abc


class Detector(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def clear(self):
        """Clear anything everything."""
        return

    @abc.abstractmethod
    def detect(self, image, threshold=None):
        """Detect desired objects in the image.
        Args:
            image: image to be detected
            threshold: threshold to filter images
        Returns:
            should return detected boundary boxes
        """
        return

    def detect_in_subplot(self, image, xmin, ymin, xmax, ymax, threshold=None):
        """Detect desired objects in the boundary box of an image.
        Args:
            image: organic image
            xmin: x position of sub-image left top point
            ymin: y position of sub-image left top point
            xmax: x position of sub-image right bottom point
            ymax: y position of sub-image right bottom point
            threshold: threshold to filter images
        Returns:
            should return detected boundary boxes
        """
        cropped_image = image[ymin: ymax, xmin: xmax, :]
        if threshold:
            boxes = self.detect(cropped_image, threshold)
        else:
            boxes = self.detect(cropped_image)

        return [(box[0]+xmin, box[1]+ymin, box[2]+xmin, box[3]+ymin)
                for box in boxes]

    def detect_with_threshold(self, images, threshold=None):
        """Detect desired objects in a group of images.
        Args:
            images: a group of images to be detected
            threshold: threshold to filter images
        Returns:
            should return detected boundary boxes according to the images
        """
        if threshold:
            return [self.detect(image, threshold) for image in images]
        else:
            return [self.detect(image) for image in images]
