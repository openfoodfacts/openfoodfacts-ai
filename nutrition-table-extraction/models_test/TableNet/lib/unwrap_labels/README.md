# Unwrap Labels
## 

The algorithm to unwrap labels using edge markers

## How it works

A wine label, being stick to a cylinder surface, is distorted, which reduces quality of at least two
operations with it:

* Image matching (product search by a picture)
* Recognition of the text on the label

The accuracy of those operations can be highly improved if the label is unwrapped, and the current library
provides an easy to use algorithm for that.

A cylinder distortion of the label on a picture makes it look like a barrel (see an example below):
![original image](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/original.jpg)

The following schema shows, how that happens:
![schema](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/label_schema.png)


The thing is, to revert the distortion, it's only required to specify 6 markers:
![markers image](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/corner-points.jpg)

The library creates a mesh, which will be transformed into a plane:
![mesh](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/original-with-mesh.jpg)

The unwrapped image is below. There are two ways to unapply transform - with or without
interpolation. The interpolated method generates a better image but requires to have
scipy dependency installed (it might be an issue if you use Amazon lambda, which has limited
library size):

![unwrapped](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/unwrapped.jpg)


## Unwrapping + Stitching
See our AI based service for auto unwrapping and stitching [https://perfectlabel.io](https://bit.ly/3cnwu0D)
