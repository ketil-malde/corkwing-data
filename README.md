# Download and organize corkwing image data for segmentation

This component copies images with polygon masks in XML format as
generetated by CVAT, and from a list of XML annotation files, produces
directories "images" and "segmentation_masks", as well as a file
"annotations.csv" linking the image files and containing bounding
boxes for the masks.
