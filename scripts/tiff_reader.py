import tifffile as tiff


img = "/home/mitchell/SERDP_WS/serdp_calibration/images/sonar/sonar0.tiff"
sonar_img = tiff.imread(img)
for i in range(sonar_img.shape[0]):
    for j in range(sonar_img.shape[1]):
        print("img", sonar_img[i,j])
