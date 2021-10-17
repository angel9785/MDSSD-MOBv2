import scipy.io
mat = scipy.io.loadmat('/media/fereshteh/8653-3589/MultispectralPedestrianDetection-master/CVPRW2017/CVPR2017_Koenig_FusionRPNBF_Detections_KAIST_Reasonable.mat')
print(mat.__sizeof__())
print(mat.__len__())
print((mat["dt"][0][10][1]))
l=0
for i in range(0,2251):
    l=l+len(mat["dt"][0][i])
print(l)