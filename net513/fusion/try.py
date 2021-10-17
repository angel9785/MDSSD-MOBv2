import numpy as np
import math
LAMR=74747
fppi=300/(2402)
print(fppi)
mr=(1-0.83138)
print(0.82138*1506)
fppi_tmp = np.insert(fppi,0,-1)
mr_tmp = np.insert(mr,0,1)
ref=np.logspace(-1.0,0,num=9)
print(ref[1:])
print(ref)
ref=ref[1:]
# ref=ref[1:10]
# ref=10. ^ (-2:.25:0)
# for i,ref_i in enumerate(ref):
#     j=np.where(fppi_tmp<= ref_i)[-1][-1]
#     ref[i]=mr_tmp[j]
for i,ref_i in enumerate(ref):
    j=np.where(fppi_tmp<= ref_i)[-1][-1]
    # if ref_i==0.1:
        # x=mr_tmp[i]
    ref[i] = mr_tmp[j]
print(ref)
lamr=math.exp(np.mean(np.log(np.maximum(1e-10,ref))))
print(lamr)