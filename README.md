# Optical_flow

function remind

from optical_flow.mian imort optical_flow

__para:__  
__window:__ the window size, should not be too small  
__stride:__ the stride of window, control the density of optical flow  
__grad_th:__ filter the small grad  
__low:__ the low threshold for filtering the noise  
__high:__ the high threshold for filtering the noise  
__arrow_length:__ the length of arrow  

#### you should change the para above to adjust different imgs.

<img width="600" height="350" src="https://github.com/Wu-Xiuchao/Optical_flow/blob/master/example.png"/>

test video：  
```
python demo.py [input video] [output video]
```

__Note:__  
__input video:__  if -1,it means to use the first camera, else it is the path of input video  
__output video:__  the save path of the output video  
