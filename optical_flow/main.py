import cv2
import numpy as np 
from scipy import signal
import math


""" 光流函数
参数:
	L1,L2 均为t1,t2图像
	window 为窗口大小
	stride 步长
	grad_th 梯度阈值
	low 为低阈值
	high 为高阈值
返回:
	一张绘制好光流的图
"""
def optical_flow(L1_img,L2_img,window=5,stride=20,grad_th=1e-2,low=1e-2,high=10,arrow_length=6):

	L1 = cv2.cvtColor(L1_img,cv2.COLOR_BGR2GRAY)
	L2 = cv2.cvtColor(L2_img,cv2.COLOR_BGR2GRAY)

	L1 = np.array(L1)/255.
	L2 = np.array(L2)/255.

	pad = window // 2

	kernel_x = np.array([[-1.,1.],[-1.,1.]])
	kernel_y = np.array([[-1.,-1.],[1.,1.]])
	kernel_t = np.array([[1.,1.],[1.,1.]])

	fx = signal.convolve2d(L1,kernel_x,boundary='symm',mode='same')
	fy = signal.convolve2d(L1,kernel_y,boundary='symm',mode='same')
	ft = signal.convolve2d(L2,kernel_t,boundary='symm',mode='same') + signal.convolve2d(L1,-kernel_t,boundary='symm',mode='same')

	for i in np.arange(pad, L1.shape[0]-pad, stride):
		for j in np.arange(pad, L1.shape[1]-pad, stride):

			if math.sqrt(fx[i,j]**2+fy[i,j]**2) < grad_th:
				continue
			Ix = fx[i-pad:i+pad+1, j-pad:j+pad+1].reshape((-1,1))
			Iy = fy[i-pad:i+pad+1, j-pad:j+pad+1].reshape((-1,1))
			It = ft[i-pad:i+pad+1, j-pad:j+pad+1].reshape((-1,1))

			A = np.hstack([Ix,Iy])
			b = -1 * It

			ATA = np.matmul(A.T,A)
			try:
				inverse = np.linalg.inv(ATA)
			except:
				continue
			else:
				eigvals = abs(np.linalg.eigvals(ATA))
				if np.min(eigvals) < low or np.max(eigvals)/np.min(eigvals) > high:
					continue
				u,v = np.squeeze(np.matmul(np.matmul(inverse,A.T),b))
				old = (j,i)
				new = (int(j+arrow_length*u), int(i+arrow_length*v))
				cv2.arrowedLine(L2_img,new,old,(0,0,255),1,cv2.LINE_AA)

	return L2_img







