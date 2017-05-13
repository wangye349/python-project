#this project is used to realize surf algorithm
#so this project is gonna to finish by the following steps
#np.size(img_array_after_gassian,1)
from PIL import Image
import numpy as np
def gassian_function(input_file_name,output_file_name):
    img = Image.open(input_file_name)
    img_array = np.array(img)
    img_array_for_gassian = [[]]
    print img_array[0][0]
    for i in range(np.size(img_array,0)):
        for j in range(np.size(img_array,1)):
            if(j%3 == 0):
                img_array_for_gassian[i].append(img_array[i][j])
        img_array_for_gassian.append([])
    del img_array_for_gassian[np.size(img_array_for_gassian)-1]
    img_array_after_gassian_temp = np.copy(img_array_for_gassian)
    for i in range(1,np.size(img_array,0)-1):
        for j in range(1,np.size(img_array_for_gassian,1)-1):
            img_array_after_gassian_temp[i][j] = (1.5*img_array_for_gassian[i][j] + 2*img_array_for_gassian[i][j-1]\
            + 2*img_array_for_gassian[i][j+1] + 2*img_array_for_gassian[i-1][j] + 2*img_array_for_gassian[i+1][j]\
            + 2*img_array_for_gassian[i-1][j-1] + 2*img_array_for_gassian[i-1][j+1] + 2*img_array_for_gassian[i+1][j-1]\
            + 2*img_array_for_gassian[i+1][j+1])/17.5

    img_array_after_gassian_before = np.zeros((np.size(img_array,0),np.size(img_array,1)))
    #img_array_after_gassian_temp = img_array_after_gassian_temp.tolist()
    print np.size(img_array_after_gassian_before,0),np.size(img_array_after_gassian_before,1)
    print np.size(img_array_after_gassian_temp,0),np.size(img_array_after_gassian_temp,1)
    print img_array_after_gassian_temp[0][0],img_array_after_gassian_temp[0][1]
    for i in range(np.size(img_array,0)):
        for j in range(np.size(img_array,1)):
            label = int(j/3)
            # print label
            # print img_array_after_gassian_temp[i][label]
            img_array_after_gassian_before[i][j] = img_array_after_gassian_temp[i][label][0]

    img_array_after_gassian = img_array_after_gassian_before.tolist()
    img_array_after_gassian = np.array(img_array_after_gassian)
    img_array_finish_gassian = np.uint8(img_array_after_gassian)
    img_last = Image.fromarray(img_array_finish_gassian)
    img_last.save(output_file_name)

def integral_image(input_array):
    output_array = np.zeros((np.size(input_array,0)+1,np.size(input_array,1)+1))
    for i in range(np.size(input_array,0)):
        for j in range(np.size(input_array,1)):
            output_array[i][j]=output_array[i][j-1]+output_array[i-1][j]-output_array[i-1][j-1]+input_array[i][j]
    output_array_temp = output_array.tolist()
    print output_array_temp
    del output_array_temp[np.size(output_array_temp,0)-1]
    print output_array_temp
    numbles = np.size(output_array_temp,1)-1
    for i in range(np.size(output_array_temp,0)):
        del output_array_temp[i][numbles]#np.size(output_array_temp,1)-1]
    output_array = np.array(output_array_temp)
    return output_array

def use_for_multiply(image_array,matrix_size,x_label,y_label):
    output_array_temp = np.zeros((matrix_size,matrix_size))
    x_begin = x_label-int(matrix_size/2)
    y_begin = y_label-int(matrix_size/2)
    print x_begin
    print y_begin
    for i in range(matrix_size):
        for j in range(matrix_size):
            output_array_temp[i][j] = image_array[x_begin+i-1][y_begin+j-1]
    return output_array_temp

def from_image_to_array(file_name):
    img = Image.open(file_name)
    #print img
    img_array = np.array(img)
    #print img_array
    output_array_r = np.zeros((np.size(img_array,0),np.size(img_array,1)))
    output_array_g = np.copy(output_array_r)
    output_array_b = np.copy(output_array_g)
    for i in range(np.size(img_array,0)):
        for j in range(np.size(img_array,1)):
            output_array_r[i][j] = img_array[i][j][0]
            output_array_g[i][j] = img_array[i][j][1]
            output_array_b[i][j] = img_array[i][j][2]
    print img_array
    return output_array_r,output_array_g,output_array_b
    #print img_array[0][0]

    #print np.size(img_array,0)
    #print np.size(img_array,1)
    #print np.size(img_array,2)
    #print img_array
    
def fro_array_to_image(input_array,output_file_name):
    print np.size(input_array,0), np.size(input_array,1), np.size(input_array,2)
    print input_array[0]
    print input_array[1]
    print input_array[2]
    output_array = np.zeros((np.size(input_array,1),np.size(input_array,2),3))
    for i in range(np.size(input_array,1)):
        for j in range(np.size(input_array,2)):
            output_array[i][j][0] = input_array[0][i][j]
            output_array[i][j][1] = input_array[1][i][j]
            output_array[i][j][2] = input_array[2][i][j]
    print np.size(output_array,0),np.size(output_array,1),np.size(output_array,2)
    print output_array
    img_array = np.uint8(output_array)
    img_last = Image.fromarray(img_array)
    img_last.save(output_file_name)
    return output_array

def hessian_matrix(matrix_size,input_array):

    #integral_image_array = integral_image(input_array)
    GXX = np.zeros((matrix_size,matrix_size))
    GXY = np.zeros((matrix_size,matrix_size))
    GXX_side_ve = int(matrix_size/3)
    print GXX_side_ve
    GXX_side_ho = int(matrix_size/6.0 + 0.5)
    print GXX_side_ho
    GXY_rec_size = int(matrix_size/3)
    GXY_side_len = int((matrix_size - 2*GXY_rec_size -1)/2)
    GXY_middle = int(matrix_size/2)+1
    for i in range(GXX_side_ho,np.size(GXX,0)-GXX_side_ho):
        for j in range(np.size(GXX,1)):
            if(GXX_side_ve <= j <= 2*GXX_side_ve-1):
                GXX[i][j] = -2
            else:
                GXX[i][j] = 1
    print GXX.T
    for i in range(GXY_side_len,matrix_size - GXY_side_len):
        for j in range(GXY_side_len,matrix_size - GXY_side_len):
            if(i!=GXY_middle-1) and (j!=GXY_middle-1):
                if(GXY_side_len<=i<GXY_rec_size+GXY_side_len)and(GXY_side_len<=j<GXY_rec_size+GXY_side_len):
                    GXY[i][j] = 1
                elif(GXY_side_len+GXY_rec_size+1<=i<matrix_size-GXY_side_len)and(GXY_side_len+GXY_rec_size+1<=j<matrix_size-GXY_side_len):
                    GXY[i][j] = 1
                else:
                    GXY[i][j] = -1
    print GXY.T

    output_array_temp1 = np.copy(input_array)
    output_array_temp2 = np.copy(input_array)
    print use_for_multiply(input_array,9,4,4)
    print np.multiply(use_for_multiply(input_array,9,8,8),GXX.T)
    for i in range(int(matrix_size/2),np.size(input_array,0)-int(matrix_size/2)):
        for j in range(int(matrix_size/2),np.size(input_array,1)-int(matrix_size/2)):
            output_array_temp1[i][j] = np.sum(np.multiply(use_for_multiply(input_array,matrix_size,i,j),GXX.T))
            output_array_temp2[i][j] = np.sum(np.multiply(use_for_multiply(input_array,matrix_size,i,j),GXY.T))
    output_array = np.copy(input_array)
    for i in range(np.size(input_array,0)):
        for j in range(np.size(input_array,1)):
            output_array[i][j] = output_array_temp1[i][j]*output_array_temp2[i][j]-0.81*output_array_temp2[i][j]*output_array_temp2[i][j]

    return output_array
